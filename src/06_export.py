# -*- coding: utf-8 -*-
"""
06 导出预测界面 —— 把调好参的模型搬进一个自包含的网页。

网页要在浏览器里直接算出预测值，所以不能调用 Python。做法是把拟合好的
Ridge 管线**拆成参数**导出（中位数、均值、标准差、one-hot 类别、系数、截距），
再用 web/predict.js 在 JS 里重放同一套变换。因为模型是线性的，
这个过程是**精确**的，不是近似 —— 但前提是两边算的必须是同一件事，
所以本脚本最后会调 node 把全部 43,213 行数据整表跑一遍，逐行比对 JS 与
Python 的预测值，差值超过 1e-6 就报错。

为什么网页里部署的是 Ridge 而不是 04 里 CV 得分更高的 Ridge+Poly2：
    见 fit_final 的注释，差值只有 +0.018，但要拿平方项去接用户随便填的数
    并不划算。

关于数据量：43,213 行不可能整表塞进网页（散点图就是四万个 SVG 圆点）。
所以这一节的思路是 —— **能精确的都在 Python 侧算准，只有散点才抽样**：
    直方图          全部 43,213 行，每个区县一条，一个点都不抽
    分箱中位数线     全部 43,213 行按十分位分箱
    拟合线 / r 值    全部 43,213 行
    散点            抽样约 1,400 行，按区县分层，页面上标明「抽样」

产出：
    web/model.json              模型参数（同时也是 node 校验的输入）
    web/index.html              自包含网页（模板 + 内联的模型和数据）
    results/app_test_cases.csv  校验用的样例，人可读
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from common import (CATEGORICAL, NUMERIC, RAW_CSV, RES, ROOT, SEED, TARGET,
                    WINDOW_YEARS, guard_no_leakage, load_clean, make_cv,
                    make_preprocessor, rmse)

WEB = ROOT / "web"
TEMPLATE = WEB / "template.html"
INDEX = WEB / "index.html"
MODEL_JSON = WEB / "model.json"

ALPHAS = np.logspace(-3, 3, 25)
TEST_SIZE = 0.2

# 直方图：单价 2 万 ~ 15.2 万，每箱 4 千。这个区间盖住了 99.9% 的房源。
HIST_LO, HIST_BIN, HIST_N = 20_000, 4_000, 33

# 散点抽样：按区县分层，大区县按比例抽，小区县保底 25 个点（否则门头沟只有 8 个点，
# 图上那一片就空了，看着像模型没覆盖到）。
SAMPLE_TOTAL, SAMPLE_MIN = 1_400, 25

# 分箱中位数线的箱数（按分位数切，每箱样本量相同）
BIN_Q = 10

# 表单的取值范围。这不是数据本身的 min/max，是**给用户划的安全区**：
# 线性模型在训练数据覆盖之外是外推，输入越极端越不可信，
# 所以宁可把滑块卡在观测范围内，也不要给出一个看起来很精确的外推值。
FORM_RANGES = {
    "面积": (10, 400), "房龄": (0, 70), "总层数": (1, 50),
    "梯户比": (0.0, 2.0), "关注人数": (0, 500),
}

# 没有真实房源时的默认房源（朝阳一套两居，取数据集中位数附近），
# 让网页一打开就是「算过一次」的状态，而不是空表。
DEFAULT_INPUT = {
    "区县": "朝阳", "面积": 89.0, "室": 2, "厅": 1, "卫": 1,
    "总层数": 18, "房龄": 15, "梯户比": 0.3, "关注人数": 45,
    "电梯": 1, "满五": 1, "近地铁": 1,
    "楼层位置": "中", "装修": "精装", "建筑类型": "板楼", "建筑结构": "结构6",
}


def jsonable(v):
    """NaN / NaT → None，其余原样。JSON 没有 NaN 字面量。"""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def fit_final(df: pd.DataFrame):
    """在**全部** 43,213 行上重新拟合最终模型。

    04/05 里的指标（CV R²、留出集 R²）描述的是泛化能力，那是在训练集上评估的。
    真正拿去用的模型没理由只用 80% 的数据，所以这里用全量重拟合。
    两者不是同一个对象，README 里说清楚了。

    这里部署的是 Ridge，不是 04 里 CV 更高的 Ridge+Poly2。差值是
    CV R² +0.678 → +0.696、RMSE 13,668 → 13,306 元/㎡（约 0.6%），
    但代价是把平方项交给用户随便填的数：二次函数在训练区间之外会掉头向下，
    「关注人数填 500」这种输入能算出负单价。网页的输入域是开放的，
    所以这一分精度不值得拿外推风险去换。实测见 README 第八节。
    """
    X = df[[c for c in df.columns if c != TARGET]]
    y = df[TARGET]
    guard_no_leakage(X, y)

    # 重新搜一次 alpha，而不是把 05 的结果硬编码进来 ——
    # 硬编码的数字早晚会和 05 跑出来的对不上。
    def pipe():
        return Pipeline([("pre", make_preprocessor()), ("model", Ridge())])

    gs = GridSearchCV(pipe(), {"model__alpha": ALPHAS}, cv=make_cv(),
                      scoring="r2", error_score="raise", n_jobs=-1)
    gs.fit(X, y)
    alpha = float(gs.best_params_["model__alpha"])

    # 泛化指标：留出集一次，供网页如实标注误差
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    hold = pipe().set_params(model__alpha=alpha).fit(X_tr, y_tr)
    holdout_r2 = float(hold.score(X_te, y_te))
    holdout_rmse = rmse(y_te, hold.predict(X_te))

    final = pipe().set_params(model__alpha=alpha).fit(X, y)
    print(f"  alpha 网格搜索最优 = {alpha:.6g}   CV R² = {gs.best_score_:+.4f}")
    print(f"  留出集 R² = {holdout_r2:+.4f}   RMSE = {holdout_rmse:,.0f} 元/㎡")
    return final, alpha, float(gs.best_score_), holdout_r2, holdout_rmse


def extract_model(pipe: Pipeline, alpha, cv_r2, holdout_r2, holdout_rmse, df) -> dict:
    """把 Pipeline 拆成可以喂给 JS 的参数。"""
    pre = pipe.named_steps["pre"]
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    model = pipe.named_steps["model"]

    impute_median = np.asarray(num.named_steps["imp"].statistics_, dtype=float)
    scaler_mean = np.asarray(num.named_steps["sc"].mean_, dtype=float)
    scaler_scale = np.asarray(num.named_steps["sc"].scale_, dtype=float)
    cat_fill = [str(v) for v in cat.named_steps["imp"].statistics_]
    categories = [[str(x) for x in arr] for arr in cat.named_steps["oh"].categories_]
    coef = np.asarray(model.coef_, dtype=float)

    n_expected = len(NUMERIC) + sum(len(c) for c in categories)
    assert len(coef) == n_expected, (
        f"系数个数 {len(coef)} 与「数值列 + one-hot 列」的维度 {n_expected} 对不上，"
        "说明 JS 侧的特征拼接顺序会错位")
    assert np.isfinite(coef).all() and np.isfinite(model.intercept_)

    return {
        "meta": {
            "model": f"Ridge(alpha={alpha:.4g})",
            "alpha": alpha,
            "trained_on": int(len(df)),
            "cv_r2": round(cv_r2, 4),
            "holdout_r2": round(holdout_r2, 4),
            "rmse": round(holdout_rmse, 1),
            "target": TARGET,
            "unit": "元/㎡",
            "median_all": float(df[TARGET].median()),
            "mean_all": float(df[TARGET].mean()),
            "std_all": float(df[TARGET].std()),
            "p10": float(df[TARGET].quantile(0.10)),
            "p90": float(df[TARGET].quantile(0.90)),
            "seed": SEED,
        },
        "numeric_cols": NUMERIC,
        "cat_cols": CATEGORICAL,
        # 一律按 float64 原样导出，不做任何四舍五入。
        # 试过把 scaler 留 6 位小数、系数留 10 位有效数字「让 JSON 好看点」，
        # 结果 one-hot 系数动辄上万，这点截断误差乘上去能到 1e-2 元/㎡ ——
        # node 校验第一轮就是这么被抓出来的。json.dumps 对 float 输出 repr，
        # 是能精确往返的最短表示，没必要再自己截。
        "impute_median": [float(v) for v in impute_median],
        "scaler_mean": [float(v) for v in scaler_mean],
        "scaler_scale": [float(v) for v in scaler_scale],
        "cat_fill": cat_fill,
        "categories": categories,
        "coef": [float(v) for v in coef],
        "intercept": float(model.intercept_),
    }


def _eta2(df: pd.DataFrame, group: str, target: str) -> float:
    """组间方差 / 总方差 —— 分组变量对目标的一元解释力。"""
    grand = df[target].mean()
    vc = df[group].value_counts()
    ssb = sum(n * (df.loc[df[group] == k, target].mean() - grand) ** 2 for k, n in vc.items())
    return float(ssb / ((df[target] - grand) ** 2).sum())


def _provenance() -> dict:
    """原始数据的规模与年份跨度。

    页脚要写「原始共 N 条，跨 X–Y 年」，这两个数字得从数据里读出来 ——
    写死在模板里的话，哪天换了数据集，页面说的就不是它自己画的那份数了。
    只读 tradeTime 一列：全表 59 MB，usecols 让 pandas 跳过其余列的解析。
    """
    t = pd.read_csv(RAW_CSV, usecols=["tradeTime"], encoding="utf-8",
                    low_memory=False)["tradeTime"].astype(str).str[:4]
    years = pd.to_numeric(t, errors="coerce").dropna()
    return {"raw_rows": int(len(t)), "raw_year_min": int(years.min()),
            "raw_year_max": int(years.max()), "window_years": list(WINDOW_YEARS)}


def _trend(x: pd.Series, y: pd.Series) -> dict:
    m = x.notna() & y.notna()
    k, b = np.polyfit(x[m], y[m], 1)
    return {"slope": float(k), "intercept": float(b),
            "r": float(np.corrcoef(x[m], y[m])[0, 1]), "n": int(m.sum())}


def _binned(x: pd.Series, y: pd.Series, q: int = BIN_Q) -> dict:
    """按分位数把 x 切成 q 箱，取每箱 y 的中位数。每箱样本量相同。

    用分位数而不是等宽切，是因为面积、房龄都是右偏的：等宽切会让
    「300㎡ 以上」那一箱只剩几十套，中位数抖得没法看。
    """
    m = x.notna() & y.notna()
    xs, ys = x[m].to_numpy(dtype=float), y[m].to_numpy(dtype=float)
    edges = np.unique(np.quantile(xs, np.linspace(0, 1, q + 1)))
    idx = np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)
    centers, meds, counts = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if not sel.any():
            continue
        # 横坐标取箱内的**中位面积**，不是箱边界的中点。
        # 面积右偏得厉害：最右一箱是 (114, 1000]，中点 557 落在几乎没有房源的地方，
        # 而该箱实际中位面积只有 170 左右 —— 用中点画线等于把这条线甩到空处。
        # y 取的是中位数，x 也取中位数，两者才是同一件事的两个维度。
        centers.append(float(np.median(xs[sel])))
        meds.append(float(np.median(ys[sel])))
        counts.append(int(sel.sum()))
    # p01/p99 是给横轴用的，不是给数据用的：面积最大值 1,000㎡，
    # 但 99% 的房源在 250㎡ 以内，按最大值画横轴会把绝大多数点挤成左边一坨。
    return {"x": centers, "y": meds, "n": counts,
            "lo": float(edges[0]), "hi": float(edges[-1]),
            "p01": float(np.quantile(xs, 0.01)), "p99": float(np.quantile(xs, 0.99))}


def extract_data(df: pd.DataFrame) -> dict:
    """导出画图要用的统计量 + 一份分层抽样的散点。"""
    d = df.copy()

    # 区县按房源数从多到少排，下拉框里常用的排前面
    order = d["区县"].value_counts()
    districts = list(order.index)
    didx = {name: i for i, name in enumerate(districts)}

    # ---- 直方图：全部房源，一个点都不抽 ----
    edges = HIST_LO + np.arange(HIST_N + 1) * HIST_BIN
    assert d[TARGET].max() < edges[-1], "有房源超出直方图范围，最后一箱会少算"
    hist_all, _ = np.histogram(d[TARGET], bins=edges)
    hist_by = []
    for name in districts:
        h, _ = np.histogram(d.loc[d["区县"] == name, TARGET], bins=edges)
        hist_by.append([int(v) for v in h])

    stat = d.groupby("区县")[TARGET].agg(["median", "size"])

    # ---- 分层抽样：只有散点图用它 ----
    parts = []
    for name in districts:
        sub = d[d["区县"] == name]
        quota = max(SAMPLE_MIN, round(SAMPLE_TOTAL * len(sub) / len(d)))
        step = max(1, len(sub) // quota)
        parts.append(sub.iloc[::step])
    sample = pd.concat(parts)
    rows = [[round(float(r["面积"]), 1), jsonable(r["房龄"]),
             didx[r["区县"]], round(float(r[TARGET]), 0)] for _, r in sample.iterrows()]

    # ---- 分箱中位数、拟合线：全部房源 ----
    age_trend_by = []
    for name in districts:
        sub = d[d["区县"] == name]
        age_trend_by.append(_trend(sub["房龄"], sub[TARGET]) if len(sub) >= 30 else None)

    cards = {c: sorted(d[c].dropna().unique().tolist())
             for c in CATEGORICAL if c != "区县"}

    return {
        "districts": districts,
        "district_count": [int(stat.loc[n, "size"]) for n in districts],
        "district_median": [float(round(stat.loc[n, "median"])) for n in districts],
        "district_eta2": round(_eta2(d, "区县", TARGET), 4),
        "provenance": _provenance(),
        "hist": {"lo": HIST_LO, "bin": HIST_BIN, "all": [int(v) for v in hist_all],
                 "by_district": hist_by},
        "sample": rows,
        "sample_note": f"抽样 {len(rows):,} 套（共 {len(d):,} 套，按区县分层）",
        "area_bins": _binned(d["面积"], d[TARGET]),
        "age_bins": _binned(d["房龄"], d[TARGET]),
        "area_trend": _trend(d["面积"], d[TARGET]),
        "age_trend": _trend(d["房龄"], d[TARGET]),
        "age_trend_by_district": age_trend_by,
        "cats": cards,
        "rooms": sorted(int(v) for v in d["室"].dropna().unique()),
        "halls": sorted(int(v) for v in d["厅"].dropna().unique()),
        "baths": sorted(int(v) for v in d["卫"].dropna().unique()),
        "ranges": {k: list(v) for k, v in FORM_RANGES.items()},
        "default_input": DEFAULT_INPUT,
    }


def render(model: dict, data: dict) -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    for token, payload in [("__PREDICT_JS__", (WEB / "predict.js").read_text(encoding="utf-8")),
                           ("__MODEL_JSON__", json.dumps(model, ensure_ascii=False,
                                                         separators=(",", ":"))),
                           ("__DATA_JSON__", json.dumps(data, ensure_ascii=False,
                                                        separators=(",", ":")))]:
        assert token in html, f"模板里缺少占位符 {token}"
        html = html.replace(token, payload.replace("</script", "<\\/script"))
    INDEX.write_text(html, encoding="utf-8")
    print(f"  已生成 {INDEX.relative_to(ROOT)}  （{INDEX.stat().st_size / 1024:.0f} KB）")


def verify_page_script() -> bool:
    """用 node 把生成的 index.html 里的内联脚本编译一遍，只查语法。

    为什么需要这一步：verify_with_node 验的是 web/predict.js 里的**预测函数**，
    而页面脚本是另一份代码。页面脚本哪怕只是多写了一个右括号，预测函数照样
    能通过全部 43,213 行的比对，网页打开却是一张空表 —— 这个坑实测踩过一次：
    模板里多了个 `}`，标题、下拉框、数字全空，而 node 校验打印的是 ✓。
    语法错误只需要编译、不需要 DOM，所以 vm.Script 就够，不必上 headless 浏览器。
    """
    node = shutil.which("node")
    if node is None:
        print("  [跳过] 没找到 node，无法检查页面脚本的语法。")
        return False

    html = INDEX.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert blocks, "index.html 里一个内联 <script> 都没有，模板的占位符可能没被替换"

    js = f"""
const vm = require('vm');
const blocks = {json.dumps(blocks, ensure_ascii=False)};
let bad = 0;
blocks.forEach((code, i) => {{
  try {{ new vm.Script(code, {{filename: 'script#' + i}}); }}
  catch (e) {{ bad++; console.error('  脚本块 ' + i + ' 语法错误：' + e.message); }}
}});
console.log(JSON.stringify({{blocks: blocks.length, bad}}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "syntax.js"
        script.write_text(js, encoding="utf-8")
        out = subprocess.run([node, str(script)], capture_output=True,
                             text=True, encoding="utf-8")
    if out.returncode != 0:
        print("  [失败] node 语法检查脚本本身报错：")
        print(out.stderr.strip()[:2000])
        return False

    res = json.loads(out.stdout.strip().splitlines()[-1])
    if res["bad"]:
        print(out.stderr.strip())
        print(f"  页面脚本 {res['blocks']} 个块，{res['bad']} 个编译不过 —— 网页会是一张空表")
        return False
    print(f"  页面脚本 {res['blocks']} 个块全部通过语法检查  ✓")
    return True


def verify_with_node(model: dict, pipe: Pipeline, df: pd.DataFrame) -> bool:
    """让 node 用 web/predict.js 把整表跑一遍，和 Python 逐行对答案。"""
    node = shutil.which("node")
    if node is None:
        print("  [跳过] 没找到 node，无法自动比对 JS 与 Python 的预测值。")
        print("         网页逻辑未经验证 —— 装了 node 再跑一次本脚本即可。")
        return False

    X = df[[c for c in df.columns if c != TARGET]]
    py_pred = pipe.predict(X)
    cases = [{c: jsonable(v) for c, v in row.items()} for row in X.to_dict("records")]

    # 期望值同样按 float64 原样传过去：这里若做 round(…, 6)，
    # 光是「期望值自己」就带了 5e-7 的误差，会把真正的差异淹没。
    js = f"""
const {{predictUnitPrice}} = require({json.dumps(str(WEB / 'predict.js'))});
const model = require({json.dumps(str(MODEL_JSON))});
const cases = {json.dumps(cases, ensure_ascii=False)};
const expected = {json.dumps([float(v) for v in py_pred])};
let worst = 0, worstAt = -1, worstRel = 0;
cases.forEach((c, i) => {{
  const got = predictUnitPrice(model, c);
  const d = Math.abs(got - expected[i]);
  if (d > worst) {{ worst = d; worstAt = i; worstRel = d / Math.abs(expected[i]); }}
}});
console.log(JSON.stringify({{worst, worstAt, worstRel, n: cases.length}}));
"""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "verify.js"
        script.write_text(js, encoding="utf-8")
        out = subprocess.run([node, str(script)], capture_output=True,
                             text=True, encoding="utf-8")
    if out.returncode != 0:
        print("  [失败] node 校验脚本报错：")
        print(out.stderr.strip()[:2000])
        return False

    res = json.loads(out.stdout.strip().splitlines()[-1])
    # 容差 1e-6 元/㎡：单价是 5~6 位数，这个绝对容差相当于相对误差 ~1e-11，
    # 已经贴着 float64 的精度极限了。网页把价格显示到整数位，所以这点差别
    # 连显示的最后一位都影响不到。
    ok = res["worst"] <= 1e-6
    print(f"  JS 与 Python 逐行比对 {res['n']:,} 行：最大差值 {res['worst']:.3e} 元/㎡"
          f"（相对 {res['worstRel']:.1e}）{'  ✓ 一致' if ok else '  ✗ 超出容差'}")
    if not ok:
        print(f"    最大差值出现在第 {res['worstAt']} 行")
    assert ok, "网页的预测值与 Python 不一致，不能发布"
    return True


def main() -> int:
    print("=" * 68)
    print("06 导出预测界面")
    print("=" * 68)

    df = load_clean()
    data = extract_data(df)

    n = len(df)
    print(f"\n拟合最终模型（全部 {n:,} 行）：")
    pipe, alpha, cv_r2, holdout_r2, holdout_rmse = fit_final(df)

    model = extract_model(pipe, alpha, cv_r2, holdout_r2, holdout_rmse, df)
    MODEL_JSON.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n导出数据与渲染网页：")
    print(f"  散点抽样 {len(data['sample']):,} 行（{data['sample_note']}）")
    print(f"  直方图用全部 {n:,} 行，{len(data['hist']['all'])} 箱；"
          f"{len(data['districts'])} 个区县各一条")
    print(f"  {len(model['coef'])} 个特征系数")
    render(model, data)

    print("\n校验网页：")
    syntax_ok = verify_page_script()
    verified = verify_with_node(model, pipe, df)
    assert syntax_ok, "页面脚本有语法错误，网页会是一张空表"

    # 留一份人可读的样例，方便手工核对
    X = df[[c for c in df.columns if c != TARGET]].head(12).copy()
    X["Python预测单价"] = np.round(pipe.predict(X), 2)
    X.to_csv(RES / "app_test_cases.csv", index=False, encoding="utf-8-sig")

    print(f"\n完成：双击 {INDEX.relative_to(ROOT)} 即可使用"
          f"{'（已通过 node 校验）' if verified else '（注意：未经 node 校验）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
