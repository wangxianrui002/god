# -*- coding: utf-8 -*-
"""
06 导出预测界面 —— 把调好参的模型搬进一个自包含的网页。

网页要在浏览器里直接算出预测值，所以不能调用 Python。做法是把拟合好的
Ridge 管线**拆成参数**导出（中位数、均值、标准差、one-hot 类别、系数、截距），
再用 web/predict.js 在 JS 里重放同一套变换。因为模型是线性的，
这个过程是**精确**的，不是近似 —— 但前提是两边算的必须是同一件事，
所以本脚本最后会调 node 把 306 行数据整表跑一遍，逐行比对 JS 与 Python 的
预测值，差值超过 1e-6 就报错。

产出：
    web/model.json        模型参数（同时也是 node 校验的输入）
    web/index.html        自包含网页（模板 + 内联的模型和数据）
    results/app_test_cases.csv  校验用的样例，人可读
"""
from __future__ import annotations

import json
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

from common import (CATEGORICAL, NUMERIC, RES, ROOT, SEED, TARGET,
                    guard_no_leakage, load_clean, make_cv, make_preprocessor,
                    rmse)

WEB = ROOT / "web"
TEMPLATE = WEB / "template.html"
INDEX = WEB / "index.html"
MODEL_JSON = WEB / "model.json"

ALPHAS = np.logspace(-3, 3, 25)
TEST_SIZE = 0.2

# 没有真实房源时的默认房源（望京一套两居，取自数据集中位数附近），
# 让网页一打开就是「算过一次」的状态，而不是空表。
DEFAULT_INPUT = {
    "地段": "望京", "面积": 91.5, "房间数": 2, "厅数": 1,
    "建成年": 2005, "总层数": 12, "关注人数": 15,
    "装修": "精装", "形式": "板楼", "近地铁": 1, "南北通透": 1,
}


def jsonable(v):
    """NaN / NaT → None，其余原样。JSON 没有 NaN 字面量。"""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def fit_final(df: pd.DataFrame):
    """在**全部** 306 行上重新拟合最终模型。

    04/05 里的指标（CV R²、留出集 R²）描述的是泛化能力，那是在训练集上评估的。
    真正拿去用的模型没理由只用 80% 的数据，所以这里用全量重拟合。
    两者不是同一个对象，README 里说清楚了。
    """
    X = df[[c for c in df.columns if c != TARGET]]
    y = df[TARGET]
    guard_no_leakage(X, y)

    # 重新搜一次 alpha，而不是把 05 的结果硬编码进来 ——
    # 硬编码的数字早晚会和 05 跑出来的对不上。
    def pipe():
        return Pipeline([("pre", make_preprocessor()), ("model", Ridge())])

    gs = GridSearchCV(pipe(), {"model__alpha": ALPHAS}, cv=make_cv(),
                      scoring="r2", error_score="raise", n_jobs=1)
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
        f"说明 JS 侧的特征拼接顺序会错位")
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


def extract_data(df: pd.DataFrame) -> dict:
    """导出画图要用的 306 行原始数据 + 各地段统计。"""
    d = df.copy()

    # 地段按房源数从多到少排，下拉框里常用的排前面
    order = d["地段"].value_counts()
    districts = list(order.index)
    didx = {name: i for i, name in enumerate(districts)}

    decors = sorted(d["装修"].dropna().unique().tolist())
    forms = sorted(d["形式"].dropna().unique().tolist())
    didx_dec = {v: i for i, v in enumerate(decors)}
    didx_form = {v: i for i, v in enumerate(forms)}

    rows = []
    for _, r in d.iterrows():
        rows.append([
            jsonable(round(float(r["面积"]), 2)),
            jsonable(r["房间数"]), jsonable(r["厅数"]), jsonable(r["总层数"]),
            jsonable(r["建成年"]), jsonable(r["关注人数"]),
            int(r["近地铁"]), int(r["南北通透"]),
            didx_dec.get(r["装修"]), didx_form.get(r["形式"]),
            didx[r["地段"]],
            round(float(r[TARGET]), 1),
        ])

    stat = d.groupby("地段")[TARGET].agg(["median", "size"])

    # 两条趋势线在 Python 侧算好，网页只负责画 —— 这样网页上的 r 值和
    # README 里 fig02/fig05 的数字必然一致。
    def trend(xs, ys):
        m = xs.notna() & ys.notna()
        k, b = np.polyfit(xs[m], ys[m], 1)
        return {"slope": float(k), "intercept": float(b),
                "r": float(np.corrcoef(xs[m], ys[m])[0, 1])}

    area_trend = trend(d["面积"], d[TARGET])
    year_trend = trend(d["建成年"], d[TARGET])

    return {
        "fields": ["面积", "房间数", "厅数", "总层数", "建成年", "关注人数",
                   "近地铁", "南北通透", "装修", "形式", "地段", "单价"],
        "districts": districts,
        "district_median": [round(float(stat.loc[n, "median"]), 0) for n in districts],
        "district_count": [int(stat.loc[n, "size"]) for n in districts],
        "decors": decors,
        "forms": forms,
        "rows": rows,
        "area_trend": area_trend,
        "year_trend": year_trend,
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


def verify_with_node(model: dict, data: dict, pipe: Pipeline, df: pd.DataFrame) -> bool:
    """让 node 用 web/predict.js 把 306 行整表跑一遍，和 Python 逐行对答案。"""
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
    print(f"  JS 与 Python 逐行比对 {res['n']} 行：最大差值 {res['worst']:.3e} 元/㎡"
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
    data_df = extract_data(df)

    print("\n拟合最终模型（全部 306 行）：")
    pipe, alpha, cv_r2, holdout_r2, holdout_rmse = fit_final(df)

    model = extract_model(pipe, alpha, cv_r2, holdout_r2, holdout_rmse, df)
    MODEL_JSON.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n导出数据与渲染网页：")
    print(f"  {len(data_df['rows'])} 行房源、{len(data_df['districts'])} 个地段、"
          f"{sum(len(c) for c in model['categories'])} 个 one-hot 列 → "
          f"{len(model['coef'])} 个特征")
    render(model, data_df)

    print("\n校验网页预测逻辑：")
    verified = verify_with_node(model, data_df, pipe, df)

    # 留一份人可读的样例，方便手工核对
    X = df[[c for c in df.columns if c != TARGET]].head(12).copy()
    X["Python预测单价"] = np.round(pipe.predict(X), 2)
    X.to_csv(RES / "app_test_cases.csv", index=False, encoding="utf-8-sig")

    print(f"\n完成：双击 {INDEX.relative_to(ROOT)} 即可使用"
          f"{'（已通过 node 校验）' if verified else '（注意：未经 node 校验）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
