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

页面除「估价器」外还是本项目的**课堂汇报页**：数据口径、防泄露三道关、两版模型
对比、换数据 / 换模型的收益拆解、新字段一元解释力、调参为什么不涨分、已知局限。
这些数字全部从 results/ 下的产出物读进来（见 _report），**一个都不在模板里写死** ——
页面上说的和 results/ 里记的必须是同一份数。

导出前跑**三道**校验，缺任何一道都不发布：
    verify_template_refs   模板里 80+ 处数字路径与元素引用是否真的存在（不查这个，
                           路径写错就是一张空表，而下面两道都会打 ✓）
    verify_with_node       predict.js 整表 43,213 行与 Python 逐行比对（数值）
    verify_page_script     生成后的页面脚本能否编译（语法）

产出：
    web/model.json              模型参数（同时也是 node 校验的输入）
    web/index.html              自包含网页（模板 + 内联的模型、数据和实测结果）
    results/app_test_cases.csv  校验用的样例，人可读
"""
from __future__ import annotations

import base64
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

from common import (CATEGORICAL, LEAK_R2_THRESHOLD, NUMERIC, RAW_CSV, RES, ROOT,
                    SEED, TARGET, WINDOW_YEARS, guard_no_leakage, load_clean,
                    make_cv, make_preprocessor, rmse)
from common_v2 import GBDT_FAMILY

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
    CV R² +0.678 → +0.696（**1.8 个百分点**，不是 0.6 —— 这个数曾经写错，
    和页面自己注入的 poly_gain 对不上）、RMSE 13,669 → 13,306 元/㎡，
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


def _results(name: str) -> pd.DataFrame:
    """读 results/ 下的产出物。

    网页上每一个来自实验的数字都必须**从这些文件读**，不能抄进模板变成常量：
    抄的那一份不会跟着重跑走，早晚会出现「页面说的」和「results/ 里的」不一致。
    """
    path = RES / name
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path.relative_to(ROOT)}，汇报页要引用它。"
            "先按 CLAUDE.md §3.1 的产出物表把对应脚本跑完，再来导出网页。")
    return pd.read_csv(path)


def _provenance_v2() -> dict:
    """第二版建模表的规模：行数与小区数。汇报页的对照表要用。

    只读「小区」一列 —— 全表 7.5 MB，usecols 让 pandas 跳过其余列。
    """
    path = ROOT / "data" / "house_v2_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path.relative_to(ROOT)}，汇报页要引用第二版的规模。"
            "先跑 src/11_prepare_v2.py。")
    col = pd.read_csv(path, usecols=["小区"], encoding="utf-8-sig",
                      low_memory=False)["小区"]
    return {"rows": int(len(col)), "communities": int(col.nunique())}


def _report(n_rows: int, districts: list[str]) -> dict:
    """汇报页要引用的全部实测结果。

    这些数字分别由 02 / 04 / 05 / 12 / 13 / 14 算出并落盘在 results/ 下，
    本函数只做「读出来 + 摊平成页面好用的形状」，**不做任何再计算**（唯一例外
    是换数据 / 换模型那两段减法，它需要把两个不同来源的表放在同一个口径下比）。
    所以页面上的数与本项目其它地方的数必然一致。
    """
    s1 = _results("model_scores.csv")          # 04：第一版 5 个模型
    s2 = _results("model_scores_v2.csv")       # 12：第二版 9 个模型 × 2 套 CV
    folds = _results("model_folds_v2.csv")     # 12：逐折明细
    tune1 = _results("tuning_results.csv")     # 05：第一版调参
    tune2 = _results("tuning_results_v2.csv")  # 13：第二版调参
    ratio = _results("leak_ratio.csv").iloc[0]  # 02：总价派生量与目标的相关系数
    # 04：真把「总价_万」放进特征再跑一遍 Ridge 的结果。必须读这张表而不是拿
    # 「小区均价」的单特征 R² 顶替 —— 两者恰好都约等于 0.868，抄错了看不出来。
    demo = _results("leak_demo.csv").set_index("含泄露列")["5折CV_R2"]
    # 02：全量与建模窗口各自的价格水位跨度。第一节「为什么要切窗口」用它 ——
    # 这句话从前写的是「全量跨越 1.75 倍」，而 1.75 其实是窗口内的读数，全量是 4.05 倍。
    span = _results("price_span.csv").iloc[0]
    # 14：「新字段比地理维度低多少」。第六节原先只说「低一个数量级」—— 那是
    # 对比 0.0368 与 0.5068 得出的保守说法，实际是 14 倍。倍数由表算，不写进模板。
    eta2 = _results("eta2_v2.csv")
    eta2_geo = float(eta2.loc[eta2["类别"] == "地理", "eta2"].max())
    eta2_nongeo = float(eta2.loc[eta2["类别"] == "非地理", "eta2"].max())

    def leak(name: str) -> list[dict]:
        """泄露筛查表 + 三分类。

        判定列只区分「判为泄露 / 正常」两档，而本项目的重点是**三**档：
        数值判据抓住的（小区均价）、数值判据放过但语义上是泄露的（总价_万）、
        以及正常特征。第三档是这套方法学最值得讲的一点，所以在导出时就把
        类别定死，别让模板去猜。
        """
        df = _results(name)
        return [{"f": str(r["特征"]), "r2": float(r["单特征CV_R2"]),
                 "corr": float(r["相关系数"]),
                 "cls": ("caught" if r["判定"] == "泄露"
                         else "semantic" if r["特征"] == "总价_万" else "ok")}
                for _, r in df.iterrows()]

    def leak_row(name: str, feature: str) -> dict:
        """按**列名**取某一行的筛查结果。

        正文点名引用「小区均价」「总价_万」这两列，用名字取而不是按下标 ——
        表是按 R² 排序的，下标会随数据变，名字不会。取不到就直接报错：
        哪天这两列不在筛查表里了，正文说的就不是页面图上的东西了。
        """
        df = _results(name)
        hit = df.loc[df["特征"] == feature]
        if hit.empty:
            raise KeyError(f"{name} 里没有「{feature}」，汇报页的正文点名要引用它")
        r = hit.iloc[0]
        r2 = float(r["单特征CV_R2"])
        return {"r2": r2, "corr": float(r["相关系数"]),
                # 离阈值多远（可为负）。第二版那个「刚好越过」的说法要有数撑着
                "margin": r2 - LEAK_R2_THRESHOLD,
                "cls": "caught" if r["判定"] == "泄露" else "semantic"}

    def simpson(label: str) -> dict:
        """fig06 那张辛普森图里的某个口径。由 03_eda.py 落盘（图上画的和这里读的
        是同一次计算），按**口径名**取行 —— 表比 R² 表短，但同样不该按下标取。"""
        df = _results("simpson_age.csv")
        hit = df.loc[df["口径"] == label]
        if hit.empty:
            raise KeyError(f"simpson_age.csv 里没有「{label}」，汇报页正文点名要引用它")
        r = hit.iloc[0]
        return {"v": float(r["值"]), "note": str(r["说明"])}

    def curve(label: str) -> float:
        """fig09 那条验证曲线上的一个读数，由 05_tune.py 落盘。

        第七节整节的论据就是这条曲线的形状（「最大缝隙只有 0.0004」「从哪个 alpha 开始塌」），
        所以这几个数必须和画图时用的是同一次计算 —— 不能只看图凭印象写。
        """
        df = _results("val_curve_v1.csv")
        hit = df.loc[df["口径"] == label]
        if hit.empty:
            raise KeyError(f"val_curve_v1.csv 里没有「{label}」，汇报页正文点名要引用它")
        return float(hit.iloc[0]["值"])

    # 排除性验证（12_model_compare.py 的 unseen_plate()）：两套 CV 十折里，
    # 有多少样本的板块根本没在训练集出现过。第八节用它排除「外推」这个替代解释。
    unseen = _results("unseen_plate.csv")
    unseen_rows = int(unseen["没见过的行数"].sum())
    unseen_max = float(unseen["没见过的板块占比"].max())

    # 逐折明细摊平成 {模型: {口径: [5 折分数]}}，页面用它画折间分布
    per_fold: dict[str, dict[str, list[float]]] = {}
    for (m, scheme), g in folds.groupby(["模型", "方案"], sort=False):
        key = "GroupKFold" if str(scheme).startswith("Group") else "KFold"
        per_fold.setdefault(str(m), {})[key] = [float(v) for v in g["R2"]]

    # ---- 换数据 / 换模型：同一把尺子（普通 KFold）量两段收益 --------------
    # 与 14_eda_v2.py 的算法一致：第一段固定用 Ridge，第二段固定用同一份新数据，
    # 但「谁算冠军」按主口径 GroupKFold 定 —— 拿只在 KFold 上赢的堆叠来算收益，
    # 等于用偏乐观的尺子量收益，见 README §九。
    r1 = float(s1.loc[s1["模型"].str.contains(r"Ridge\("), "5折CV_R2"].iloc[0])
    r2 = float(s2.loc[s2["模型"].str.contains("岭回归 Ridge"), "KFold_CV_R2"].iloc[0])
    gi = s2["GroupKFold_CV_R2"].idxmax()
    champ, gb = str(s2.loc[gi, "模型"]), float(s2.loc[gi, "KFold_CV_R2"])
    ki = s2["KFold_CV_R2"].idxmax()

    # GBDT 家族相对线性模型的差距 —— 报「一个区间」而不是「一个数」：
    # 四个 GBDT 的差距实测并不同（调参前 XGBoost / CatBoost 只高 1.9 个点，
    # 前三名高 5.4 个点），说成「高 5~9 个点」是把区间两头的数都写错了。
    gbdt = s2.loc[s2["模型"].isin(GBDT_FAMILY)]
    assert len(gbdt) == len(GBDT_FAMILY), \
        f"GBDT_FAMILY 里有模型不在 model_scores_v2.csv 里：{set(GBDT_FAMILY) - set(gbdt['模型'])}"
    lin = float(s2.loc[s2["模型"].str.contains("岭回归"), "GroupKFold_CV_R2"].iloc[0])
    lo, hi = gbdt.loc[gbdt["GroupKFold_CV_R2"].idxmin()], gbdt.loc[gbdt["GroupKFold_CV_R2"].idxmax()]

    # 前三名之间的差距（「打平」这个说法的量化依据）。
    top3 = s2["GroupKFold_CV_R2"].nlargest(3)
    top_gap = float(top3.iloc[0] - top3.iloc[-1])

    # XGBoost 的坏折到底坏在哪 —— 逐折跟其余三个 GBDT 的同折均值比。
    # **不能写「每一折都低」**：实测折 4 它比同折均值高 0.0003，折 1 也只是差 0.0006，
    # 真正塌掉的是折 3。把「哪一折塌了、占总差距多少」算出来，比一句笼统的
    # 「每折都差」既准确又更有说服力。
    # CatBoost 的「欠拟合」两个症状。**都不能说成「全场最小」**：
    # 落差 0.0247 全场排第 4（均值基线和两个线性模型更小），只在四个 GBDT 里最小；
    # 折间标准差要指名是**普通 KFold 那一列**（0.0019，全场最低，均值基线除外），
    # GroupKFold 那一列它其实比 HistGBR / LightGBM 都大 —— 不指名就会说错。
    cat_name = "CatBoost"
    cat_row = s2.loc[s2["模型"] == cat_name].iloc[0]
    fam = s2.loc[s2["模型"].isin(GBDT_FAMILY)]
    # +1 是因为「最小」= 第 1 名，而 rank() 从 1 起算、本身已经是对的，这里只是写清楚。
    cat_gap_rank_fam = int(fam["GroupKFold_落差"].rank().loc[cat_row.name])
    cat_kf_std_rank = int(s2["KFold_CV_std"].rank().loc[cat_row.name])

    xgb_name = "XGBoost"
    peers = [n for n in GBDT_FAMILY if n != xgb_name]
    fx = np.array(per_fold[xgb_name]["GroupKFold"])
    peer_mean = np.array([per_fold[n]["GroupKFold"] for n in peers]).mean(axis=0)
    deficit = peer_mean - fx                       # 正数 = 这一折落后
    behind = deficit[deficit > 1e-3]
    worst_i = int(np.argmax(deficit))
    peer_spread = [float(np.ptp(per_fold[n]["GroupKFold"])) for n in peers]

    return {
        # 区县个数：正文里「第一版的区县只有 13 个取值」这句要用。
        # 不写成常量是因为它来自数据（`districts` 是从建模表现推出来的）。
        "n_districts": len(districts),
        # 价格水位跨度：**全量**与**建模窗口**是两个数，正文引用时不能混。
        # 曾经把窗口那个（1.75）当成全量写进正文，低估了切窗口的理由。
        "span": {"full": float(span["全量跨度倍数"]),
                 "window": float(span["窗口跨度倍数"]),
                 "lo": float(span["全量年度中位数最低"]),
                 "hi": float(span["全量年度中位数最高"]),
                 "lo_year": int(span["最低年份"]), "hi_year": int(span["最高年份"])},
        "v1": {
            "models": [{"name": str(r["模型"]), "cv": float(r["5折CV_R2"]),
                        "std": float(r["CV标准差"]), "hold": float(r["留出集_R2"]),
                        "rmse": float(r["留出集RMSE"])} for _, r in s1.iterrows()],
            "n_test": int(np.ceil(n_rows * TEST_SIZE)),
            "n_train": n_rows - int(np.ceil(n_rows * TEST_SIZE)),
            # 正文要引用的两个差值。写在这里而不是让模板做减法：模板只能从
            # report 里取，让它自己算就又出现了「页面上的数和 results/ 里的不一样」的口子。
            "poly_gain": float(s1.iloc[0]["5折CV_R2"] - s1.iloc[1]["5折CV_R2"]),
            "knn_gap": float(s1.iloc[0]["5折CV_R2"] - s1.iloc[3]["5折CV_R2"]),
        },
        "v2": {
            "models": [{"name": str(r["模型"]), "kf": float(r["KFold_CV_R2"]),
                        "kf_std": float(r["KFold_CV_std"]),
                        "gk": float(r["GroupKFold_CV_R2"]),
                        "gk_std": float(r["GroupKFold_CV_std"]),
                        "hold": float(r["留出集_R2"]), "rmse": float(r["留出集_RMSE"]),
                        "mae": float(r["留出集_MAE"]),
                        "gap": float(r["GroupKFold_落差"])} for _, r in s2.iterrows()],
            "folds": per_fold,
            "optimism": float(s2["GroupKFold_落差"].median()),
            # 排除性验证：有多少样本落在「训练集没见过的板块」上。
            # 第八节要用它说明落差不是「整片区域没见过」造成的。落盘在
            # results/unseen_plate.csv（由 12_model_compare.py 的 unseen_plate() 产生）。
            # max_pct 是**百分数**（0.0068 表示 0.0068%），不是比例 ——
            # 正文在它后面直接跟一个「%」，存成比例会让那个 % 说错话。
            "unseen_plate": {"rows": unseen_rows,
                             "max_pct": unseen_max * 100.0, "n_folds": len(unseen)},
            "provenance": _provenance_v2(),
            # 「GBDT 比线性模型高多少」：一个区间，两头的模型名都带上，
            # 因为「为什么垫底的两个只高一点」正是下一段要解释的事。
            "gbdt_span": {
                "lin": lin, "lin_name": "岭回归 Ridge",
                "lo": {"name": str(lo["模型"]), "gk": float(lo["GroupKFold_CV_R2"]),
                       "gap": float(lo["GroupKFold_CV_R2"]) - lin},
                "hi": {"name": str(hi["模型"]), "gk": float(hi["GroupKFold_CV_R2"]),
                       "gap": float(hi["GroupKFold_CV_R2"]) - lin},
                "n": len(gbdt),
                "top_gap": top_gap,
                # 「打平」到底有多平：领先第三名的差距 ÷ 冠军自己的折间标准差。
                # 直接说「远小于标准差」是空口比较，给个倍数才站得住。
                "top_gap_ratio": top_gap / float(s2["GroupKFold_CV_std"].iloc[0])},
            # XGBoost 那一节要用的：坏在哪一折、占总差距多少、极差是别人的几倍
            "xgb": {
                "name": xgb_name,
                "worst_fold": worst_i + 1,
                "worst_v": float(fx[worst_i]),
                "peer_mean": float(peer_mean[worst_i]),
                "share": float(deficit[worst_i] / behind.sum()) if behind.size else 0.0,
                "behind_n": int(behind.size), "n_folds": int(fx.size),
                "spread": float(np.ptp(fx)),
                # 倍数的两头都算出来，正文说「约 N 倍」时用的是同一个区间
                "ratio_lo": float(np.ptp(fx) / max(peer_spread)),
                "ratio_hi": float(np.ptp(fx) / min(peer_spread)),
            },
            "cat": {
                "name": cat_name,
                "gap": float(cat_row["GroupKFold_落差"]),
                "gap_rank_fam": cat_gap_rank_fam,       # 在四个 GBDT 里排第几小
                "kf_std": float(cat_row["KFold_CV_std"]),
                "kf_std_rank": cat_kf_std_rank,         # 在全部 9 个模型里排第几小
                "n_fam": len(fam), "n_all": len(s2),
                # 在主口径 GroupKFold 上，四个 GBDT 里垫底的是 XGBoost 而不是
                # CatBoost —— 两者只差 0.0003，实质并列。正文如果说「CatBoost 排最后」
                # 就和同一页 gbdt_span.lo.name 说的「XGBoost 垫底」自相矛盾。
                "gk_rank_fam": int(fam["GroupKFold_CV_R2"].rank(ascending=False).loc[cat_row.name]),
                "gk_gap_to_last": float(cat_row["GroupKFold_CV_R2"]
                                        - fam["GroupKFold_CV_R2"].min()),
                # CatBoost 真正垫底的是这两列：普通 KFold 与留出集。
                "kf_rank_fam": int(fam["KFold_CV_R2"].rank(ascending=False).loc[cat_row.name]),
                # 反面对照：XGBoost 分数一样低，但抖动完全是另一回事
                "xgb_gk_std": float(s2.loc[s2["模型"] == xgb_name, "GroupKFold_CV_std"].iloc[0]),
            },
        },
        "leak": {"v1": leak("leak_screen.csv"), "v2": leak("leak_screen_v2.csv"),
                 "threshold": LEAK_R2_THRESHOLD,
                 "ratio_corr": float(ratio["与单价相关系数"]),
                 "ratio_maxdiff": float(ratio["最大差_元每平米"]),
                 # 正文点名要引用的三处，按名字取
                 "named": {
                     "小区均价": leak_row("leak_screen.csv", "小区均价"),
                     "总价_万": leak_row("leak_screen.csv", "总价_万"),
                     "总价_万_v2": leak_row("leak_screen_v2.csv", "总价_万"),
                     # 第六节拿它当「第一版最强的新字段」的对照水位线
                     "近地铁": leak_row("leak_screen.csv", "近地铁"),
                 },
                 # 把「小区均价」当特征用的代价：Ridge 的 CV R² 会从 r1 虚高到这里。
                 # 正文那句「凭空高 N 个百分点」就是这两个数之差。
                 "fake_gain": float(leak_row("leak_screen.csv", "小区均价")["r2"] - r1),
                 # 「总价_万」版的同一件事，但这是**真跑了一遍 Ridge**（04 的
                 # leak_demo），不是拿单特征 R² 顶替。第二节正文引用的是这一组。
                 "demo": {"normal": float(demo["—"]), "leaked": float(demo["总价_万"]),
                          "gain": float(demo["总价_万"] - demo["—"])}},
        "tuning": {
            "v1": [{"item": str(r["搜索对象"]), "best": str(r["最优参数"]),
                    "before": float(r["调参前CV_R2"]), "after": float(r["调参后CV_R2"]),
                    "gain": float(r["净提升"])} for _, r in tune1.iterrows()],
            "v2": [{"name": str(r["模型"]), "before": float(r["调参前_GroupKFold"]),
                    "after": float(r["调参后_GroupKFold"]),
                    "before_kf": float(r["调参前_KFold"]),
                    "after_kf": float(r["调参后_KFold"]), "gain": float(r["净提升"]),
                    # 「搜索口径 ≠ 汇报口径」那一段要拿这两个差值对比着说。
                    # 减法放在这里做，不让模板去减 —— 否则页面上就会出现一个
                    # 只存在于 JS 里、results/ 里查不到的数。
                    "gain_kf": float(r["调参后_KFold"] - r["调参前_KFold"]),
                    # 两个口径涨幅之差 = 这一批调参里「只兑现到 KFold」的部分。
                    # 也在 Python 里算好：页面上不该出现 results/ 查不到的数。
                    "gain_diff": float(r["调参后_KFold"] - r["调参前_KFold"]
                                       - r["净提升"]),
                    "params": str(r["最优参数"]), "n": int(r["候选数"]),
                    "secs": float(r["耗时秒"])} for _, r in tune2.iterrows()],
        },
        "eta2": [{"f": str(r["字段"]), "n": int(r["取值数"]), "kind": str(r["类别"]),
                  "v": float(r["eta2"])} for _, r in eta2.iterrows()],
        # 第六节：最强的新字段 ÷ 最强的地理字段。这个比值是「低一个数量级」的精确版。
        "eta2_span": {"geo": round(eta2_geo, 4), "nongeo": round(eta2_nongeo, 4),
                      "ratio": round(eta2_geo / eta2_nongeo, 1)},
        # 辛普森悖论：全市看是正的、拆开看符号是混的，正文三段都引这里的数
        "simpson": {"all": simpson("全市合并"), "within": simpson("扣掉区县均值"),
                    "max": simpson("各区县最大"), "min": simpson("各区县最小"),
                    "pos": simpson("各区县为正的比例")},
        # fig09 验证曲线上的读数（第七节）。「最大缝隙」和「从哪开始塌」是那一节的全部论据
        "curve": {"alpha_best": curve("Ridge 最优 alpha"),
                  "alpha_lo": curve("Ridge alpha 下界"),
                  "ridge_gap": curve("Ridge 最大训练-验证缝隙"),
                  "ridge_collapse": curve("Ridge 开始塌的 alpha"),
                  "ridge_floor": curve("Ridge 最大 alpha 处验证得分"),
                  "knn_k1_train": curve("KNN k=1 训练得分"),
                  "knn_k1_valid": curve("KNN k=1 验证得分"),
                  "knn_gap": curve("KNN 最大训练-验证缝隙"),
                  "knn_best_k": curve("KNN 最优 k"),
                  "knn_best_valid": curve("KNN 最优 k 处验证得分")},
        "gain": {"ridge_v1": r1, "ridge_v2": r2, "champ": champ, "champ_kf": gb,
                 "data": r2 - r1, "model": gb - r2,
                 "kf_best": str(s2.loc[ki, "模型"]),
                 "kf_best_v": float(s2.loc[ki, "KFold_CV_R2"])},
    }


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
        # 「面积 vs 单价」与「面积 vs 总价」的对照 —— 这就是「目标为什么选单价」
        # 的全部代价。总价在 02 里已按防泄露规则剔除，这里用 单价 × 面积 还原：
        # 总价_万 不过是它除以一万，而相关系数不受常数因子影响。
        "area_total_trend": _trend(d["面积"], d[TARGET] * d["面积"]),
        "age_trend_by_district": age_trend_by,
        "cats": cards,
        "rooms": sorted(int(v) for v in d["室"].dropna().unique()),
        "halls": sorted(int(v) for v in d["厅"].dropna().unique()),
        "baths": sorted(int(v) for v in d["卫"].dropna().unique()),
        "ranges": {k: list(v) for k, v in FORM_RANGES.items()},
        "default_input": DEFAULT_INPUT,
        # 汇报页的实测结果（两版模型对比 / 泄露筛查 / 调参 / η² / 收益拆解）
        "report": _report(len(d), districts),
    }


def figures() -> dict[str, str]:
    """把 figures/ 下的实测分析图读成 data: URI，供模板内嵌。

    为什么内嵌而不是 `<img src="figures/fig01....png">`：网页的交付形态是
    **单个 HTML 文件、双击即开、断网可用**（README §八）。写成相对路径的话，
    把 index.html 单独拷到别处、或者换个目录打开，图就全裂了。
    代价是文件变大（13 张 PNG 共约 3 MB，base64 后约 4 MB），换来的是
    「一个文件就是全部」—— 课堂汇报时最不容易出事的那种形态。
    """
    fig_dir = ROOT / "figures"
    out: dict[str, str] = {}
    for png in sorted(fig_dir.glob("fig*.png")):
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        out[f"__FIG_{png.stem}__"] = f"data:image/png;base64,{b64}"
    assert out, f"{fig_dir} 下一张 fig*.png 都没有 —— 先跑 03/04/05/12/13/14 生成图"
    return out


def render(model: dict, data: dict) -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    for token, payload in [("__PREDICT_JS__", (WEB / "predict.js").read_text(encoding="utf-8")),
                           ("__MODEL_JSON__", json.dumps(model, ensure_ascii=False,
                                                         separators=(",", ":"))),
                           ("__DATA_JSON__", json.dumps(data, ensure_ascii=False,
                                                        separators=(",", ":")))]:
        assert token in html, f"模板里缺少占位符 {token}"
        html = html.replace(token, payload.replace("</script", "<\\/script"))

    figs = figures()
    # 每张图都必须真的被模板用到，且模板里不能留没被替换的 __FIG_*__ ——
    # 前者防「导出器读了图但页面没引用」的白干，后者防图裂成一行占位符字符串。
    for token, uri in figs.items():
        assert token in html, f"{token} 在模板里没有引用，这张图不会被显示"
        html = html.replace(token, uri)
    left = re.findall(r"__FIG_\w+__", html)
    assert not left, f"模板引用了不存在的图：{sorted(set(left))}"

    INDEX.write_text(html, encoding="utf-8")
    mb = INDEX.stat().st_size / 1024 / 1024
    size = f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1024:.0f} KB"
    print(f"  内嵌实测图 {len(figs)} 张（figures/fig*.png，base64 后占文件大头）")
    print(f"  已生成 {INDEX.relative_to(ROOT)}  （{size}）")


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


def verify_template_refs(data: dict, model: dict) -> None:
    """检查模板里对「数据」和「 DOM 」的每一处引用是否真的存在。

    为什么需要这第三道检查：node 那两道验的是**语法**（vm.Script 只编译不执行）
    和 **predict.js 的数值**，而页面脚本是另一份代码。下面两类错误它们一律放行，
    两类都会让页面变成一张空表：

      1. <span data-n="a.b.c"> 的路径在注入的数据里取不到值 —— fillNumbers 抛错，
         整个 IIFE 中断，标题、下拉框、数字全不填；
      2. $("某id") 在 HTML 里不存在 —— initForm / render 抛 TypeError，同上。

    这类事故实测发生过一次（模板里多一个 `}`），当时两道 node 检查都打了 ✓。
    模板里有 80 多个 data-n 和 30 多个元素引用，靠人眼睛过一遍不现实，所以固化
    在这里，每次导出都跑。格式串的合法集合是从模板里**读**出来的，不在这里
    另抄一份 —— 抄的那份会先过期。
    """
    html = TEMPLATE.read_text(encoding="utf-8")
    # data-n 只在标签里查：JS 注释里也有 data-n="a.b.c:格式" 这种示例，
    # <script> 的内容不是 DOM，剔掉再扫，否则会拿注释当引用去查数据。
    markup = re.sub(r"<script>[\s\S]*?</script>", "", html)
    # 元素引用反过来 —— $("id") 全都写在 <script> 里，所以要在**整份**模板里找，
    # 而被引用的 id 定义在标签里。两处范围不能弄反，弄反了这项检查就是在空跑。
    refs = sorted(set(re.findall(r'\$\("([^"]+)"\)', html)))

    fmts = set(re.findall(r"^\s*(\w+):\s*v\s*=>", html, re.M))
    assert fmts, "没能从模板里读到 FMT 的格式串，检查方式要跟着模板一起改"

    # 把「注入处数」回填给页面：页脚拿它说明「指标数字都是注入的」。
    # 必须放在下面那个校验循环**之前** —— 回填的两个数本身也是 data-n 引用的对象，
    # 放在后面的话这一处引用会在同一次调用里被自己判为「取不到值」。
    # 写死在模板里的话，每加一处引用就得记得改一次，那正是本页想避免的事。
    data["report"].setdefault("verify", {}).update(
        n_numbers=len(re.findall(r'data-n="[^"]+"', markup)), n_refs=len(refs))

    bad: list[str] = []
    for m in re.finditer(r'data-n="([^"]+)"', markup):
        raw = m.group(1)
        if ":" not in raw:
            bad.append(f"data-n 没写格式：{raw}")
            continue
        path, f = raw.split(":", 1)
        if f not in fmts:
            bad.append(f"未知的格式串：{raw}（已知：{sorted(fmts)}）")
            continue
        # 两个来源：model.xxx 指 model.json，其余指注入的数据
        v: object = model if path.startswith("model.") else data
        keys = (path[len("model."):] if path.startswith("model.") else path).split(".")
        try:
            for k in keys:
                v = v[int(k)] if isinstance(v, list) else v[k]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            bad.append(f"data-n 在注入的数据里取不到值：{raw}")
            continue
        if v is None or isinstance(v, (dict, list)):
            bad.append(f"data-n 指向的不是标量：{raw} → {v!r}")

    ids = set(re.findall(r'\bid="([^"]+)"', markup))
    for i in [r for r in refs if r not in ids]:
        bad.append(f'$("{i}") 在 HTML 里没有对应元素，页面脚本会抛 TypeError')

    assert not bad, "模板引用了不存在的东西，网页会是半张空表：\n  - " + "\n  - ".join(bad)
    assert refs, "一个 $(\"id\") 都没扫到，说明扫描范围又写反了"
    print(f"  模板引用 {markup.count('data-n=')} 处数字路径、"
          f"{len(refs)} 处元素引用，全部存在  ✓")


def verify_with_node(model: dict, pipe: Pipeline, df: pd.DataFrame) -> float | None:
    """让 node 用 web/predict.js 把整表跑一遍，和 Python 逐行对答案。

    返回最大差值（元/㎡）；没装 node 时返回 None。这个数会被写进页脚 ——
    「实测最大差值多少」是本项目敢说「网页显示的数就是 Python 算的数」的全部依据，
    让页面自己引述它，而不是让读者去翻终端日志。
    """
    node = shutil.which("node")
    if node is None:
        print("  [跳过] 没找到 node，无法自动比对 JS 与 Python 的预测值。")
        print("         网页逻辑未经验证 —— 装了 node 再跑一次本脚本即可。")
        return None

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
        return None

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
    return float(res["worst"])


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

    # 校验都放在渲染**之前**：页脚要写「实测最大差值多少」，这个数只能由校验本身提供。
    # 两个 node 校验读的是已落盘的 model.json / predict.js，不依赖 index.html，
    # 所以可以先跑；verify_page_script 验的是生成后的页面脚本，必须排在 render 之后。
    #
    # node 比对排在模板检查**前面**，虽然它慢：模板里有一处
    # `<span data-n="report.verify.text">` 引用的是**校验结果本身**，
    # 先查模板就会查到一个还不存在的路径（实测踩过）。顺序反了报的错是
    # 「模板引用了不存在的东西」，而真正的问题是顺序。
    print("\n校验模型与 JS 是否算同一件事：")
    worst = verify_with_node(model, pipe, df)
    # text 是一个**字符串**，永远非空 —— 页面用 <span data-n="report.verify.text"> 引用它，
    # 而 fillNumbers 取到 null 会抛错、整页数字全不填。没装 node 时给一句说明，
    # 也比给一个 null 然后让页面炸掉好。
    data["report"]["verify"] = {
        "worst": worst, "n": n,
        "text": (f"{worst:.3e} 元/㎡（{n:,} 行逐行比对）" if worst is not None
                 else "本次导出没跑 node 校验（环境里没有 node）"),
    }

    print("\n校验模板引用：")
    verify_template_refs(data, model)

    print("\n导出数据与渲染网页：")
    print(f"  散点抽样 {len(data['sample']):,} 行（{data['sample_note']}）")
    print(f"  直方图用全部 {n:,} 行，{len(data['hist']['all'])} 箱；"
          f"{len(data['districts'])} 个区县各一条")
    print(f"  {len(model['coef'])} 个特征系数")
    render(model, data)

    print("\n校验页面脚本：")
    syntax_ok = verify_page_script()
    assert syntax_ok, "页面脚本有语法错误，网页会是一张空表"

    # 留一份人可读的样例，方便手工核对
    X = df[[c for c in df.columns if c != TARGET]].head(12).copy()
    X["Python预测单价"] = np.round(pipe.predict(X), 2)
    X.to_csv(RES / "app_test_cases.csv", index=False, encoding="utf-8-sig")

    print(f"\n完成：双击 {INDEX.relative_to(ROOT)} 即可使用"
          f"{'（已通过 node 校验）' if worst is not None else '（注意：未经 node 校验）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
