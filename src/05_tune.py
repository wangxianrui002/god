# -*- coding: utf-8 -*-
"""
05 交叉验证调参 —— 教程 §2.6.5「交叉验证」与 §2.4「过拟合与欠拟合」

04 用的是各模型的默认参数。这一节问一个问题：**调参到底有没有用？**

三个网格搜索：
    Ridge 的 alpha              —— 惩罚力度
    Ridge+多项式 的 多项式次数   —— 模型复杂度
    KNN 的 n_neighbors          —— 邻居数（越大越平滑）

外加两条验证曲线（validation_curve），把「训练得分」和「验证得分」
随参数变化的走势画出来 —— 这就是教程 §2.4 讲的欠拟合/过拟合：
两条线都低 = 欠拟合，训练高验证低 = 过拟合，中间那个「验证得分最高」的点
才是该选的参数。

产出：
    results/tuning_results.csv
    figures/fig08_validation_curve.png
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from common import (BASELINE, C_BLUE, C_ORANGE, DegreeOnNumeric, FIG, INK, INK_2,
                    MUTED, RES, SEED, SURFACE, TARGET, cv_scores, guard_no_leakage,
                    load_clean, make_cv, make_preprocessor, rmse,
                    setup_chinese_font)

TEST_SIZE = 0.2
ALPHAS = np.logspace(-3, 3, 25)          # 0.001 ~ 1000，对数等分
KS = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 60]


def ridge_pipe(degree: int | None = None) -> Pipeline:
    """Ridge 管线。degree=None 表示不做多项式展开。"""
    steps = [("pre", make_preprocessor())]
    if degree is not None:
        steps.append(("poly", DegreeOnNumeric(degree)))
    steps.append(("model", Ridge()))
    return Pipeline(steps)


def knn_pipe() -> Pipeline:
    return Pipeline([("pre", make_preprocessor()),
                     ("model", KNeighborsRegressor())])


def fmt_params(params: dict) -> str:
    """把 {'model__alpha': np.float64(0.316...)} 写成 'alpha=0.316228'。

    直接 str(dict) 会把 np.float64(...) 的 repr 原样写进 CSV，很难看。
    """
    parts = []
    for k, v in params.items():
        key = k.split("__")[-1]
        parts.append(f"{key}={v:g}" if isinstance(v, (int, float, np.number)) else f"{key}={v}")
    return ", ".join(parts)


def search(est: Pipeline, grid: dict, X, y, label: str) -> dict:
    """跑一次网格搜索，返回结果摘要。

    用 error_score='raise'：GridSearchCV 默认把拟合失败的组合记成 nan 继续跑，
    那样「失败」看起来只是「得分低」，会被当成一个普通候选。
    """
    gs = GridSearchCV(est, grid, cv=make_cv(), scoring="r2",
                      error_score="raise", n_jobs=1)
    gs.fit(X, y)
    print(f"  {label:34s} 最优 {gs.best_params_}   CV R² = {gs.best_score_:+.3f}"
          f"   （{len(gs.cv_results_['params'])} 组候选）")
    return {"label": label, "best_params": gs.best_params_,
            "best_score": float(gs.best_score_), "search": gs}


def fig08_curves(alpha_tr, alpha_te, k_tr, k_te) -> None:
    """两条验证曲线：Ridge 的 alpha、KNN 的 n_neighbors。"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for ax, xs, tr, te, xlabel, note, logx in [
        (axes[0], ALPHAS, alpha_tr, alpha_te, "alpha（惩罚力度）",
         "alpha 太小 → 过拟合；太大 → 欠拟合", True),
        (axes[1], np.array(KS, dtype=float), k_tr, k_te, "n_neighbors（邻居数）",
         "邻居太少 → 每个点自成一伙；太多 → 退化成全局平均", False),
    ]:
        for scores, color, name in [(tr, C_BLUE, "训练集得分"),
                                    (te, C_ORANGE, "验证集得分")]:
            m, sd = scores.mean(axis=1), scores.std(axis=1)
            ax.plot(xs, m, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, label=name)
            # 折间标准差带：带子越宽说明模型对「碰巧分到哪一折」越敏感
            ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.16, linewidth=0)

        best_i = int(np.argmax(te.mean(axis=1)))
        ax.axvline(xs[best_i], color=BASELINE, linewidth=1.4, linestyle="--")
        # 标签贴着最优点往下放，别放到坐标轴角落再拉一条斜穿全图的长引线。
        ax.annotate(f"验证得分最高  x = {xs[best_i]:.3g}",
                    xy=(xs[best_i], te.mean(axis=1)[best_i]),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=1))
        if not logx:
            ax.set_xlim(-2, max(xs) * 1.15)   # 给右端的标签留出地方

        if logx:
            ax.set_xscale("log")
            # 坑：对数轴的默认刻度标签是 mathtext 的 $10^{-1}$，那个减号是 U+2212，
            # SimHei 里没有，rcParams['axes.unicode_minus']=False 也管不到 mathtext，
            # 于是满屏 "does not have a glyph for −"。改写成纯文本刻度即可。
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())

        ax.set_xlabel(xlabel)
        ax.set_ylabel("R^2（5 折）")
        ax.set_title(note, fontsize=11.5)
        ax.legend(loc="lower right")

    fig.suptitle("验证曲线：训练得分与验证得分随复杂度变化的走势"
                 "（阴影 = 折间标准差）", y=1.03, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_validation_curve.png")
    plt.close(fig)


def main() -> int:
    setup_chinese_font()

    df = load_clean()
    X = df[[c for c in df.columns if c != TARGET]]
    y = df[TARGET]
    guard_no_leakage(X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)

    rows = []

    # ---- 0. 调参前的参照：04 里用的默认参数 ---------------------------
    print("=" * 74)
    print("调参前（04 用的默认参数，同一套 5 折）")
    print("=" * 74)
    base_ridge = cv_scores(ridge_pipe(), X, y).mean()
    base_knn = cv_scores(knn_pipe(), X, y).mean()
    print(f"  Ridge(alpha=1)         CV R² = {base_ridge:+.3f}")
    print(f"  KNN(k=5)               CV R² = {base_knn:+.3f}\n")

    # ---- 1. Ridge 的 alpha -------------------------------------------
    print("=" * 74)
    print("网格搜索")
    print("=" * 74)
    r1 = search(ridge_pipe(), {"model__alpha": ALPHAS}, X, y, "Ridge：调 alpha")
    rows.append({"搜索对象": "Ridge.alpha", "最优参数": fmt_params(r1["best_params"]),
                 "调参前CV_R2": round(float(base_ridge), 4),
                 "调参后CV_R2": round(r1["best_score"], 4)})

    # ---- 2. Ridge + 多项式次数 ---------------------------------------
    # 管线上固定带 poly 步骤（degree=1 时 PolynomialFeatures(1) 是恒等变换），
    # 这样次数才能和 alpha 放进同一个网格一起搜。
    r2 = search(ridge_pipe(degree=2),
                {"poly__degree": [1, 2, 3], "model__alpha": np.logspace(-2, 2, 9)},
                X, y, "Ridge：调 多项式次数 + alpha")
    rows.append({"搜索对象": "Ridge.多项式次数",
                 "最优参数": fmt_params(r2["best_params"]),
                 "调参前CV_R2": round(float(base_ridge), 4),
                 "调参后CV_R2": round(r2["best_score"], 4)})

    # ---- 3. KNN 的 n_neighbors ---------------------------------------
    r3 = search(knn_pipe(), {"model__n_neighbors": KS}, X, y, "KNN：调 n_neighbors")
    rows.append({"搜索对象": "KNN.n_neighbors", "最优参数": fmt_params(r3["best_params"]),
                 "调参前CV_R2": round(float(base_knn), 4),
                 "调参后CV_R2": round(r3["best_score"], 4)})

    # ---- 4. 验证曲线 --------------------------------------------------
    print("\n计算验证曲线……")
    alpha_tr, alpha_te = validation_curve(
        ridge_pipe(), X, y, param_name="model__alpha", param_range=ALPHAS,
        cv=make_cv(), scoring="r2", error_score="raise")
    k_tr, k_te = validation_curve(
        knn_pipe(), X, y, param_name="model__n_neighbors", param_range=KS,
        cv=make_cv(), scoring="r2", error_score="raise")

    # ---- 5. 最终模型：把最优参数固定下来，在留出集上验一次 -------------
    print("\n" + "=" * 74)
    print("最终模型在留出集上的表现")
    print("=" * 74)

    finals = {
        "Ridge（alpha 已调）": ridge_pipe().set_params(**r1["best_params"]),
        "KNN（n_neighbors 已调）": knn_pipe().set_params(**r3["best_params"]),
    }
    for name, est in finals.items():
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        r2h = float(est.score(X_te, y_te))
        print(f"  {name:24s} 留出集 R² = {r2h:+.3f}   RMSE = {rmse(y_te, pred):>8,.0f} 元/㎡")

    res = pd.DataFrame(rows)
    res["净提升"] = (res["调参后CV_R2"] - res["调参前CV_R2"]).round(4)
    res.to_csv(RES / "tuning_results.csv", index=False, encoding="utf-8-sig")

    best_alpha = r1["best_params"]["model__alpha"]
    best_k = r3["best_params"]["model__n_neighbors"]
    print(f"""
怎么读这些结果：
  · Ridge 的最优 alpha = {best_alpha:.3g}，CV R² 从 {base_ridge:+.3f} 提到 {r1['best_score']:+.3f}，
    提升 {(r1['best_score'] - base_ridge) * 100:.1f} 个百分点。调参确实有用 —— 但幅度有限，
    因为天花板是数据本身决定的，不是参数。
  · 多项式次数那一栏，最优是 {r2['best_params'].get('poly__degree')} 次 —— 也就是「不加多项式」。
    把 alpha 和次数放在同一个网格里一起搜，网格自己选了最简单的那一档：
    加了平方项之后维度上万、样本只有 306 个，多出来的维度没带来信息，只带来了噪声。
  · 看左图的验证曲线：alpha 从 0.001 到 0.316，训练得分一直趴在 0.9 以上，
    验证得分只有 0.36~0.43 —— 两条线之间的巨大缝隙就是过拟合。alpha 调大后
    训练得分往下掉、验证得分先升后降，在 0.316 处取到最高，再往右两条线一起塌下去，
    那就是欠拟合。教程 §2.4 讲的「欠拟合-恰好-过拟合」三段，在这张图上一眼可见。
  · KNN 最优 k = {best_k}。看验证曲线：k=1 时训练得分是满分 1.0 —— 每个点最近的
    邻居就是它自己，纯背答案。k 增大后训练得分下降、验证得分上升，两条线靠拢，
    这才是学到了规律。这就是教程 §2.4 说的过拟合。
  · 但 KNN 调到头也只有 {r3['best_score']:+.3f}，远不如线性的 {r1['best_score']:+.3f}。
    原因是 143 个地段 one-hot 之后空间又高维又稀疏，「距离」在这种空间里没有意义 ——
    这是维度灾难，调 k 治不好。""")

    fig08_curves(alpha_tr, alpha_te, k_tr, k_te)
    print(f"\n已写出：{RES / 'tuning_results.csv'}")
    print("已生成：fig08_validation_curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
