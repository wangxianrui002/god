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
    figures/fig09_validation_curve.png
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
                    RES, SEED, SURFACE, TARGET, cv_scores, guard_no_leakage,
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


def knn_distances(n_candidates: int, n_rows: int, n_splits: int = 5) -> float:
    """整个 KNN 网格搜索要算多少个「点对距离」。

    一次 ``predict`` = 测试行数 × 训练行数 个距离；网格搜索总共要做
    「候选数 × 折数」次 predict。

    为什么要算而不是写死：第一版 README 和源码注释里同时存在「十几亿」和
    「八千万」两个说法，相差三个数量级，至少有一个是早期小数据集时代的残留。
    数字写死的注释早晚会过期，算出来的不会。README 里引用的就是这个函数的结果。
    """
    per_fold = n_rows / n_splits
    return n_candidates * n_splits * per_fold * (n_rows - per_fold)


def search(est: Pipeline, grid: dict, X, y, label: str) -> dict:
    """跑一次网格搜索，返回结果摘要。

    用 error_score='raise'：GridSearchCV 默认把拟合失败的组合记成 nan 继续跑，
    那样「失败」看起来只是「得分低」，会被当成一个普通候选。
    """
    # n_jobs=-1：KNN 的一次 predict 要算「测试行数 × 训练行数」个距离，
    # 整个网格搜索是「候选数 × 折数」次 predict，量级在 10^10。串行跑要几分钟。
    # 并行只影响速度，不影响结果 —— 每一折的划分由 make_cv() 的随机种子固定。
    gs = GridSearchCV(est, grid, cv=make_cv(), scoring="r2",
                      error_score="raise", n_jobs=-1)
    gs.fit(X, y)
    print(f"  {label:34s} 最优 {gs.best_params_}   CV R² = {gs.best_score_:+.3f}"
          f"   （{len(gs.cv_results_['params'])} 组候选）")
    return {"label": label, "best_params": gs.best_params_,
            "best_score": float(gs.best_score_), "search": gs}


def overfit_caption(tr_mean: np.ndarray, te_mean: np.ndarray, low_label: str) -> str:
    """按实测的训练-验证缝隙，给这张子图选一句**说得通**的副标题。

    为什么不能写死：教科书式的「左边过拟合、右边欠拟合」在 Ridge 这张图上
    不成立 —— 43,213 行喂 11 个数值特征，训练得分和验证得分整条曲线几乎重合，
    最小的那个 alpha（惩罚几乎为零）也看不到过拟合。照抄教科书会让图在骗人：
    读者会去找那道「缝」，而图上根本没有。所以先量缝隙，再决定怎么说。
    """
    gap = float(np.max(tr_mean - te_mean))
    if gap < 0.02:
        return f"{low_label}再小也不见过拟合 —— 两条线整条重合；惩罚太大才开始欠拟合"
    return f"{low_label}太小 → 过拟合；太大 → 欠拟合"


def fig09_curves(alpha_tr, alpha_te, k_tr, k_te) -> None:
    """两条验证曲线：Ridge 的 alpha、KNN 的 n_neighbors。"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for ax, xs, tr, te, xlabel, note, logx in [
        (axes[0], ALPHAS, alpha_tr, alpha_te, "alpha（惩罚力度）",
         overfit_caption(alpha_tr.mean(axis=1), alpha_te.mean(axis=1), "alpha"), True),
        (axes[1], np.array(KS, dtype=float), k_tr, k_te, "n_neighbors（邻居数）",
         overfit_caption(k_tr.mean(axis=1), k_te.mean(axis=1), "邻居数"), False),
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
    fig.savefig(FIG / "fig09_validation_curve.png")
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
    n_dist = knn_distances(len(KS), len(X))
    print(f"  KNN 网格搜索：{len(KS)} 个候选 × 5 折，共约 {n_dist:.2e} "
          f"（{n_dist / 1e8:.0f} 亿）次点对距离计算")
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

    # 验证曲线上的几个关键读数，后面直接引用，不写死数字
    a_tr, a_te = alpha_tr.mean(axis=1), alpha_te.mean(axis=1)
    k_trm, k_te_m = k_tr.mean(axis=1), k_te.mean(axis=1)
    a_best_i, k_best_i = int(np.argmax(a_te)), int(np.argmax(k_te_m))

    # Ridge 这条曲线到底有没有过拟合，用实测缝隙回答，不靠教科书默认印象。
    # a_gap_i 是缝隙最大那一处的下标：下面打印时**必须**报这一处的训练/验证得分。
    # 曾经报的是 index 0（alpha 最小那一头）的得分，而 a_gap 取的是全曲线最大值 ——
    # 两个数不是同一次读数，读者拿它们对不上。
    a_gap_i = int(np.argmax(a_tr - a_te))
    a_gap = float(np.max(a_tr - a_te))
    # 欠拟合从哪开始：验证得分跌破峰值 2 个百分点的第一个 alpha
    under = np.where(a_te < a_te.max() - 0.02)[0]
    a_under = ALPHAS[under[0]] if len(under) else None
    k_gap = float(np.max(k_trm - k_te_m))

    # 落盘而不是只打印：展示网页（06_export.py）第七节要引用「最大缝隙」「从哪个 alpha
    # 开始塌」这几个读数，让它读文件而不是在模板里抄一份 —— 抄一份早晚会和这里跑出来的
    # 对不上，而这条曲线的形状正是那一节的全部论据。
    pd.DataFrame([
        {"口径": "Ridge 最优 alpha", "值": float(best_alpha), "说明": "网格搜索选出的惩罚力度"},
        {"口径": "Ridge 最大训练-验证缝隙", "值": round(a_gap, 4),
         "说明": f"出现在 alpha={ALPHAS[int(np.argmax(a_tr - a_te))]:g}"},
        {"口径": "Ridge 开始塌的 alpha", "值": float(a_under) if a_under else float("nan"),
         "说明": "验证得分跌破峰值 2 个百分点的第一个 alpha"},
        {"口径": "Ridge 最大 alpha 处验证得分", "值": round(float(a_te[-1]), 4),
         "说明": f"alpha={ALPHAS[-1]:g}，惩罚压掉真实信号"},
        {"口径": "Ridge alpha 下界", "值": float(ALPHAS[0]), "说明": "惩罚几乎为零的一头"},
        {"口径": "KNN k=1 训练得分", "值": round(float(k_trm[0]), 4), "说明": "每个点的最近邻居是它自己"},
        {"口径": "KNN k=1 验证得分", "值": round(float(k_te_m[0]), 4), "说明": "同一处的验证得分"},
        {"口径": "KNN 最大训练-验证缝隙", "值": round(k_gap, 4), "说明": "教科书式的过拟合"},
        {"口径": "KNN 最优 k", "值": int(best_k), "说明": "验证得分最高的邻居数"},
        {"口径": "KNN 最优 k 处验证得分", "值": round(float(k_te_m[k_best_i]), 4), "说明": ""},
    ]).to_csv(RES / "val_curve_v1.csv", index=False, encoding="utf-8-sig")

    print(f"""
怎么读这些结果：
  · Ridge 的最优 alpha = {best_alpha:.3g}，CV R² 从 {base_ridge:+.3f} 到 {r1['best_score']:+.3f}，
    提升 {(r1['best_score'] - base_ridge) * 100:.1f} 个百分点 —— 也就是说，**调了等于没调**。
    这不是说教程 §2.6.5 讲的交叉验证调参没意义，而是这份数据的瓶颈不在超参数上：
    网格 25 组候选走完，最高分和最低分之间几乎没有区别。
  · 左图是这次最值得看的一张。教程 §2.4 的教科书图景是「alpha 小 → 过拟合，
    大 → 欠拟合」，但在这份数据上**只有右半边成立**：
      - alpha 从 {ALPHAS[0]:g} 到约 {ALPHAS[a_best_i]:g}，训练得分和验证得分整条几乎重合，
        整条曲线 25 个点的缝隙全在 {np.min(a_tr - a_te):.4f}~{a_gap:.4f} 之间，
        最大的一处出现在 alpha={ALPHAS[a_gap_i]:g}（训练 {a_tr[a_gap_i]:.3f} vs 验证 {a_te[a_gap_i]:.3f}）
        —— 在最右端那一头，不在左边。
        惩罚几乎为零时**也不见过拟合**，因为 43,213 行喂 11 个数值特征，
        模型复杂度远小于样本量能支撑的规模。
      - 唯一的边界在右边：alpha 超过 {a_under:g} 附近才开始塌，到 {ALPHAS[-1]:g} 掉到 {a_te[-1]:.3f}，
        那才是欠拟合 —— 惩罚大到把真实信号也压掉了。
    所以「过拟合」不是线性模型的固有病，是「模型复杂度 ÷ 样本量」不够小时的病。
    这和 04 里 Ridge+Poly2 的结论翻转是同一件事的两面（见 README §七）。
  · 多项式次数那一栏，网格选的是 {r2['best_params'].get('poly__degree')} 次 + alpha={r2['best_params'].get('model__alpha'):g}，
    CV R² {r2['best_score']:+.4f}，比不加多项式高 {(r2['best_score'] - base_ridge) * 100:+.1f} 个百分点。
    注意它比 04 里固定 alpha=1 的 Ridge+Poly2（+0.696）还高一点 ——
    次数和 alpha 一起搜，网格自己找到了更好的组合。306 行时它选的是 degree=1。
  · KNN 最优 k = {best_k}，CV R² {r3['best_score']:+.3f}，比 k=5 的 {base_knn:+.3f} 高
    {(r3['best_score'] - base_knn) * 100:+.1f} 个百分点 —— 调参在 KNN 上是**真的有用**。
    看右图：k=1 时训练得分是 {k_trm[0]:.3f}（满分）—— 每个点最近的邻居就是它自己，
    纯背答案；验证得分只有 {k_te_m[0]:.3f}，缝隙 {k_gap:.3f}。这才是教程 §2.4 说的过拟合，
    而且比 04 里用的 k=5 还严重。k 增大后训练得分下降、验证得分上升，两条线靠拢，
    在 k={best_k} 处验证得分最高 {k_te_m[k_best_i]:.3f}。
  · 但 KNN 调到头也只有 {r3['best_score']:+.3f}，仍比线性的 {r1['best_score']:+.3f} 低
    {(r1['best_score'] - r3['best_score']) * 100:.1f} 个百分点。原因是类别列 one-hot 之后
    空间维度高、样本稀，「距离」在这种空间里没有意义 —— 这是维度灾难，调 k 治不好。""")

    fig09_curves(alpha_tr, alpha_te, k_tr, k_te)
    print(f"\n已写出：{RES / 'tuning_results.csv'}")
    print("已生成：fig09_validation_curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
