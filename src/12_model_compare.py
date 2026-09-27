# -*- coding: utf-8 -*-
"""
12 模型对比（第二版）—— 8 个模型 × 2 套交叉验证方案

这是本次实验的核心产出。同时回答三个问题：

  1. **换模型值不值**：第一版的 4 个教程模型 vs GBDT 家族 vs 堆叠集成，同一份数据比。
  2. **换数据值不值**：同一个 Ridge，跑在第一版（成交价 4.3 万行）和第二版
     （挂牌价 7.4 万行，多了朝向/楼层/小区/环线）上，差多少。
     两个收益必须分开算，否则「涨了多少」说不清是谁的功劳。
  3. **分数有多乐观**：同一套 KFold 与 GroupKFold（按小区分组）各跑一遍。
     同一个小区可能有多套房，普通 KFold 会把它们分到训练/验证两侧，
     等于让模型见过相似样本。第一版没有小区标识列只能算了；
     第二版有 community，所以这一条现在能**量化**成具体数字。

产出：
    results/model_scores_v2.csv     主表（每个模型一行）
    results/model_folds_v2.csv      逐折明细（画误差棒用）
    figures/fig10_model_compare_v2.png
    figures/fig11_data_vs_model.png
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from common import (BASELINE, C_AQUA, C_BLUE, C_CRITICAL, C_ORANGE, FIG, INK,
                    INK_2, MUTED, RES, SEED, SURFACE, TARGET, guard_no_leakage,
                    rmse, setup_chinese_font)
from common_v2 import (GROUP_COL, cv_scores, group_cv_scores, load_clean, load_xy,
                       make_models, make_stack, shuffle_once)

TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# 待比较的模型
# --------------------------------------------------------------------------
def make_contestants() -> dict:
    """8 个单模型 + 1 个堆叠集成。顺序 = 报告里表格的顺序。"""
    models = make_models()

    # 堆叠的基学习器：三个第三方 GBDT 库 + sklearn 自带的 HistGBR。
    # 不放线性模型和 KNN —— 它们的 CV R² 低 8~15 个点，加进去只会稀释。
    base = {k: v for k, v in models.items()
            if k in ("LightGBM", "XGBoost", "CatBoost", "直方图梯度提升 HistGBR")}
    models["堆叠 GBDT×4 → Ridge"] = make_stack(base)
    return models


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def evaluate(name: str, model, X, y, groups, tr, te) -> tuple[dict, list[dict]]:
    """一个模型跑三套评估：留出集、KFold、GroupKFold。"""
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]

    t0 = time.time()
    model.fit(Xtr, ytr)
    fit_s = time.time() - t0
    pred = model.predict(Xte)

    fold_k = cv_scores(model, X, y)
    fold_g = group_cv_scores(model, X, y, groups)

    row = {
        "模型": name,
        "留出集_R2": float(model.score(Xte, yte)),
        "留出集_RMSE": rmse(yte, pred),
        "留出集_MAE": float(mean_absolute_error(yte, pred)),
        "KFold_CV_R2": float(fold_k.mean()),
        "KFold_CV_std": float(fold_k.std()),
        "GroupKFold_CV_R2": float(fold_g.mean()),
        "GroupKFold_CV_std": float(fold_g.std()),
        "拟合秒": round(fit_s, 1),
    }
    # 乐观偏差：普通 KFold 比按小区分组高多少。正数 = KFold 偏乐观。
    row["GroupKFold_落差"] = row["KFold_CV_R2"] - row["GroupKFold_CV_R2"]

    folds = [{"模型": name, "方案": "KFold 5 折", "折": i + 1, "R2": float(v)}
             for i, v in enumerate(fold_k)]
    folds += [{"模型": name, "方案": "GroupKFold 5 折（按小区）", "折": i + 1, "R2": float(v)}
              for i, v in enumerate(fold_g)]
    return row, folds


# --------------------------------------------------------------------------
# 出图
# --------------------------------------------------------------------------
def fig10(res: pd.DataFrame) -> None:
    """两套 CV 方案并排 —— 这张图的重点是两根柱子的**差**，不是长度。"""
    import matplotlib.pyplot as plt

    d = res.sort_values("GroupKFold_CV_R2").reset_index(drop=True)
    ypos = np.arange(len(d))
    h = 0.36

    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    b1 = ax.barh(ypos + h / 2 + 0.015, d["KFold_CV_R2"], height=h,
                 xerr=d["KFold_CV_std"], color=C_BLUE, capsize=2.5,
                 label="普通 KFold 5 折（同小区可能被切到两边）")
    b2 = ax.barh(ypos - h / 2 - 0.015, d["GroupKFold_CV_R2"], height=h,
                 xerr=d["GroupKFold_CV_std"], color=C_ORANGE, capsize=2.5,
                 label="GroupKFold 5 折（整个小区只进一侧）")

    # 数值标签要让开误差棒 —— 固定偏移会被长误差棒盖住（XGBoost 的折间标准差
    # 是别的模型的 7 倍，第一批图里它的数字就被自己的误差棒吃掉了）。
    for bars, errs in ((b1, d["KFold_CV_std"]), (b2, d["GroupKFold_CV_std"])):
        for bar, e in zip(bars, errs):
            w = bar.get_width()
            ax.text(w + e + 0.014, bar.get_y() + bar.get_height() / 2, f"{w:+.3f}",
                    va="center", ha="left", fontsize=9, color=INK_2)

    # 每一行的落差用一条灰色连线标出来 —— 读者一眼看到的是「差多少」，
    # 这正是本次补强要回答的问题。
    for i, r in d.iterrows():
        ax.plot([r["GroupKFold_CV_R2"], r["KFold_CV_R2"]], [i, i],
                color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)), zorder=0)
        if r["GroupKFold_落差"] > 0.001:
            ax.text(r["KFold_CV_R2"] + 0.055, i + 0.30,
                    f"落差 {r['GroupKFold_落差']:.4f}", fontsize=8, color=C_CRITICAL)

    ax.axvline(0, color=BASELINE, linewidth=1.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["模型"], color=INK_2)
    ax.set_xlim(0, float(d["KFold_CV_R2"].max()) + 0.16)
    ax.set_xlabel("R^2（5 折交叉验证均值，误差棒 = 折间标准差）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    # SimHei 没有上标 ²，图里一律写 R^2
    # SimHei 里既没有上标 ²，也没有真正的减号 U+2212（会变成空白方块），
    # 所以图上一律写 R^2、用 ASCII 的 "-"。这个坑在 04_regression.py 里已经踩过一次。
    ax.set_title("第二版数据：9 个模型的 R^2，以及普通 KFold 的乐观偏差\n"
                 "落差 = 普通 KFold - GroupKFold，中位数 "
                 f"{d['GroupKFold_落差'].median():.4f}；"
                 "虚线连接同一模型的两套分数",
                 loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_model_compare_v2.png")
    plt.close(fig)


def fig11(v1_ridge: float, v2_ridge: float, v2_best: float,
          v1_best: float, v1_best_name: str, best_name: str) -> None:
    """把「换数据」和「换模型」两个收益拆开 —— 一根柱子，逐段加高。

    为什么值得单独画一张：第一版报告的结论是「缺的是数据不是算法」。
    换数据确实涨了，但涨了多少、剩下的余量在不在模型侧，
    只有把两段分开量才说得清。
    """
    import matplotlib.pyplot as plt

    steps = [
        ("第一版的 Ridge\n（成交价 4.3 万行）", v1_ridge, MUTED),
        (f"＋换数据 {v2_ridge - v1_ridge:+.4f}\n（挂牌价 7.4 万行，\n多出朝向/楼层/小区/环线）",
         v2_ridge - v1_ridge, C_AQUA),
        (f"＋换模型 {v2_best - v2_ridge:+.4f}\n（Ridge → {best_name}）",
         v2_best - v2_ridge, C_BLUE),
    ]

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    bottom = 0.0
    for label, val, color in steps:
        ax.bar([0], [val], bottom=bottom, width=0.46, color=color, label=label,
               edgecolor=SURFACE, linewidth=0.8)
        # 数值直接标在柱子右侧，不靠颜色分辨长短
        ax.text(0.26, bottom + val / 2, f"{val:+.4f}", va="center", ha="left",
                fontsize=12, color=INK)
        bottom += val
        # 每一段的累积高度标在左边，读者能顺着台阶读下来
        ax.text(-0.26, bottom, f"{bottom:+.4f}", va="center", ha="right",
                fontsize=9.5, color=INK_2)

    # 第一版的分数用一条横贯的参考线标出来，作为「起点」
    ax.axhline(v1_ridge, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=0)
    ax.text(0.62, v1_ridge, f"第一版起点 {v1_ridge:+.4f}\n（{v1_best_name}）",
            va="center", ha="left", fontsize=9, color=MUTED)

    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_ylabel("5 折交叉验证 R^2")
    ax.set_xlim(-0.85, 1.55)
    ax.set_ylim(0, bottom * 1.12)
    ax.axhline(0, color=BASELINE, linewidth=1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("「换数据」和「换模型」各值多少\n"
                 f"换数据 {v2_ridge - v1_ridge:+.4f}，换模型 {v2_best - v2_ridge:+.4f} —— "
                 "两段都要，不是二选一",
                 loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_data_vs_model.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def main(figs_only: bool = False) -> int:
    pd.set_option("display.width", 200)
    print("=" * 78)
    print("12 模型对比（第二版：挂牌数据 + GBDT）")
    print("=" * 78)

    # 堆叠集成要跑 10 分钟（内层 5 折 × 4 个基模型 × 外层 5 折），
    # 而调图只是改坐标轴。所以留一个只重画图的入口，
    # 从已落盘的结果表读数据：`uv run python src/12_model_compare.py --figs-only`
    if figs_only:
        res = pd.read_csv(RES / "model_scores_v2.csv")
        # 下面报 MAE 占比要用到目标中位数，所以即使不建模也要读一次建模表。
        # 只读一列，很便宜。
        y = load_clean()[TARGET]
        print("--figs-only：跳过建模，直接读 results/model_scores_v2.csv 重画图")
    else:
        X, y, groups = shuffle_once(*load_xy())
        print(f"样本 {len(X):,} 行 × {X.shape[1]} 特征；"
              f"{groups.nunique():,} 个「{GROUP_COL}」，"
              f"平均 {len(X) / groups.nunique():.1f} 套/小区")

        # 建模前必须过闸：X 严格来自 feature_columns()，这里再验一遍。
        # 注意传进去的是**真实特征集**（不含只用于演示的「总价_万」）。
        guard_no_leakage(X, y)
        print(f"防泄露校验通过（{len(X.columns)} 个特征：列名黑名单 + 单特征 R^2 筛查）")

        tr, te = train_test_split(np.arange(len(X)), test_size=TEST_SIZE,
                                  random_state=SEED)
        rows, folds = [], []
        for name, model in make_contestants().items():
            t0 = time.time()
            row, f = evaluate(name, model, X, y, groups, tr, te)
            rows.append(row)
            folds.extend(f)
            print(f"  {name:26s} KFold {row['KFold_CV_R2']:+.4f}  "
                  f"GroupKFold {row['GroupKFold_CV_R2']:+.4f}  "
                  f"落差 {row['GroupKFold_落差']:+.4f}  ({time.time() - t0:.0f}s)")

        res = pd.DataFrame(rows).sort_values("GroupKFold_CV_R2", ascending=False)
        res.to_csv(RES / "model_scores_v2.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(folds).to_csv(RES / "model_folds_v2.csv", index=False,
                                   encoding="utf-8-sig")

    # ---- 主表 ---------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("主表（按 GroupKFold 排序）")
    print(f"{'=' * 78}")
    show = res[["模型", "KFold_CV_R2", "GroupKFold_CV_R2", "GroupKFold_落差",
                "留出集_R2", "留出集_RMSE", "留出集_MAE", "拟合秒"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    best = res.iloc[0]
    lin = res[res["模型"].str.contains("线性回归")].iloc[0]
    print(f"\n  冠军：{best['模型']}  GroupKFold R^2 = {best['GroupKFold_CV_R2']:+.4f}"
          f"（KFold {best['KFold_CV_R2']:+.4f}）")
    print(f"  留出集 RMSE {best['留出集_RMSE']:,.0f} 元/㎡，"
          f"MAE {best['留出集_MAE']:,.0f} 元/㎡"
          f"（目标中位数 {y.median():,.0f}，MAE 占 {best['留出集_MAE'] / y.median():.1%}）")

    # ---- 乐观偏差 -----------------------------------------------------
    print(f"\n{'=' * 78}")
    print("乐观偏差：普通 KFold 比 GroupKFold 高多少")
    print(f"{'=' * 78}")
    gap = res["GroupKFold_落差"]
    worst = res.loc[gap.idxmax()]
    print(f"  中位数 {gap.median():+.4f}   最大 {gap.max():+.4f}（{worst['模型']}）"
          f"   最小 {gap.min():+.4f}")
    print("\n  含义：同一小区的房子高度相似，普通 KFold 把它们切到训练/验证两侧，"
          "\n  模型等于「见过邻居家的房子」。按小区整组切分后，R^2 平均掉 "
          f"{gap.median():.4f}，"
          "\n  这才是对「预测一个**没见过的小区**的房子」的诚实估计。")

    # ---- 逐折明细 -----------------------------------------------------
    # README §13.3 引用了「XGBoost 每一折都低、且折间极差是别人 3 倍」这个结论。
    # 那个结论是从 results/model_folds_v2.csv 里手算的 —— 手算的东西没法复现，
    # 所以把它做成脚本输出，报告里的每个数字都能从这一次运行里读到。
    if not figs_only:
        print(f"\n{'=' * 78}")
        print("逐折明细：GroupKFold 各折 R^2（谁在抖，谁在稳）")
        print(f"{'=' * 78}")
        f = pd.DataFrame(folds)
        piv = (f[f["方案"].str.startswith("GroupKFold")]
               .pivot_table(index="模型", columns="折", values="R2")
               .reindex(res["模型"]))
        head = "  " + f"{'模型':24s}" + "".join(f"{'折' + str(i):>9}" for i in piv.columns)
        print(head + f"{'极差':>9}")
        for name, r in piv.iterrows():
            cells = "".join(f"{v:>9.4f}" for v in r)
            print(f"  {name:24s}{cells}{r.max() - r.min():>9.4f}")

        # 折间极差 = 稳定性。极差最大的那个模型，分数最不可信。
        rng = (piv.max(axis=1) - piv.min(axis=1)).sort_values(ascending=False)
        worst_fold = rng.index[0]
        others = rng.drop(worst_fold)
        print(f"\n  「{worst_fold}」的折间极差 {rng.iloc[0]:.4f}，"
              f"是其余 {len(others)} 个模型的 {rng.iloc[0] / others.median():.1f} 倍"
              f"（中位 {others.median():.4f}）。")
        print("  同一个模型换一折就抖这么多，说明它在这份数据上**不稳定** ——"
              "\n  它的均值不能和其它模型直接比名次。")

    # ---- 换数据 vs 换模型 ---------------------------------------------
    print(f"\n{'=' * 78}")
    print("两个收益分开算：「换数据」与「换模型」")
    print(f"{'=' * 78}")
    v1 = RES / "model_scores.csv"
    if v1.exists():
        s1 = pd.read_csv(v1)
        r1 = s1[s1["模型"].str.contains("Ridge\\(alpha=1\\)")].iloc[0]
        v1_ridge = float(r1["5折CV_R2"])
        v1_best = float(s1["5折CV_R2"].max())
        v1_best_name = s1.loc[s1["5折CV_R2"].idxmax(), "模型"]
        gain_data = lin["KFold_CV_R2"] - v1_ridge
        gain_model = best["KFold_CV_R2"] - lin["KFold_CV_R2"]
        print("  同一个 Ridge（模型不变，只换数据）")
        print(f"    第一版 KFold R^2  {v1_ridge:+.4f}")
        print(f"    第二版 KFold R^2  {lin['KFold_CV_R2']:+.4f}")
        print(f"    ── 换数据净收益    {gain_data:+.4f}")
        print("\n  同一份第二版数据（数据不变，只换模型）")
        print(f"    线性回归          {lin['KFold_CV_R2']:+.4f}")
        print(f"    {best['模型']:14s}  {best['KFold_CV_R2']:+.4f}")
        print(f"    ── 换模型净收益    {gain_model:+.4f}")
        print(f"\n  两者合计 {gain_data + gain_model:+.4f}："
              f"第一版最好的 {v1_best_name}（{v1_best:+.4f}）→ "
              f"{best['模型']}（{best['KFold_CV_R2']:+.4f}）")
        print(f"\n  换个角度看：换模型（{gain_model:+.4f}）"
              f"{'大于' if gain_model > gain_data else '小于'}"
              f"换数据（{gain_data:+.4f}）。第一版 README 的结论是"
              "「缺的是数据不是算法」——")
        print("  换数据确实有效，但**模型侧的余量同样可观**："
              "线性模型吃不下的交互项，")
        print("  梯度提升树吃得下。两件事都要做，不是二选一。")
        have_v1 = True
    else:
        print("  未找到 results/model_scores.csv（第一版结果），跳过对照。")
        have_v1 = False
        v1_ridge = v1_best = float("nan")
        v1_best_name = ""

    # ---- 出图 ---------------------------------------------------------
    setup_chinese_font()
    fig10(res)
    print("\n  图 → figures/fig10_model_compare_v2.png")
    if have_v1:
        fig11(v1_ridge, float(lin["KFold_CV_R2"]), float(best["KFold_CV_R2"]),
              v1_best, v1_best_name, best["模型"])
        print("  图 → figures/fig11_data_vs_model.png")

    print(f"\n{'=' * 78}")
    if figs_only:
        print("仅重画图，结果表未改动")
    else:
        print(f"已写入：results/model_scores_v2.csv（{len(res)} 行）")
        print("        results/model_folds_v2.csv（逐折明细）")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="模型对比（第二版）")
    ap.add_argument("--figs-only", action="store_true",
                    help="跳过建模，从 results/model_scores_v2.csv 重画图")
    sys.exit(main(ap.parse_args().figs_only))
