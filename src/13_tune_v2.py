# -*- coding: utf-8 -*-
"""
13 调参（第二版）—— 在 GroupKFold 上搜索，就在 GroupKFold 上报告

与第一版 `05_tune.py` 的两点不同：

1. **搜索目标和报告口径统一**。第一版在普通 KFold 上搜、也在普通 KFold 上报。
   第二版既然已经确认 GroupKFold 才是诚实的口径，那就直接用它当搜索目标 ——
   拿一个偏乐观的指标去选超参数，再拿另一个指标去汇报，选出来的参数未必是
   后者意义下的最优，而且读者会以为「调参提升」和「主表」是同一把尺子量出来的。

2. **只调负担得起的模型**。CatBoost 在这份数据上单次拟合约 20 秒（墙钟，随负载浮动）
   （它要对「板块」的 270 个取值算有序目标统计），一个几十组候选的网格
   要跑几十分钟。所以它的网格**没有跑**，这一点在报告里明写出来 ——
   宁可不调，也不假装调过。

产出：
    results/tuning_results_v2.csv
    figures/fig12_tuning_v2.png
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from common import (C_AQUA, C_BLUE, C_CRITICAL, FIG, INK,
                    INK_2, MUTED, RES, SEED, SURFACE, setup_chinese_font)
from common_v2 import (cv_scores, group_cv_scores, load_xy, make_group_cv,
                       make_models, shuffle_once)

# 每个模型一组搜索空间。取值范围刻意放宽，让数据自己选 ——
# 第一版的经验是「网格自己会选」（degree 从 1 变成 3），别替它预设答案。
SPACES = {
    "LightGBM": {
        "model__n_estimators": [200, 400, 700, 1000],
        "model__learning_rate": [0.02, 0.03, 0.06, 0.1],
        "model__num_leaves": [15, 31, 63, 127],
        "model__min_child_samples": [5, 20, 50, 100],
    },
    "直方图梯度提升 HistGBR": {
        "model__max_iter": [200, 400, 700, 1000],
        "model__learning_rate": [0.02, 0.03, 0.06, 0.1],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [5, 20, 50, 100],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}
N_ITER = 24          # 每组搜索抽多少组候选；4^4=256 全跑没必要


def tune(name: str, model, X, y, groups) -> dict:
    """在 GroupKFold 上随机搜索，返回调参前后的分数。"""
    base_g = group_cv_scores(model, X, y, groups).mean()
    base_k = cv_scores(model, X, y).mean()

    t0 = time.time()
    search = RandomizedSearchCV(
        model, SPACES[name], n_iter=N_ITER, cv=make_group_cv(),
        scoring="r2", random_state=SEED, n_jobs=-1,
        error_score="raise", refit=True)
    search.fit(X, y, groups=groups)
    took = time.time() - t0

    best = search.best_estimator_
    # refit=True 之后 best_estimator_ 已在全量数据上拟合过，但 cross_val_score
    # 会重新 clone 并在每折重训，所以这里拿到的仍是干净的交叉验证分数。
    tuned_g = group_cv_scores(best, X, y, groups).mean()
    tuned_k = cv_scores(best, X, y).mean()

    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return {"模型": name, "调参前_GroupKFold": base_g, "调参后_GroupKFold": tuned_g,
            "调参前_KFold": base_k, "调参后_KFold": tuned_k,
            "净提升": tuned_g - base_g, "最优参数": ", ".join(f"{k}={v}" for k, v in params.items()),
            "候选数": N_ITER, "耗时秒": round(took, 1)}


def fig12(res: pd.DataFrame, noise: float) -> None:
    """调参前 vs 调参后，两套 CV 口径并排。

    这张图的重点**不是**「调完更高了」，而是「调完基本没动」——
    所以标题必须把这个结论说出来，否则读者会顺着「调参图」的惯性
    以为柱子变高了。y 轴被截断到 0.78 起，正是为了让这点差异看得见；
    但截断轴会放大视觉差异，所以要把噪声水平（折间标准差）标在图上做参照。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(res))
    w = 0.2
    pairs = [("调参前_GroupKFold", "调参前", MUTED, -1.5),
             ("调参后_GroupKFold", "调参后", C_BLUE, -0.5),
             ("调参前_KFold", "调参前", MUTED, 0.5),
             ("调参后_KFold", "调参后", C_AQUA, 1.5)]
    for col, lab, color, off in pairs:
        bars = ax.bar(x + off * w, res[col], width=w, color=color,
                      edgecolor=SURFACE, linewidth=0.6)
        for b, v in zip(bars, res[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n（GroupKFold 口径）" for m in res["模型"]], color=INK_2)
    ax.set_ylabel("R^2")
    lo = float(res[["调参前_GroupKFold", "调参后_GroupKFold",
                    "调参前_KFold", "调参后_KFold"]].min().min())
    hi = float(res[["调参前_GroupKFold", "调参后_GroupKFold",
                    "调参前_KFold", "调参后_KFold"]].max().max())
    ax.set_ylim(max(0, lo - 0.03), hi + 0.035)
    ax.grid(axis="x", visible=False)

    # 把「折间标准差」画成一条噪声带：调参的收益如果淹在这条带里，
    # 那就不是「提升」，是抖动。这是这张图唯一想说的话。
    ax.axhspan(res["调参后_GroupKFold"].mean() - noise,
               res["调参后_GroupKFold"].mean() + noise,
               color=C_CRITICAL, alpha=0.07, zorder=0)
    ax.text(len(res) - 0.42, res["调参后_GroupKFold"].mean() + noise,
            f"折间标准差 ±{noise:.4f}", color=C_CRITICAL, fontsize=9,
            ha="right", va="bottom")

    # 净提升标在每个模型那组柱子的正上方（轴顶留了位置）。
    # 不能标在底部 —— y 轴从 0.78 起，底部整片都是柱子，字会压在色块上。
    for i, r in res.iterrows():
        gain = r["净提升"]
        ax.text(i, ax.get_ylim()[1] - 0.006, f"净提升 {gain:+.4f}",
                ha="center", va="top", fontsize=10,
                color=C_CRITICAL if abs(gain) < noise else INK)

    ax.set_title("调参买不到分数：GroupKFold 上 "
                 f"{res['净提升'].min():+.4f} ~ {res['净提升'].max():+.4f}，"
                 f"远小于折间标准差 ±{noise:.4f}\n"
                 "左边一对是 GroupKFold（诚实口径），右边一对是普通 KFold；"
                 "搜索目标也是 GroupKFold —— 不拿一把尺子选、另一把尺子报",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig12_tuning_v2.png")
    plt.close(fig)


def main(figs_only: bool = False) -> int:
    print("=" * 78)
    print("13 调参（第二版，GroupKFold 口径）")
    print("=" * 78)

    # LightGBM 一组就要 376 秒。改图/改文字时没必要重跑搜索，
    # 所以留一个只重画的入口，从已落盘的结果表读：
    #   uv run python src/13_tune_v2.py --figs-only
    table = RES / "tuning_results_v2.csv"
    if figs_only:
        if not table.exists():
            print(f"  找不到 {table.name}，先正常跑一次。")
            return 1
        res = pd.read_csv(table)
        print("--figs-only：跳过搜索，直接读 results/tuning_results_v2.csv 重画图")
    else:
        X, y, groups = shuffle_once(*load_xy())
        models = make_models()
        rows = []
        for name in SPACES:
            print(f"\n  {name}：{N_ITER} 组随机候选 × 5 折 GroupKFold ……")
            r = tune(name, models[name], X, y, groups)
            rows.append(r)
            print(f"    最优 {r['最优参数']}")
            print(f"    GroupKFold {r['调参前_GroupKFold']:+.4f} → "
                  f"{r['调参后_GroupKFold']:+.4f}（{r['净提升']:+.4f}）"
                  f"    普通 KFold {r['调参前_KFold']:+.4f} → {r['调参后_KFold']:+.4f}"
                  f"    {r['耗时秒']:.0f}s")
        res = pd.DataFrame(rows)
        res.to_csv(table, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 78}")
    print(f"{'=' * 78}")
    for _, r in res.iterrows():
        print(f"  {r['模型']:24s} {r['净提升']:+.4f}")

    # ---- 主表里的冠军有没有被调参？------------------------------------
    scores = RES / "model_scores_v2.csv"
    if scores.exists():
        top = pd.read_csv(scores).iloc[0]
        tuned_names = set(res["模型"])
        if top["模型"] not in tuned_names:
            print(f"\n  ⚠ 主表冠军是「{top['模型']}」"
                  f"（GroupKFold {top['GroupKFold_CV_R2']:+.4f}），但它**不在**本次调参范围内。")
            if "CatBoost" in str(top["模型"]):
                print(f"    原因是 CatBoost 单次拟合约 20 秒（要对「板块」的 129 个取值"
                      f"\n    算有序目标统计），{N_ITER} 组 × 5 折要跑几十分钟。"
                      f"\n    所以报告里必须写明：**这个冠军用的是保守默认超参，没有调过**。")
            else:
                print(f"    需要在 SPACES 里补上它的搜索空间。")

    # ---- 这个提升算不算「提升」？ ------------------------------------
    # 唯一的判据是「和折间抖动比」。0.0008 这种量级如果小于折间标准差，
    # 那它就只是一次抽样噪声，不是调参买来的分数 —— 必须说出来，
    # 否则「调参 +0.0008」会被读成「调参有效」。
    noise = float("nan")
    if scores.exists():
        s = pd.read_csv(scores)
        noise = float(s[s["模型"].isin(res["模型"])]["GroupKFold_CV_std"].mean())
        lo_g, hi_g = res["净提升"].min(), res["净提升"].max()
        print(f"\n{'=' * 78}")
        print("这个提升算不算数")
        print(f"{'=' * 78}")
        print(f"  最优参数在 GroupKFold 上的净提升：{lo_g:+.4f} ~ {hi_g:+.4f}")
        print(f"  被调模型的 GroupKFold 折间标准差均值：{noise:.4f}")
        if abs(lo_g) < noise and abs(hi_g) < noise:
            print(f"\n  两者一比就清楚了：**提升比噪声还小。**")
            print(f"  换句话说，「调参」在这份数据上没有买到分数 —— 默认超参已经贴着上限。")
            print(f"  这不是失败，是一条结论：**瓶颈在特征，不在超参。**")
            print(f"  和 §13.6 对得上：新字段的一元解释力接近 0，价值全在交互里，")
            print(f"  而交互是树模型自己分裂出来的，调叶子数/学习率并不会多造出交互。")
        else:
            print(f"\n  提升大于噪声，可以认为是真实收益。")

    setup_chinese_font()
    fig12(res, noise)
    print(f"\n  图 → figures/fig12_tuning_v2.png")
    print(f"  表 → results/tuning_results_v2.csv")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="调参（第二版）")
    ap.add_argument("--figs-only", action="store_true",
                    help="跳过搜索，从 results/tuning_results_v2.csv 重画图")
    sys.exit(main(ap.parse_args().figs_only))
