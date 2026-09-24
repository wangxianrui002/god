# -*- coding: utf-8 -*-
"""
03 探索性数据分析 —— 在建模之前先看清楚数据长什么样。

这一节要回答三个问题，它们决定了后面所有建模决策：
    1. 目标「单价」是什么分布？
    2. 哪些特征真的和目标有关？（答案：几乎只有「地段」）
    3. 有没有会骗人的字段？（有：「价格(万)」和目标泄露）

产出：figures/fig01 ~ fig05
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (C_BLUE, C_CRITICAL, C_ORANGE, FIG, GRID, INK, INK_2, MUTED,
                    SURFACE, TARGET, build_features, div_cmap, load_clean,
                    load_raw, setup_chinese_font)


def eta2(df: pd.DataFrame, group: str = "地段") -> float:
    """组间方差 / 总方差 —— 分组变量对目标的一元解释力。"""
    grand = df[TARGET].mean()
    vc = df[group].value_counts()
    ssb = sum(g * (df.loc[df[group] == k, TARGET].mean() - grand) ** 2 for k, g in vc.items())
    return float(ssb / ((df[TARGET] - grand) ** 2).sum())


def fig01_target(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, values, title, xlabel in [
        (axes[0], df[TARGET], "原始单价分布", "单价（元/㎡）"),
        (axes[1], np.log10(df[TARGET]), "取对数后的单价分布", "log10(单价)"),
    ]:
        ax.hist(np.asarray(values, dtype=float), bins=32,
                color=C_BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("房源数")
        ax.grid(axis="x", visible=False)

    sk_raw = float(df[TARGET].skew())
    sk_log = float(np.log10(df[TARGET]).skew())
    axes[0].text(0.97, 0.95, f"偏度 {sk_raw:+.2f}", transform=axes[0].transAxes,
                 ha="right", va="top", color=INK_2, fontsize=11)
    axes[1].text(0.97, 0.95, f"偏度 {sk_log:+.2f}", transform=axes[1].transAxes,
                 ha="right", va="top", color=INK_2, fontsize=11)

    fig.suptitle(f"北京二手房单价分布（n={len(df)}）", y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_target_dist.png")
    plt.close(fig)


def fig02_area(df: pd.DataFrame) -> None:
    """面积 vs 单价（几乎无关）对比 面积 vs 总价（明显正相关）。

    这是整个练习的问题陈述：单价是「每平米」的价格，面积在定义上就被除掉了，
    所以面积几乎不含预测单价的信息。总价 = 单价 × 面积，所以面积对总价当然有效。
    """
    import matplotlib.pyplot as plt
    total = df[TARGET] * df["面积"] / 10000  # 万元，与「价格(万)」等价
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, yv, ylab, note in [
        (axes[0], df[TARGET], "单价（元/㎡）", "单价 vs 面积"),
        (axes[1], total, "总价（万元）", "总价 vs 面积"),
    ]:
        ax.scatter(df["面积"], yv, s=16, color=C_BLUE, alpha=0.55,
                   edgecolor="none", label="房源")
        k, b = np.polyfit(df["面积"], yv, 1)
        xs = np.linspace(df["面积"].min(), df["面积"].max(), 50)
        ax.plot(xs, k * xs + b, color=C_ORANGE, linewidth=2, label="线性拟合")
        r = float(np.corrcoef(df["面积"], yv)[0, 1])
        ax.set_title(f"{note}   r = {r:+.3f}")
        ax.set_xlabel("面积（㎡）")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left")

    fig.suptitle("为什么「单价」是个更难的问题：面积在定义上就被除掉了", y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_area_vs_price.png")
    plt.close(fig)


def fig03_district(df: pd.DataFrame) -> None:
    """各地段单价中位数（按房源数取前 15）。"""
    import matplotlib.pyplot as plt
    stat = (df.groupby("地段")[TARGET]
              .agg(中位数="median", 房源数="size")
              .sort_values("房源数", ascending=False)
              .head(15)
              .sort_values("中位数"))
    overall = float(df[TARGET].median())

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    cmap = plt.get_cmap("Blues")
    norm = (stat["中位数"] - stat["中位数"].min()) / (stat["中位数"].max() - stat["中位数"].min() + 1e-9)
    colors = [cmap(0.35 + 0.55 * v) for v in norm]

    bars = ax.barh(stat.index, stat["中位数"], color=colors, height=0.68)
    for bar, (_, row) in zip(bars, stat.iterrows()):
        # 注意 int()：groupby 里混了 median 之后，size 这一列会被提升成 float，
        # 不转就会打出 "(n=4.0)" 这种房源数。
        # bbox 用底色：全市中位数的虚线会从部分标签中间穿过去，套个底色挡住。
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{row['中位数']:,.0f}  (n={int(row['房源数'])})",
                va="center", ha="left", fontsize=9.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(overall, color=C_ORANGE, linewidth=2, linestyle="--")
    # 参考线的标签放在最下方那条 bar 之下的空白带里。放顶部会和标题打架。
    ax.text(overall, -0.62, f"全市中位数 {overall:,.0f}",
            color=C_ORANGE, fontsize=10, va="top", ha="center")

    ax.set_xlim(0, stat["中位数"].max() * 1.30)
    ax.set_ylim(-1.7, len(stat) - 0.5)
    ax.set_xlabel("单价中位数（元/㎡）")
    # 注意：SimHei 没有上标 ²（U+00B2），写成 η² 会变成空白方块，只能用 eta^2
    ax.set_title(f"各地段单价差异悬殊    地段对单价的方差解释率 eta^2 = {eta2(df):.3f}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_district_price.png")
    plt.close(fig)


def fig04_corr(df: pd.DataFrame, raw_feats: pd.DataFrame) -> None:
    """相关矩阵。含「价格(万)」—— 它和目标的相关性只有 0.66，不足以暴露泄露。"""
    import matplotlib.pyplot as plt
    d = df.copy()
    d["价格_万"] = raw_feats["价格_万"].values  # 仅用于展示，不进模型
    cols = ["面积", "房间数", "厅数", "总层数", "建成年", "关注人数",
            "近地铁", "南北通透", "价格_万", TARGET]
    corr = d[cols].corr()

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(corr.values, cmap=div_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(v) > 0.55 else INK_2)
    ax.set_title("相关矩阵（蓝=负相关，红=正相关，灰=无关）")
    fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson 相关系数")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_corr_heatmap.png")
    plt.close(fig)


def fig05_confound(df: pd.DataFrame) -> None:
    """辛普森悖论：整体看「房子越新越便宜」，但同地段内看是越新越贵。"""
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["建成年"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    r_all = float(np.corrcoef(d["建成年"], d[TARGET])[0, 1])
    axes[0].scatter(d["建成年"], d[TARGET], s=16, color=C_BLUE, alpha=0.5, edgecolor="none")
    k, b = np.polyfit(d["建成年"], d[TARGET], 1)
    xs = np.linspace(d["建成年"].min(), d["建成年"].max(), 50)
    axes[0].plot(xs, k * xs + b, color=C_ORANGE, linewidth=2)
    axes[0].set_title(f"全部房源合并看   r = {r_all:+.3f}\n（越新反而越便宜？）")
    axes[0].set_xlabel("建成年")
    axes[0].set_ylabel("单价（元/㎡）")

    dm = d.copy()
    dm["u"] = d[TARGET] - d.groupby("地段")[TARGET].transform("mean")
    dm["a"] = d["建成年"] - d.groupby("地段")["建成年"].transform("mean")
    r_within = float(np.corrcoef(dm["a"], dm["u"])[0, 1])
    axes[1].scatter(dm["a"], dm["u"], s=16, color=C_BLUE, alpha=0.5, edgecolor="none")
    k2, b2 = np.polyfit(dm["a"], dm["u"], 1)
    xs2 = np.linspace(dm["a"].min(), dm["a"].max(), 50)
    axes[1].plot(xs2, k2 * xs2 + b2, color=C_ORANGE, linewidth=2)
    axes[1].set_title(f"扣掉地段均值后再看   r = {r_within:+.3f}\n（同地段内，越新越贵）")
    axes[1].set_xlabel("建成年（对本地段均值取差）")
    axes[1].set_ylabel("单价（对本地段均值取差）")

    fig.suptitle("辛普森悖论：核心城区老而贵，远郊新而便宜，掩盖了真实关系", y=1.03, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_confound.png")
    plt.close(fig)


def main() -> int:
    setup_chinese_font()
    df = load_clean()
    raw_feats = build_features(load_raw())

    print(f"清洗后数据：{df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"  目标「{TARGET}」中位数 {df[TARGET].median():,.0f} 元/㎡，"
          f"标准差 {df[TARGET].std():,.0f}\n")

    print("各特征与单价的相关系数（按绝对值排序）：")
    num = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    for k, v in corr.items():
        print(f"  {k:8s} {v:+.3f}")
    print(f"\n  地段（类别变量）η² = {eta2(df):.3f}   <- 唯一真正有解释力的特征")
    print(f"  143 个地段里有 {(df['地段'].value_counts() <= 3).sum()} 个只有 3 套及以下房源，")
    print("  所以地段编码必须保留全部 143 个取值，不能只留高频的、其余归入「其他」——")
    print("  那样会把 2/3 的样本塞进一个桶里，信号全丢。")

    fig01_target(df)
    fig02_area(df)
    fig03_district(df)
    fig04_corr(df, raw_feats)
    fig05_confound(df)
    print(f"\n已生成 5 张图到 {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
