# -*- coding: utf-8 -*-
"""
03 探索性数据分析 —— 在建模之前先看清楚数据长什么样。

这一节要回答四个问题，它们决定了后面所有建模决策：
    1. 目标「单价」是什么分布？                      → fig01
    2. 面积为什么不解释单价、却能解释总价？          → fig02
    3. 哪个特征真的有用？                            → fig03 / fig04
    4. 哪个字段会骗人？                              → fig05 / fig06

产出：figures/fig01 ~ fig06
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (BASELINE, C_BLUE, C_CRITICAL, C_ORANGE, FIG, INK, INK_2,
                    LEAK_R2_THRESHOLD, SURFACE, TARGET, build_features,
                    div_cmap, load_clean, load_raw, screen_single_features,
                    seq_cmap, setup_chinese_font)

GROUP = "区县"          # 本数据集里唯一真正有解释力的特征
AUDIT_LEAKS = ["总价_万", "小区均价"]


def eta2(df: pd.DataFrame, group: str = GROUP) -> float:
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
        ax.hist(np.asarray(values, dtype=float), bins=40,
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

    fig.suptitle(f"北京二手房单价分布（{df['区县'].nunique()} 个区县，n={len(df):,}）",
                 y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_target_dist.png")
    plt.close(fig)


def fig02_area(df: pd.DataFrame) -> None:
    """面积 vs 单价（弱负相关）对比 面积 vs 总价（明显正相关）。

    这是整个练习的问题陈述：单价是「每平米」的价格，面积在定义上就被除掉了，
    所以面积几乎不含预测单价的信息。总价 = 单价 × 面积，所以面积对总价当然有效。

    总价由「单价 × 面积 ÷ 10000」还原，和原始数据里的 totalPrice 差在千分之一以内
    （因为 CSV 把单价取整了）—— 这里只用于画图，不进模型。
    """
    import matplotlib.pyplot as plt
    total = df[TARGET] * df["面积"] / 10000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, yv, ylab, note in [
        (axes[0], df[TARGET], "单价（元/㎡）", "单价 vs 面积"),
        (axes[1], total, "总价（万元）", "总价 vs 面积"),
    ]:
        ax.scatter(df["面积"], yv, s=6, color=C_BLUE, alpha=0.25,
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
    """各区县单价中位数。只有 13 个区县，全部画出来，不做「取前 N 个」。"""
    import matplotlib.pyplot as plt
    stat = (df.groupby(GROUP)[TARGET]
              .agg(中位数="median", 房源数="size")
              .sort_values("中位数"))
    overall = float(df[TARGET].median())

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    cmap = seq_cmap()
    norm = (stat["中位数"] - stat["中位数"].min()) / (stat["中位数"].max() - stat["中位数"].min() + 1e-9)
    colors = [cmap(0.18 + 0.72 * v) for v in norm]

    bars = ax.barh(stat.index, stat["中位数"], color=colors, height=0.68)
    for bar, (_, row) in zip(bars, stat.iterrows()):
        # bbox 用底色：全市中位数的虚线会从部分标签中间穿过去，套个底色挡住。
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{row['中位数']:,.0f}  (n={int(row['房源数']):,})",
                va="center", ha="left", fontsize=9.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axvline(overall, color=C_ORANGE, linewidth=2, linestyle="--")
    # 参考线的标签放在最下方那条 bar 之下的空白带里。放顶部会和标题打架。
    ax.text(overall, -0.62, f"全市中位数 {overall:,.0f}",
            color=C_ORANGE, fontsize=10, va="top", ha="center")

    ax.set_xlim(0, stat["中位数"].max() * 1.26)
    ax.set_ylim(-1.7, len(stat) - 0.5)
    ax.set_xlabel("单价中位数（元/㎡）")
    # 注意：SimHei 没有上标 ²（U+00B2），写成 η² 会变成空白方块，只能用 eta^2
    ax.set_title(f"最贵的西城是最便宜的房山的 {stat['中位数'].max() / stat['中位数'].min():.2f} 倍"
                 f"    区县对单价的方差解释率 eta^2 = {eta2(df):.3f}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig03_district_price.png")
    plt.close(fig)


def fig04_corr(df: pd.DataFrame, leaks: pd.DataFrame) -> None:
    """相关矩阵。含「总价_万」—— 它和目标的相关性只有 0.47，远不足以暴露泄露。

    leaks 是按行对齐的泄露候选列（仅用于展示，不进模型）。
    """
    import matplotlib.pyplot as plt
    d = df.copy()
    d["总价_万"] = leaks["总价_万"].values
    cols = ["面积", "室", "厅", "卫", "总层数", "房龄", "梯户比",
            "关注人数", "电梯", "满五", "近地铁", "总价_万", TARGET]
    corr = d[cols].corr()

    fig, ax = plt.subplots(figsize=(9.0, 7.4))
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
    ax.set_title("相关矩阵（蓝=负相关，红=正相关，灰=无关）\n"
                 "注意「总价_万」和单价只有 "
                 f"{corr.loc['总价_万', TARGET]:.2f} —— 详见 fig05", fontsize=12.5)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson 相关系数")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_corr_heatmap.png")
    plt.close(fig)


def fig05_leak_screen(screen: pd.DataFrame) -> None:
    """单特征交叉验证 R^2 排行 —— 本项目的泄露判据长什么样。

    这张图是整个练习里最该看的一张：两个泄露列里，只有一个冲过了阈值线。
    """
    import matplotlib.pyplot as plt
    d = screen.iloc[::-1].reset_index(drop=True)     # 最强的画在最上面
    ypos = np.arange(len(d))
    # 颜色只表示「判据拦没拦住」，不表示「是不是泄露」—— 这两件事在这张图上正好不一致，
    # 而那个不一致就是这张图存在的理由。两个已知泄露列另用红边标出来。
    caught = (d["判定"] == "泄露").to_numpy()
    audit = d["特征"].isin(AUDIT_LEAKS).to_numpy()

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    bars = ax.barh(ypos, d["单特征CV_R2"], height=0.66,
                   color=[C_CRITICAL if c else C_BLUE for c in caught],
                   edgecolor=[C_CRITICAL if a else "none" for a in audit],
                   linewidth=[1.8 if a else 0 for a in audit])

    ax.axvline(LEAK_R2_THRESHOLD, color=C_ORANGE, linewidth=2, linestyle="--")
    ax.text(LEAK_R2_THRESHOLD, len(d) - 0.15,
            f"  判据阈值 {LEAK_R2_THRESHOLD}", color=C_ORANGE, fontsize=10,
            va="center", ha="left")

    for bar, name, c, a in zip(bars, d["特征"], caught, audit):
        w = bar.get_width()
        note = ""
        if name == "总价_万":
            note = "  ← 已知泄露，判据没拦住"
        elif c:
            note = "  ← 判据判为泄露"
        ax.text(w + (0.012 if w >= 0 else -0.012), bar.get_y() + bar.get_height() / 2,
                f"{w:+.3f}" + note, va="center", ha="left" if w >= 0 else "right",
                fontsize=9.5, color=C_CRITICAL if (c or a) else INK_2)

    ax.axvline(0, color=BASELINE, linewidth=1.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["特征"])
    ax.set_xlim(-0.10, float(d["单特征CV_R2"].max()) * 1.30)
    ax.set_xlabel("只拿这一个特征去预测单价，5 折交叉验证能得到的 R^2")
    ax.grid(axis="y", visible=False)
    # 标题里的数字必须从同一张表里取。原来这里是硬编码的 "+0.223"，
    # 换一次数据集就会变成一句谎话 —— 而图上其它数字都是算出来的，
    # 读者没有理由怀疑标题。LEAK_R2_THRESHOLD 的注释里记着这个坑。
    tot = float(d.loc[d["特征"] == "总价_万", "单特征CV_R2"].squeeze())
    ax.set_title("泄露判据：单特征 CV R^2（不是相关系数）\n"
                 f"红边 = 已知的泄露列。「小区均价」被拦住；「总价_万」只有 {tot:+.3f}，"
                 "判据放它过去了",
                 fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_leak_screen.png")
    plt.close(fig)


def fig06_simpson(df: pd.DataFrame) -> None:
    """辛普森陷阱：全市看「房龄 +0.33 越老越贵」，分区看几乎归零，且各区符号还不一致。"""
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["房龄"]).copy()

    r_all = float(np.corrcoef(d["房龄"], d[TARGET])[0, 1])
    per = (d.groupby(GROUP)
             .apply(lambda s: float(np.corrcoef(s["房龄"], s[TARGET])[0, 1]),
                    include_groups=False)
             .sort_values())
    u = d[TARGET] - d.groupby(GROUP)[TARGET].transform("mean")
    a = d["房龄"] - d.groupby(GROUP)["房龄"].transform("mean")
    r_within = float(np.corrcoef(a, u)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6),
                             gridspec_kw={"width_ratios": [1, 1.05]})

    ax = axes[0]
    ax.scatter(d["房龄"], d[TARGET], s=5, color=C_BLUE, alpha=0.18, edgecolor="none")
    k, b = np.polyfit(d["房龄"], d[TARGET], 1)
    xs = np.linspace(d["房龄"].min(), d["房龄"].max(), 50)
    ax.plot(xs, k * xs + b, color=C_ORANGE, linewidth=2)
    ax.set_title(f"全部房源合并看   r = {r_all:+.3f}\n（房龄每多 10 年，单价高约 {k * 10:,.0f} 元/㎡？）",
                 fontsize=11.5)
    ax.set_xlabel("房龄（年）")
    ax.set_ylabel("单价（元/㎡）")
    ax.set_ylim(0, 160_000)

    ax = axes[1]
    ypos = np.arange(len(per))
    ax.barh(ypos, per.values, height=0.62,
            color=[C_ORANGE if v >= 0 else C_BLUE for v in per.values])
    for y, (name, v) in zip(ypos, per.items()):
        ax.text(v + (0.012 if v >= 0 else -0.012), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK_2)
    ax.axvline(0, color=BASELINE, linewidth=1.4)
    ax.axvline(r_within, color=C_CRITICAL, linewidth=2, linestyle="--")
    ax.text(r_within, len(per) - 0.3, f"  合并后 {r_within:+.3f}",
            color=C_CRITICAL, fontsize=10, va="center", ha="left")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{n}（n={int((d[GROUP] == n).sum()):,}）" for n in per.index])
    ax.set_xlim(-0.62, 0.62)
    ax.set_xlabel("该区县内部，房龄与单价的相关系数")
    ax.grid(axis="y", visible=False)
    ax.set_title("拆到各区县内部  有正有负，方向都不统一\n"
                 f"（扣掉区县均值后总体只剩 r = {r_within:+.3f}）", fontsize=11.5)

    fig.suptitle("房龄的「越老越贵」多半是区县差异伪装的"
                 f"（核心城区老而贵、远郊新而便宜），各区县 r 从 {per.min():+.2f} 到 {per.max():+.2f}",
                 y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_simpson_age.png")
    plt.close(fig)


def main() -> int:
    setup_chinese_font()
    df = load_clean()
    feats, _, _ = build_features(load_raw())
    # 必须和 02_clean.py 做同样的 dropna，行序才能和建模表逐行对上 ——
    # 否则 fig04 会把 A 房的「总价_万」贴到 B 房的单价上，还看不出错。
    feats = feats.dropna(subset=[TARGET]).reset_index(drop=True)
    assert len(feats) == len(df), f"行数对不上：feats {len(feats)} vs clean {len(df)}"
    leaks = feats[["总价_万", "小区均价"]]

    print(f"建模表：{df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"  目标「{TARGET}」中位数 {df[TARGET].median():,.0f} 元/㎡，"
          f"标准差 {df[TARGET].std():,.0f}\n")

    print("各特征与单价的相关系数（按绝对值排序）：")
    num = [c for c in df.columns if c != TARGET and pd.api.types.is_numeric_dtype(df[c])]
    corr = df[num + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    for k, v in corr.items():
        print(f"  {k:8s} {v:+.3f}")
    print(f"\n  {GROUP}（类别变量）η² = {eta2(df):.3f}   <- 唯一真正有解释力的特征")
    print(f"  共 {df[GROUP].nunique()} 个区县，全部保留做 one-hot —— "
          "取值少、每个取值都有上千条样本，不需要做「取前 N 个 + 其他」的合并。")

    print("\n单特征泄露筛查：")
    screen = screen_single_features(feats.drop(columns=[TARGET]), feats[TARGET])
    print(screen.to_string(index=False))

    fig01_target(df)
    fig02_area(df)
    fig03_district(df)
    fig04_corr(df, leaks)
    fig05_leak_screen(screen)
    fig06_simpson(df)
    print(f"\n已生成 6 张图到 {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
