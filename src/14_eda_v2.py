# -*- coding: utf-8 -*-
"""
14 探索性分析（第二版）—— 新字段到底带来了什么

换数据的动机是「第一版缺朝向/楼层/小区/板块，所以 R² 卡在 0.70」。
换完之后要正面回答一个问题：**这些新字段的一元解释力有多大？**

答案是**分成两半**的，这也是这张图唯一值得看的地方：

  地理维度（板块 0.642、环线 0.507）—— 看着很强，但**它不新**。
    第一版「区县」13 个取值就有 0.627，板块只是把同一件事切得更细（129 个取值），
    一元解释力只从 0.627 挪到 0.642。地理信息在第一版就已经吃满了。
  非地理维度（朝向 0.037、装修 0.010、楼层位置 0.002）—— 这才是真正新的信息轴，
    而它的一元解释力**低一个数量级**，比第一版的「近地铁」（0.1056）还不如。

所以只从这张图看，应该得出「换数据几乎白换」的结论 —— 可实测模型分数
明明涨了（最好模型 +0.1595，同一个 Ridge 也有 +0.0917）。两个数字对不上。

对不上的部分只能来自**特征之间的交互**：「同一个板块里朝向好的更贵」
「同一个小区里楼层不同价不同」—— 这类信息不出现在任何单变量的 η² 里，
只存在于组合中。线性模型造不出组合项，树模型按分裂逐层组合，这才把它变成分数。
这正好解释了「换数据」和「换模型」两件事为什么必须一起做。

产出：
    figures/fig13_eta2_v2.png
    results/eta2_v2.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import (BASELINE, C_BLUE, C_CRITICAL, C_ORANGE, FIG, INK_2,
                    MUTED, RES, SURFACE, setup_chinese_font)
from common_v2 import GROUP_COL, eta2, load_clean

# 第一版的实测值（results/ 里没有存 η²，这里是 README §三 记载的数字）
V1_DISTRICT_ETA2 = 0.627
V1_NOTE = "第一版最强特征\n「区县」13 个取值"

# 本版要考察的字段，按 η² 从大到小画
FIELDS = ["板块", "环线", "朝向", "装修", "楼层位置"]

# 哪些字段和地理有关。分开着色是这张图的全部重点：
# 蓝色两根是「地理，但第一版已经吃过」，橙色三根是「真正的新信息轴」。
GEO_FIELDS = {"板块", "环线"}


def fig13(df: pd.DataFrame, vals: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    # 把第一版的「区县」当成一根参照柱插进来，读者才知道新字段是涨了还是跌了
    names = [V1_NOTE] + list(vals)
    scores = [V1_DISTRICT_ETA2] + [vals[k] for k in vals]
    # 蓝色 = 地理（第一版已吃过），橙色 = 真正的新信息轴。
    # 楼层位置的 0.0016 低到柱子和 0 分不出来，用红色提醒「这就是零」。
    colors = [MUTED] + [C_BLUE if k in GEO_FIELDS
                        else (C_CRITICAL if vals[k] < 0.005 else C_ORANGE)
                        for k in vals]

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    x = np.arange(len(names))
    bars = ax.bar(x, scores, width=0.62, color=colors, edgecolor=SURFACE, linewidth=0.8)
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK_2)
    # 0.002 画在 0~0.8 的轴上等于零高度，柱子会「消失」——
    # 不标一句，读者会以为图坏了，而不是以为这个字段真的没用。
    bx = bars[-1].get_x() + bars[-1].get_width() / 2
    # 箭头落在柱子右侧的空白处，别穿过「0.002」那个数值标签
    ax.annotate("柱子高度≈0\n（不是画漏了）", xy=(bx + 0.24, 0.004), xytext=(bx + 0.05, 0.10),
                ha="center", va="bottom", fontsize=9, color=C_CRITICAL,
                arrowprops=dict(arrowstyle="->", color=C_CRITICAL, linewidth=1.1))
    ax.axhline(V1_DISTRICT_ETA2, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)),
               zorder=0)
    ax.text(len(names) - 0.45, V1_DISTRICT_ETA2 + 0.018,
            f"第一版「区县」的水位线 {V1_DISTRICT_ETA2:.3f}", color=MUTED,
            fontsize=9, ha="right", va="bottom")

    # 在「板块/环线」和「朝向/装修/楼层位置」中间画一道分隔，标出两种解读
    ax.axvline(2.5, color=BASELINE, linewidth=1.3, linestyle=(0, (3, 3)), zorder=0)
    ax.text(1.0, max(scores) * 1.14, "地理维度：强，但第一版就有\n（板块 129 个取值只是把 13 个区县切得更细）",
            ha="center", va="center", fontsize=9.5, color=C_BLUE)
    ax.text(4.0, max(scores) * 1.14, "非地理维度：真正的新信息轴\n而它的一元解释力低一个数量级",
            ha="center", va="center", fontsize=9.5, color=C_ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(names, color=INK_2, fontsize=9.5)
    ax.set_ylabel("eta^2（组间方差 / 总方差）")
    ax.set_ylim(0, max(scores) * 1.28)
    ax.grid(axis="x", visible=False)
    ax.set_title("换数据之后，新字段的一元解释力有多大？\n"
                 "地理那两根不新（区县早已这个水位），真正新的朝向/装修/楼层位置接近 0 —— "
                 "可模型分数明明涨了",
                 loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig13_eta2_v2.png")
    plt.close(fig)


def main() -> int:
    df = load_clean()
    print("=" * 78)
    print("14 探索性分析（第二版）")
    print(f"{'=' * 78}")
    print(f"建模表 {len(df):,} 行 × {df.shape[1]} 列；"
          f"{df[GROUP_COL].nunique():,} 个小区")

    print(f"\n{'=' * 78}")
    print("新字段的一元解释力 eta^2")
    print(f"{'=' * 78}")
    print(f"  参照：第一版「区县」13 个取值，eta^2 = {V1_DISTRICT_ETA2}（第一版最强特征）")
    vals = {}
    for c in FIELDS:
        v = eta2(df, c)
        vals[c] = v
        n = df[c].nunique()
        kind = "地理" if c in GEO_FIELDS else "非地理"
        flag = "  ← 与第一版最强特征同级" if v >= 0.5 else ""
        print(f"  {c:6s} {n:>4} 个取值  [{kind}]  eta^2 = {v:.4f}{flag}")

    # 结论要按「地理 / 非地理」分开下，不能笼统说「新字段很弱」——
    # 环线 0.507 一点都不弱，弱的是它**不新**。
    geo = {k: vals[k] for k in vals if k in GEO_FIELDS}
    non = {k: vals[k] for k in vals if k not in GEO_FIELDS}
    best = max(vals, key=vals.get)
    print(f"\n  地理维度：{', '.join(f'「{k}」{v:.4f}' for k, v in geo.items())}"
          f"  —— 看着强，但第一版「区县」就有 {V1_DISTRICT_ETA2}，"
          f"\n            最好的一根只高了 {vals[best] - V1_DISTRICT_ETA2:+.4f}。"
          "地理信息在第一版就已经吃满，换数据只是切得更细。")
    print(f"  非地理维度：{', '.join(f'「{k}」{v:.4f}' for k, v in non.items())}"
          "  —— 这才是真正新的信息轴，"
          "\n            而它全部低于 0.05，比第一版的「近地铁」(0.1056) 还不如。")

    # ---- 关键反差 -----------------------------------------------------
    # 落差要拿**同一个模型**比才有意义：Ridge 换数据是一段，Ridge 换 GBDT 是另一段。
    # 用「第一版冠军 → 第二版冠军」会把两件事混在一起，说不清是谁的功劳。
    scores, v1_path = RES / "model_scores_v2.csv", RES / "model_scores.csv"
    if scores.exists() and v1_path.exists():
        s2, s1 = pd.read_csv(scores), pd.read_csv(v1_path)
        r1 = float(s1.loc[s1["模型"].str.contains("Ridge\\("), "5折CV_R2"].iloc[0])
        r2 = float(s2.loc[s2["模型"].str.contains("岭回归 Ridge"), "KFold_CV_R2"].iloc[0])
        # 「换模型」这一步取**主口径 GroupKFold 的冠军**（HistGBR），不取 KFold 的最大值。
        # KFold 的最大值是堆叠，但它只在普通 KFold 上赢（见 12 的 fig10），
        # 拿它来讲「换模型值多少」等于用被高估的尺子量收益。这里必须和 README §九 对齐。
        champ = s2.loc[s2["GroupKFold_CV_R2"].idxmax(), "模型"]
        gb = float(s2.loc[s2["GroupKFold_CV_R2"].idxmax(), "KFold_CV_R2"])
        gb_name = champ
        kf_best = float(s2["KFold_CV_R2"].max())
        kf_best_name = s2.loc[s2["KFold_CV_R2"].idxmax(), "模型"]
        gain_data, gain_model = r2 - r1, gb - r2
        print(f"""
{'=' * 78}
这张图最该讲的一件事：**它和模型分数是矛盾的**
{'=' * 78}
  一元解释力：地理维度已经饱和 —— 最强的「{best}」{vals[best]:.3f}，
              只比第一版的区县高 {vals[best] - V1_DISTRICT_ETA2:+.3f}；
              真正新开的非地理维度全在 0.05 以下（最低的「楼层位置」
              只有 {vals['楼层位置']:.4f}，等于没有）。
  模型分数：  同一个 Ridge，换数据后从 {r1:+.4f} 涨到 {r2:+.4f}（{gain_data:+.4f}）；
              同一份新数据，换 {gb_name} 再涨到 {gb:+.4f}（{gain_model:+.4f}）。

  如果「R² 由特征的一元解释力决定」，两组数字都对不上。
  对不上的原因：**R² 也由特征之间的交互决定**。
  「同一个板块里朝向好的更贵」「同一个小区里楼层不同价不同」——
  这类信息在任何一个单变量的 eta^2 里都看不见，它只存在于组合中。

  但这里有个第一版没料到的结果：**换数据连线性模型都涨了 {gain_data:+.4f}**。
  原以为「Ridge 造不出交互项，所以吃不下新字段」；实测它吃得下不少。
  多出来的解释力来自**地理切分变细**（13 个区县 → 129 个板块 + 6 个环线），
  而不是来自朝向/装修/楼层位置 —— 后者一元 η² 全在 0.05 以下，
  只有树模型按分裂逐层组合，才把它们变成分数。""")
        print("\n  所以两件事的价值量级相当，都要做："
              f"\n    换数据（同一 Ridge）      {gain_data:+.4f}"
              f"\n    换模型（同一份数据）      {gain_model:+.4f}"
              f"\n    合计                      {gain_data + gain_model:+.4f}"
              f"（{r1:+.4f} → {gb:+.4f}）")
        # 两段都只能在普通 KFold 口径下比 —— 第一版没有小区列，GroupKFold 算不出来。
        # 但「哪个模型算冠军」必须按主口径 GroupKFold 定，否则就是用被高估的尺子选模型。
        print("\n  ⚠ 口径说明：上面两段的**数值**都是普通 KFold（第一版没有小区列，"
              "GroupKFold 算不出来）；\n    但「换模型」这一步选的冠军是按**主口径 GroupKFold** 定的："
              f"{gb_name}。\n    普通 KFold 的最大值其实属于「{kf_best_name}」"
              f"（{kf_best:+.4f}，比 {gb_name} 高 {kf_best - gb:+.4f}），\n    "
              "但它只在 KFold 上赢、GroupKFold 上反而更低 —— "
              "拿它算收益等于用偏乐观的尺子量收益，见 figures/fig10_model_compare_v2.png。")

    # 落盘成 CSV，而不是只打印：展示网页（06_export.py）要引用这几个数，
    # 让它读文件而不是抄一份常量进来 —— 抄一份早晚会和这里跑出来的对不上。
    pd.DataFrame([
        {"字段": c, "取值数": int(df[c].nunique()),
         "类别": "地理" if c in GEO_FIELDS else "非地理", "eta2": round(vals[c], 4)}
        for c in FIELDS
    ]).to_csv(RES / "eta2_v2.csv", index=False, encoding="utf-8-sig")

    setup_chinese_font()
    fig13(df, vals)
    print("\n  图 → figures/fig13_eta2_v2.png")
    print("  表 → results/eta2_v2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
