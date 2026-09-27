# -*- coding: utf-8 -*-
"""
11 清洗与特征工程（第二版）—— 从挂牌原始数据构造建模表

产出：
    data/house_v2_clean.csv         建模表（特征 + 目标 + 小区分组列）
    results/leak_screen_v2.csv      单特征泄露筛查结果
    results/dropped_rows_v2.csv     被清洗规则剔除的记录

与第一版 02_clean.py 的关系：清洗哲学完全一致（阈值放宽、异常值不删、缺失值保留
交给 Pipeline、稀有类别两步走），换的只是数据集和字段。凡是能复用的规则都从
common.py 导入，不重新实现。
"""
from __future__ import annotations

import sys

import pandas as pd
from pandas.api import types as pdt

from common import (LEAK_R2_THRESHOLD, MIN_CATEGORY_COUNT, OTHER, RES, TARGET,
                    screen_single_features)
from common_v2 import (AREA_MAX_V2, AREA_MIN_V2, CLEAN_V2, DROP_ALWAYS_V2, GROUP_COL,
                       PRICE_MAX_V2, PRICE_MIN_V2, build_features, eta2,
                       feature_columns, load_raw)

# 故意留在特征集里跑一遍筛查，好让结果落进 leak_screen_v2.csv。
# 「总价_万」是第一版那个「数值判据拦不住、只有语义判据拦得住」的泄露陷阱，
# 在本数据集里一模一样地复现了 —— 这正是换数据后仍然要跑同一套判据的理由。
AUDIT_LEAKS = ["总价_万"]


def main() -> int:
    raw = load_raw()
    print("=" * 68)
    print("11 清洗与特征工程（第二版）")
    print("=" * 68)
    print(f"读取原始数据：{raw.shape[0]:,} 行 × {raw.shape[1]} 列")

    feats, dropped, stats = build_features(raw)

    print(f"\n{'=' * 68}")
    print("清洗")
    print(f"{'=' * 68}")
    print(f"  原始               {stats['原始']:>9,} 行")
    print(f"  → 清洗规则剔除      {-stats['剔除']:>9,} 行")
    print(f"  = 建模表           {stats['清洗后']:>9,} 行")
    print(f"\n  规则：{TARGET} 不在 [{PRICE_MIN_V2:,}, {PRICE_MAX_V2:,}] 元/㎡、"
          f"面积不在 [{AREA_MIN_V2:g}, {AREA_MAX_V2:g}] ㎡")

    if stats["稀有类别"] or stats["样本不足"]:
        print(f"\n  稀有类别处理（少于 {MIN_CATEGORY_COUNT} 行的取值）：")
        for c, d in stats["样本不足"].items():
            print(f"    {c:6s} " + "  ".join(f"{k}({v}行)" for k, v in d.items())
                  + f"  → 合计不足 {MIN_CATEGORY_COUNT} 行，按缺失处理")
        for c, d in stats["稀有类别"].items():
            n = sum(d.values())
            print(f"    {c:6s} {len(d):>3} 个稀有取值共 {n:,} 行 → 并成「{OTHER}」"
                  f"（例：{', '.join(list(d)[:3])}）")

    feats = feats.dropna(subset=[TARGET]).reset_index(drop=True)

    # ---- 特征取值范围 -------------------------------------------------
    print(f"\n{'=' * 68}")
    print("特征取值范围")
    print(f"{'=' * 68}")
    for c in feature_columns():
        s = feats[c]
        miss = int(s.isna().sum())
        if pdt.is_numeric_dtype(s):
            print(f"  {c:8s} 缺失 {miss:>6,}   min={s.min():>10,.2f}  "
                  f"max={s.max():>10,.2f}  中位={s.median():>9,.2f}")
        else:
            vc = s.value_counts(dropna=False)
            top = "  ".join(f"{k}:{v:,}" for k, v in list(vc.items())[:3])
            print(f"  {c:8s} 缺失 {miss:>6,}   取值 {s.nunique():>3} 个   {top}")

    print(f"\n目标「{TARGET}」：中位数 {feats[TARGET].median():,.0f} 元/㎡，"
          f"均值 {feats[TARGET].mean():,.0f}，标准差 {feats[TARGET].std():,.0f}")
    q = feats[TARGET].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    print("  分位数：" + "  ".join(f"{k:.0%}={v:,.0f}" for k, v in q.items()))

    # ---- 新地理特征的解释力（对照第一版的区县 η²=0.627）---------------
    print(f"\n{'=' * 68}")
    print("新地理特征对目标的方差解释率 eta^2")
    print(f"{'=' * 68}")
    print(f"  （第一版的「区县」13 个取值，eta^2 = 0.627，是当时最强的单一特征）")
    for c in ["板块", "环线", "朝向", "装修", "楼层位置"]:
        v = eta2(feats, c)
        print(f"  {c:6s} {feats[c].nunique():>3} 个取值   eta^2 = {v:.3f}")

    # ---- 泄露筛查 -----------------------------------------------------
    print(f"\n{'=' * 68}")
    print(f"泄露筛查：逐个数值特征单独预测目标，CV R² 超过 {LEAK_R2_THRESHOLD} 判为泄露")
    print(f"{'=' * 68}")
    audit = feats.drop(columns=[TARGET])
    screen = screen_single_features(audit, feats[TARGET])
    print(screen.head(10).to_string(index=False))
    screen.to_csv(RES / "leak_screen_v2.csv", index=False, encoding="utf-8-sig")

    r2_total = float(screen.loc[screen["特征"] == "总价_万", "单特征CV_R2"].iloc[0])
    corr_total = audit["总价_万"].corr(feats[TARGET])
    ratio = feats["总价_万"] * 10000 / feats["面积"]
    margin = r2_total - LEAK_R2_THRESHOLD

    # 第一版同一列的实测值，用来对照「同一个泄露列换份数据结局会变」
    v1 = RES / "leak_screen.csv"
    if v1.exists():
        s1 = pd.read_csv(v1)
        r2_v1 = float(s1.loc[s1["特征"] == "总价_万", "单特征CV_R2"].iloc[0])
        v1_line = (f"  第一版同一列的 R² 只有 {r2_v1:+.4f}，比阈值低 "
                   f"{LEAK_R2_THRESHOLD - r2_v1:.4f}，**数值判据完全拦不住**；")
        v1_tail = ("同一个泄露列、同一套阈值，换一份数据结论就反过来 ——\n"
                   "  说明 R² 判据的判别力本身依赖数据，不能当成保证。")
    else:
        v1_line, v1_tail = "", ""

    print(f"""
  「总价_万」—— 和第一版完全相同的陷阱，换数据后又出现了：
    与目标的相关系数 {corr_total:+.3f}，单特征 CV R² {r2_total:+.4f}，
    阈值 {LEAK_R2_THRESHOLD} —— 这次**刚刚越过线 {margin:+.4f}**，拦住了，但属于侥幸。
{v1_line}
    而它始终是实打实的泄露：「总价 × 10000 ÷ 面积」与目标的相关系数是
    {ratio.corr(feats[TARGET]):.10f}，最大差 {(ratio - feats[TARGET]).abs().max():,.1f} 元/㎡
    （这点差是总价只存到整万元造成的舍入）。换句话说，总价和面积两列一起递进去，
    等于把目标原样递了进去。数值判据看不见它的原因是：**线性模型自己造不出
    「比值」这个非线性组合**，判据的有效性依赖于测试它用的模型类别。

  {v1_tail}
  **结论：列名/语义那一层（正则 price）才是能依赖的那一道。**
  这条线在这个数据集上离阈值只差 {abs(margin):.4f}，而合法特征最强的「近地铁」
  是 {float(screen.loc[screen['特征'] == '近地铁', '单特征CV_R2'].iloc[0]):+.4f} ——
  阈值仍然分得开合法特征，但已经贴不住泄露列了。""")

    # ---- 写建模表 -----------------------------------------------------
    keep = feature_columns() + [TARGET, GROUP_COL]
    clean = feats[keep].copy()

    leaked = [c for c in clean.columns if c in DROP_ALWAYS_V2 and c != TARGET]
    assert not leaked, f"建模表里混进了必须排除的列：{leaked}"
    assert GROUP_COL not in feature_columns(), (
        f"「{GROUP_COL}」是分组列，绝不能出现在特征里 —— "
        f"同小区的价格天然含本行信息，那是 target encoding 级别的泄露")
    assert not [c for c in clean.columns if c in AUDIT_LEAKS], "演示用的泄露列漏进了建模表"
    assert TARGET not in feature_columns(), "目标列不能出现在特征清单里"

    clean.to_csv(CLEAN_V2, index=False, encoding="utf-8-sig")
    mb = CLEAN_V2.stat().st_size / 1e6
    print(f"\n{'=' * 68}")
    print(f"已写入：{CLEAN_V2.name}  {clean.shape[0]:,} 行 × {clean.shape[1]} 列（{mb:.1f} MB）")
    print(f"  特征 {len(feature_columns())} 列：{', '.join(feature_columns())}")
    print(f"  目标 1 列：{TARGET}")
    print(f"  分组 1 列：{GROUP_COL}（{clean[GROUP_COL].nunique():,} 个小区，"
          f"平均 {len(clean) / clean[GROUP_COL].nunique():.1f} 套/小区）"
          f" —— 只用于 GroupKFold，不进特征")

    cols = [c for c in [TARGET, "面积", "室", "楼层位置", "朝向", "板块", "总价_万"]
            if c in dropped.columns]
    dropped.head(200)[cols].to_csv(RES / "dropped_rows_v2.csv", index=False,
                                   encoding="utf-8-sig")
    print(f"  被剔除记录的抽样 → results/dropped_rows_v2.csv")

    # ---- 异常值：记录，但不删 -----------------------------------------
    lo, hi = feats[TARGET].quantile([0.25, 0.75])
    iqr = hi - lo
    n_out = int(((feats[TARGET] < lo - 1.5 * iqr)
                 | (feats[TARGET] > hi + 1.5 * iqr)).sum())
    print(f"""
异常值处理：按 IQR 规则，{TARGET} 有 {n_out:,}/{len(feats):,} 行落在
  [Q1-1.5IQR, Q3+1.5IQR] 之外。和第一版一样**没有删** —— 这些是核心城区与远郊的
  真实价差，不是错误值。第一版按此处理是对的，第二版沿用同一判断。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
