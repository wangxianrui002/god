# -*- coding: utf-8 -*-
"""
02 数据清洗与特征工程 —— 教程 §3.2「数据清洗」、§3.3「数据抽取」

从 data/lianjia_bj_raw.csv（全量 318,851 行）切出建模窗口、解析特征，
并把「总价」「小区均价」这两个会与目标泄露的字段挡在建模表之外。

产出：
    data/lianjia_bj_clean.csv   建模表（不含任何价格类字段）
    results/leak_screen.csv     单特征泄露筛查结果
    results/dropped_rows.csv    被清洗规则剔除的记录（抽样 200 条，便于复核）
"""
from __future__ import annotations

import sys

import pandas as pd
from pandas.api import types as pdt

from common import (AREA_MAX, AREA_MIN, CLEAN_CSV, DROP_ALWAYS, LEAK_R2_THRESHOLD,
                    MIN_CATEGORY_COUNT, OTHER, PRICE_MAX, PRICE_MIN, RES, TARGET,
                    WINDOW_YEARS, build_features, feature_columns, load_raw,
                    screen_single_features)

# 与目标互为函数关系、或名字里带价格、或天然含本行信息的列。
# 它们**故意**留在特征集里跑一遍筛查，好让结果落进 results/leak_screen.csv ——
# 这是「为什么不能只看相关系数」的证据。
AUDIT_LEAKS = ["总价_万", "小区均价"]


def eta2(df: pd.DataFrame, group: str = "区县") -> float:
    """组间方差 / 总方差 —— 分组变量对目标的一元解释力。"""
    grand = df[TARGET].mean()
    vc = df[group].value_counts()
    ssb = sum(g * (df.loc[df[group] == k, TARGET].mean() - grand) ** 2 for k, g in vc.items())
    return float(ssb / ((df[TARGET] - grand) ** 2).sum())


def main() -> int:
    raw = load_raw()
    print(f"读取原始数据：{raw.shape[0]:,} 行 × {raw.shape[1]} 列")
    print(f"  成交时间跨度：{raw['tradeTime'].min()} ~ {raw['tradeTime'].max()}")

    feats, dropped, stats = build_features(raw)

    print(f"\n{'=' * 68}")
    print("切窗口 + 清洗")
    print(f"{'=' * 68}")
    print(f"  全量               {stats['原始']:>9,} 行")
    print(f"  → 只取 {WINDOW_YEARS[0]}–{WINDOW_YEARS[1]} 年成交  "
          f"{stats['窗口内']:>9,} 行   （丢掉 {stats['原始'] - stats['窗口内']:,} 行）")
    print(f"  → 清洗规则剔除      {-stats['剔除']:>9,} 行")
    print(f"  = 建模表           {stats['清洗后']:>9,} 行")
    print(f"\n  清洗规则：单价不在 [{PRICE_MIN:,}, {PRICE_MAX:,}] 元/㎡、"
          f"面积不在 [{AREA_MIN:g}, {AREA_MAX:g}] ㎡、总价缺失或非正")

    print("\n  为什么要切窗口：全量跨 2002–2018，北京单价中位数在这期间从 3.8 万涨到 6.6 万，")
    print("  全量建模等于让模型用同一套系数解释相差 1.75 倍的两个市场。")

    if stats["稀有类别"] or stats["样本不足"]:
        print(f"\n  稀有类别处理（少于 {MIN_CATEGORY_COUNT} 行的取值）：")
        for c, d in stats["样本不足"].items():
            print(f"    {c:6s} " + "  ".join(f"{k}({v}行)" for k, v in d.items())
                  + f"  → 合计不足 {MIN_CATEGORY_COUNT} 行，按缺失处理，"
                    "由 Pipeline 归入最常见的一档")
        for c, d in stats["稀有类别"].items():
            print(f"    {c:6s} " + "  ".join(f"{k}({v}行)" for k, v in d.items())
                  + f"  → 并成「{OTHER}」")
        print("    不处理的话，one-hot 会给这些取值一人一个自由系数 ——")
        print("    实测「建筑类型=平房」只有 1 行，却拿到了 +11,363 元/㎡ 的系数，")
        print("    页面上谁选「平房」，这个数字就凭空冒出来。")

    feats = feats.dropna(subset=[TARGET]).reset_index(drop=True)

    # ---- 各字段解析情况 ----------------------------------------------
    print(f"\n{'=' * 68}")
    print("各特征取值范围")
    print(f"{'=' * 68}")
    for c in feature_columns():
        s = feats[c]
        miss = s.isna().sum()
        if pdt.is_numeric_dtype(s):
            print(f"  {c:8s} 缺失 {miss:>6,}   "
                  f"min={s.min():>10,.2f}  max={s.max():>10,.2f}  中位={s.median():>9,.2f}")
        else:
            vc = s.value_counts(dropna=False)
            top = "  ".join(f"{k}:{v:,}" for k, v in list(vc.items())[:4])
            print(f"  {c:8s} 缺失 {miss:>6,}   取值 {s.nunique():>2} 个   {top}")

    print(f"\n目标「{TARGET}」：中位数 {feats[TARGET].median():,.0f} 元/㎡，"
          f"均值 {feats[TARGET].mean():,.0f}，标准差 {feats[TARGET].std():,.0f}")
    q = feats[TARGET].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    print("  分位数：" + "  ".join(f"{k:.0%}={v:,.0f}" for k, v in q.items()))

    # ---- 泄露筛查 -----------------------------------------------------
    print(f"\n{'=' * 68}")
    print(f"泄露筛查：逐个特征单独预测「{TARGET}」，CV R² 超过 {LEAK_R2_THRESHOLD} 判为泄露")
    print(f"{'=' * 68}")
    audit = feats.drop(columns=[TARGET])
    screen = screen_single_features(audit, feats[TARGET])
    print(screen.head(12).to_string(index=False))
    if len(screen) > 12:
        print(f"  …（共 {len(screen)} 个数值特征，完整结果见 results/leak_screen.csv）")
    screen.to_csv(RES / "leak_screen.csv", index=False, encoding="utf-8-sig")

    print("\n两种泄露，两道判据各抓一个：")
    for c in AUDIT_LEAKS:
        r2 = float(screen.loc[screen["特征"] == c, "单特征CV_R2"].iloc[0])
        corr = audit[c].corr(feats[TARGET])
        print(f"  {c:8s} 相关系数 {corr:+.3f}   单特征 CV R² {r2:+.4f}   → "
              f"{'判为泄露' if r2 > LEAK_R2_THRESHOLD else '数值判据放过了它'}")
    legit = screen[~screen["特征"].isin(AUDIT_LEAKS)]
    print(f"  合法特征里最强的：{legit.iloc[0]['特征']} R² {legit.iloc[0]['单特征CV_R2']:+.4f}、"
          f"{legit.iloc[1]['特征']} R² {legit.iloc[1]['单特征CV_R2']:+.4f}")

    # 总价那条为什么必须靠语义判断：它是目标的恒等重建，只是线性模型造不出比值
    ratio = feats["总价_万"] * 10000 / feats["面积"]
    print(f"""
  「小区均价」—— 数值判据抓得住。它的名字里没有价格字样，列名黑名单拦不住；
  相关系数 {audit['小区均价'].corr(feats[TARGET]):+.3f} 用 0.95 的阈值也拦不住。
  但让它单独预测一次目标，R² 是 {float(screen.loc[screen['特征'] == '小区均价', '单特征CV_R2'].iloc[0]):+.3f}，一测就露馅。
  它天然含本行信息：同小区的均价是用包括这一套在内的房源算出来的。

  「总价_万」—— 数值判据**抓不住**，只有语义判断抓得住。它和目标的相关系数
  只有 {audit['总价_万'].corr(feats[TARGET]):+.3f}，单特征 R² 只有
  {float(screen.loc[screen['特征'] == '总价_万', '单特征CV_R2'].iloc[0]):+.3f}，两道数值判据都会放它过去。
  可它是实打实的泄露 —— 「总价 × 10000 ÷ 面积」和目标的相关系数是
  {ratio.corr(feats[TARGET]):.10f}，最大差 {(ratio - feats[TARGET]).abs().max():.1f} 元/㎡
  （这点差就是 CSV 里总价只存到整万元造成的舍入）。换句话说，总价和面积
  两列一起递进去，等于把目标原样递了进去。

  之所以数值判据看不见它，是因为**线性模型自己造不出「比值」这个非线性组合**。
  判据的有效性依赖于测试它用的模型类别 —— 换个能学比值的模型，
  这一列的 R² 立刻就是 1.0。所以列名/语义那一层不能省，R² 那一层也不能省。""")

    # ---- 写建模表：剔除全部泄露字段 -----------------------------------
    keep = feature_columns() + [TARGET]
    clean = feats[keep].copy()
    leaked = [c for c in clean.columns if c in DROP_ALWAYS and c != TARGET]
    assert not leaked, f"建模表里混进了泄露字段：{leaked}"
    assert not [c for c in clean.columns if c in AUDIT_LEAKS], "演示用的泄露列漏进了建模表"

    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    mb = CLEAN_CSV.stat().st_size / 1e6
    print(f"\n已写入：{CLEAN_CSV.name}  {clean.shape[0]:,} 行 × {clean.shape[1]} 列"
          f"（{mb:.1f} MB）")
    print(f"  列：{', '.join(clean.columns)}")
    print(f"  已排除：{', '.join(c for c in DROP_ALWAYS if c != TARGET)}")

    # 被剔除的记录留个抽样，便于人工复核清洗规则是否误杀
    cols = ["单价", "面积", "室", "总价_万", "小区均价"]
    dropped.head(200)[[c for c in cols if c in dropped.columns]].to_csv(
        RES / "dropped_rows.csv", index=False, encoding="utf-8-sig")
    print("  被剔除记录的抽样 200 条 → results/dropped_rows.csv")

    # ---- 异常值：记录，但不删 -----------------------------------------
    lo, hi = feats[TARGET].quantile([0.25, 0.75])
    iqr = hi - lo
    n_out = int(((feats[TARGET] < lo - 1.5 * iqr) | (feats[TARGET] > hi + 1.5 * iqr)).sum())
    print(f"""
异常值处理：按 IQR 规则，{TARGET} 有 {n_out:,}/{len(feats):,} 行落在
  [Q1-1.5IQR, Q3+1.5IQR] 之外。这些点**没有删**：它们不是错误值，而是核心城区
  与远郊的真实价差 —— 区县对{TARGET}的方差解释率 eta^2 = {eta2(feats):.3f}，
  价格跨度大本来就是数据的主要信息。按 IQR 硬删会把最贵的核心区样本切掉，
  等于把模型最有效的预测依据删了。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
