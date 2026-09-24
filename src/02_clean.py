# -*- coding: utf-8 -*-
"""
02 数据清洗与特征工程 —— 教程 §3.2「数据清洗」、§3.3「数据抽取」

从 data/lianjia_bj.csv 解析出建模可用的数值/类别特征，
并把「价格(万)」这类会与目标泄露的字段挡在建模表之外。

产出：
    data/lianjia_bj_clean.csv   建模表（不含任何价格类字段）
    results/leak_screen.csv     单特征泄露筛查结果
"""
from __future__ import annotations

import sys

import numpy as np
from pandas.api import types as pdt

from common import (CLEAN_CSV, DROP_ALWAYS, LEAK_R2_THRESHOLD, RES, TARGET,
                    build_features, feature_columns, load_raw, screen_single_features)


def main() -> int:
    df = load_raw()
    print(f"读取原始数据：{df.shape[0]} 行 × {df.shape[1]} 列\n")

    feats = build_features(df)

    # ---- 目标缺失的行无法用于监督学习 --------------------------------
    n0 = len(feats)
    feats = feats.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"目标「{TARGET}」解析：{len(feats)}/{n0} 行可用")

    # ---- 各字段解析情况 ----------------------------------------------
    print("\n各特征解析率与取值范围：")
    for c in feature_columns():
        s = feats[c]
        if pdt.is_numeric_dtype(s):
            print(f"  {c:8s} 非空 {s.notna().sum():3d}/{len(feats)}   "
                  f"min={s.min():>9.2f}  max={s.max():>9.2f}  中位数={s.median():>8.2f}")
        else:
            vc = s.value_counts(dropna=False)
            print(f"  {c:8s} 非空 {s.notna().sum():3d}/{len(feats)}   "
                  f"取值数={s.nunique()}  最高频={dict(list(vc.items())[:4])}")

    print(f"\n目标「{TARGET}」：中位数 {feats[TARGET].median():,.0f} 元/㎡，"
          f"均值 {feats[TARGET].mean():,.0f}，标准差 {feats[TARGET].std():,.0f}")
    q = feats[TARGET].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    print("  分位数：" + "  ".join(f"{k:.0%}={v:,.0f}" for k, v in q.items()))

    # ---- 泄露筛查（用含 价格_万 的完整特征集来跑，留下证据）-----------
    print("\n" + "=" * 68)
    print(f"泄露筛查：逐个特征单独预测「{TARGET}」，CV R² 超过 {LEAK_R2_THRESHOLD} 判为泄露")
    print("=" * 68)
    audit = feats.drop(columns=[TARGET])
    screen = screen_single_features(audit, feats[TARGET])
    print(screen.to_string(index=False))
    screen.to_csv(RES / "leak_screen.csv", index=False, encoding="utf-8-sig")

    leak_r2 = float(screen.loc[screen["特征"] == "价格_万", "单特征CV_R2"].iloc[0])
    print(f"""
说明：「价格(万)」的单特征 CV R² 是 {leak_r2:+.3f}，
被判为泄露。它和「{TARGET}」的相关系数只有 {audit['价格_万'].corr(feats[TARGET]):+.3f} ——
如果只看相关系数、用 0.95 当阈值，这一列会被直接放过去。但它单独预测
「{TARGET}」的 R² 有 0.42，放进模型就是自欺欺人。
因为 {TARGET} = 价格(万)×10000/面积，「价格(万)」本来就是目标的另一种写法。""")

    # ---- 写建模表：剔除全部泄露字段 -----------------------------------
    keep = feature_columns() + [TARGET]
    clean = feats[keep].copy()
    leaked = [c for c in clean.columns if c in DROP_ALWAYS and c != TARGET]
    assert not leaked, f"建模表里混进了泄露字段：{leaked}"

    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已写入：{CLEAN_CSV.name}  {clean.shape[0]} 行 × {clean.shape[1]} 列")
    print(f"  列：{', '.join(clean.columns)}")
    print(f"  已排除：{', '.join(c for c in DROP_ALWAYS if c != TARGET)}")

    # ---- 异常值：记录，但不删 -----------------------------------------
    lo, hi = feats[TARGET].quantile([0.25, 0.75])
    iqr = hi - lo
    n_out = int(((feats[TARGET] < lo - 1.5 * iqr) | (feats[TARGET] > hi + 1.5 * iqr)).sum())
    print(f"""
异常值处理：按 IQR 规则，{TARGET} 有 {n_out}/{len(feats)} 行落在 [Q1-1.5IQR, Q3+1.5IQR] 之外。
  这些点没有删。它们不是错误值，而是核心城区/远郊的真实价差 —— 地段对
  {TARGET} 的方差解释率高达 {_eta2(feats):.3f}，价格跨度大本来就是数据的主要信息。
  按 IQR 硬删会把最贵的核心区样本切掉，等于把模型唯一有效的预测依据删了。""")
    return 0


def _eta2(df) -> float:
    """地段对目标的一元方差解释率 η²。"""
    grand = df[TARGET].mean()
    vc = df["地段"].value_counts()
    ssb = sum(g * (df.loc[df["地段"] == k, TARGET].mean() - grand) ** 2 for k, g in vc.items())
    return float(ssb / ((df[TARGET] - grand) ** 2).sum())


if __name__ == "__main__":
    sys.exit(main())
