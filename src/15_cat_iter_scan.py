# -*- coding: utf-8 -*-
"""
15 CatBoost 迭代数单向扫描（第二版附录）

**为什么要有这个脚本**：主表（`12_model_compare.py`）里所有模型一律 400 轮，于是
CatBoost 排在四个 GBDT 的最后。报告里据此下了一个克制得多的结论 ——
「CatBoost 比另外三个差」这个读法**没有依据**，证据是：只动迭代数、其余超参不动，
它的分数还在单调上升，也就是**没训够**，而不是「这个库不行」。

这条结论的分量全在那三个数上（400 / 800 / 1600 轮各是多少分）。它原先是一次
控制台里手跑的临时实验，数字直接抄进了 README —— 那正是本项目最反对的做法：
**要引用就得先写下来**，手抄的数字早晚会和跑出来的对不上，而且谁也复现不了。
所以把它补成一个脚本，落盘到 `results/cat_iter_scan.csv`。

**协议与主表严格一致**，这样 400 轮那一行必然等于主表里 CatBoost 的 GroupKFold 分数：
同一份建模表、同一个 `shuffle_once(SEED)` 打乱、同一个 `GroupKFold(5)` 按小区分组、
同一套预处理（`GbdtFrame`，类别走原生处理）、除 `n_estimators` 外超参一个不动。

运行约 10 分钟（1600 轮那一次单折就要 80 秒上下，5 折 × 3 组）。

产出：
    results/cat_iter_scan.csv
"""
from __future__ import annotations

import sys
import time

import pandas as pd
from sklearn.pipeline import Pipeline

from common import RES, SEED
from common_v2 import (CATEGORICAL, CAT_IDX, NUMERIC, CatBoostCategorical,
                       GbdtFrame, group_cv_scores, load_xy, shuffle_once)

# 与 `common_v2.make_models()` 里的 tree 超参保持一致：除了迭代数，其余原样。
TREE = dict(learning_rate=0.06)
ITERATIONS = (400, 800, 1600)


def catboost_pipeline(n_estimators: int):
    """主表里 CatBoost 那一行的复刻，只把 n_estimators 换成扫描值。"""
    return Pipeline([
        ("pre", GbdtFrame(numeric=NUMERIC, categorical=CATEGORICAL)),
        ("model", CatBoostCategorical(cat_features=CAT_IDX, random_seed=SEED,
                                      n_estimators=n_estimators, **TREE)),
    ])


def main() -> int:
    X, y, groups = load_xy()
    # 必须先打乱：GroupKFold 不打乱，而原始 CSV 按板块/环线排序。
    # 不打乱的话每一折落在连续地理区块上，分数与主表不可比。理由见 shuffle_once 文档。
    X, y, groups = shuffle_once(X, y, groups)
    print(f"数据 {len(X):,} 行，分组（小区）{groups.nunique():,} 个\n")

    rows = []
    for n in ITERATIONS:
        t0 = time.perf_counter()
        scores = group_cv_scores(catboost_pipeline(n), X, y, groups)
        dt = time.perf_counter() - t0
        print(f"  {n:>5} 轮  GroupKFold R² = {scores.mean():+.4f}  "
              f"（逐折 {' '.join(f'{s:+.4f}' for s in scores)}，{dt:.0f} 秒）")
        rows.append({"迭代数": n, "GroupKFold_CV_R2": round(float(scores.mean()), 4),
                     "GroupKFold_CV_std": round(float(scores.std()), 4),
                     "各折": " / ".join(f"{s:.4f}" for s in scores),
                     "拟合秒": round(dt, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(RES / "cat_iter_scan.csv", index=False, encoding="utf-8-sig")

    # 结论由脚本自己下，不留给读者凭印象猜：只有单调上升才叫「没训够」。
    rising = df["GroupKFold_CV_R2"].is_monotonic_increasing
    gain = df["GroupKFold_CV_R2"].iloc[-1] - df["GroupKFold_CV_R2"].iloc[0]
    print(f"\n  400 → {ITERATIONS[-1]} 轮{'单调上升' if rising else '**不是**单调上升'}，"
          f"共 {gain:+.4f}")
    if not rising:
        print("  ⚠️ 不再单调上升 —— 报告里「没训够」那条结论必须改，别照抄旧数字。")
    print(f"  → results/cat_iter_scan.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
