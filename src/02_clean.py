"""从链家原始成交记录提取 2017 年的房屋属性与单价。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLEAN_CSV, CATEGORICAL, FEATURES, RAW_CSV, TARGET, YEAR

DISTRICTS = {
    1: "东城", 2: "丰台", 3: "亦庄开发区", 4: "大兴", 5: "房山",
    6: "昌平", 7: "朝阳", 8: "海淀", 9: "石景山", 10: "西城",
    11: "通州", 12: "门头沟", 13: "顺义",
}
BUILDINGS = {1: "塔楼", 2: "平房", 3: "板塔结合", 4: "板楼"}
RENOVATIONS = {1: "毛坯", 2: "简装", 3: "精装", 4: "豪华"}
FLOORS = {"底", "低", "中", "高", "顶"}


def number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"缺少 {RAW_CSV}；请先运行 01_download.py")

    raw = pd.read_csv(RAW_CSV, encoding="utf-8", low_memory=False)
    year = number(raw["tradeTime"].astype(str).str[:4])
    d = raw.loc[year.eq(YEAR)].copy()
    n_year = len(d)

    price = number(d["price"])
    area = number(d["square"])
    total = number(d["totalPrice"])
    valid = price.between(5_000, 200_000) & area.between(10, 1_000) & total.gt(0)
    d = d.loc[valid].copy()
    out = pd.DataFrame(index=d.index)

    out[TARGET] = number(d["price"])
    out["面积"] = number(d["square"])
    out["室"] = number(d["livingRoom"])
    out["厅"] = number(d["drawingRoom"])
    out["卫"] = number(d["bathRoom"])

    floor = d["floor"].astype("string").str.strip()
    out["总层数"] = number(floor.str.extract(r"(\d+)$")[0])
    level = floor.str.extract(r"^(\D+?)\s*\d+$")[0]
    out["楼层位置"] = level.where(level.isin(FLOORS))

    age = YEAR - number(d["constructionTime"])
    out["房龄"] = age.where(age.between(0, YEAR - 1949))
    ladder = number(d["ladderRatio"])
    out["梯户比"] = ladder.where(ladder.between(0, 10))
    out["关注人数"] = number(d["followers"])
    out["电梯"] = number(d["elevator"])
    out["满五"] = number(d["fiveYearsProperty"])
    out["近地铁"] = number(d["subway"])

    out["区县"] = number(d["district"]).map(DISTRICTS)
    out["装修"] = number(d["renovationCondition"]).map(RENOVATIONS)
    out["建筑类型"] = number(d["buildingType"]).map(BUILDINGS)
    structure = number(d["buildingStructure"])
    out["建筑结构"] = structure.map(
        lambda value: f"结构{int(value)}" if pd.notna(value) else np.nan
    )

    # 类别极少的取值容易使模型记住个别样本。
    for column in CATEGORICAL:
        counts = out[column].value_counts()
        rare = counts[counts < 200].index
        if len(rare):
            replacement = "其他" if int(counts.loc[rare].sum()) >= 200 else np.nan
            out[column] = out[column].replace(dict.fromkeys(rare, replacement))

    clean = out[FEATURES + [TARGET]].reset_index(drop=True)
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"2017 年记录：{n_year:,}；剔除明显错误：{n_year - len(clean):,}")
    print(f"建模表：{len(clean):,} 行、{len(FEATURES)} 个特征，保存到 {CLEAN_CSV}")
    print("总价、小区均价、成交时间和标识符均未进入建模表。")


if __name__ == "__main__":
    main()
