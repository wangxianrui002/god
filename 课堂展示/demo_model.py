"""课堂 Demo 的模型配置快照，与上级项目 src/common.py 保持一致。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
CLEAN_CSV = ROOT / "data" / "lianjia_bj_clean.csv"
SEED = 42
TARGET = "单价"
NUMERIC = ["面积", "室", "厅", "卫", "总层数", "房龄", "梯户比", "关注人数", "电梯", "满五", "近地铁"]
CATEGORICAL = ["区县", "楼层位置", "装修", "建筑类型", "建筑结构"]
FEATURES = NUMERIC + CATEGORICAL


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    expected = set(FEATURES + [TARGET])
    if set(df.columns) != expected or df[TARGET].isna().any():
        raise ValueError("Demo 数据列或目标值与模型设置不符")
    return df


def make_demo_model() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=SEED)),
    ])
