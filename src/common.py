"""北京二手房单价预测的共用配置与建模管线。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
WEB = ROOT / "web"
RAW_CSV = DATA / "lianjia_bj_raw.csv"
CLEAN_CSV = DATA / "lianjia_bj_clean.csv"

SEED = 42
YEAR = 2017
TARGET = "单价"
NUMERIC = [
    "面积", "室", "厅", "卫", "总层数", "房龄", "梯户比",
    "关注人数", "电梯", "满五", "近地铁",
]
CATEGORICAL = ["区县", "楼层位置", "装修", "建筑类型", "建筑结构"]
FEATURES = NUMERIC + CATEGORICAL


def load_clean() -> pd.DataFrame:
    """只允许预先选定的房屋属性进入建模表。"""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"缺少 {CLEAN_CSV}；请先运行 01_download.py 和 02_clean.py"
        )
    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    expected = set(FEATURES + [TARGET])
    if set(df.columns) != expected:
        raise ValueError(f"建模表列不符。缺少：{expected - set(df.columns)}；多出：{set(df.columns) - expected}")
    if df[TARGET].isna().any():
        raise ValueError("目标单价含缺失值")
    return df


def make_preprocessor() -> ColumnTransformer:
    """填补操作放在 Pipeline 内，每次交叉验证只从训练折学习统计量。"""
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def make_models() -> dict[str, Pipeline]:
    """三种不同的回归方法；超参数固定，避免另设庞大的搜索流程。"""
    algorithms = {
        "线性回归": LinearRegression(),
        "KNN 回归": KNeighborsRegressor(n_neighbors=5),
        "决策树回归": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=20, random_state=SEED
        ),
    }
    return {
        name: Pipeline([("preprocess", make_preprocessor()), ("model", algorithm)])
        for name, algorithm in algorithms.items()
    }


def setup_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.18
