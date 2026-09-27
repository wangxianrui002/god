# -*- coding: utf-8 -*-
"""
第二版公共模块 —— 换用特征更丰富的二手房数据集（挂牌数据，含朝向/楼层/小区/环线）。

与 common.py 的分工：

  * 凡**与数据集无关**的东西 —— 随机种子、防泄露判据（`DROP_ALWAYS` 之外的三道关）、
    绘图配色、交叉验证、RMSE —— 一律从 `common.py` 导入，绝不在这里重复定义。
    那套规则只能有一份，有两份早晚会有一份先过期。
  * 这里只放第一版没有的：新数据集的字段映射、清洗阈值、特征工程，
    以及 GBDT 需要的预处理器。

第一版数据（链家成交记录 2002–2018）仍然保留在 `common.py` 里，两版对照着看才有意义 ——
「换数据」和「换模型」各带来多少提升，是本次实验最值得量化的一件事。
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# 与数据集无关的规则，全部沿用第一版，不重新定义
from common import (DATA, FIG, LEAK_R2_THRESHOLD, MIN_CATEGORY_COUNT, OTHER, RES,
                    ROOT, SEED, TARGET, cv_scores, guard_no_leakage, make_cv, rmse,
                    screen_single_features, setup_chinese_font, seq_cmap, div_cmap)

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
RAW_V2 = DATA / "house_v2_raw.csv"      # 10 产出：北京子集（全量 16 城 45 万行不入库）
CLEAN_V2 = DATA / "house_v2_clean.csv"  # 11 产出：建模表

# --------------------------------------------------------------------------
# 数据来源
# --------------------------------------------------------------------------
# Kaggle: xiaopaohadoop/second-hand-housing-dataset
#   SH-house-dataset.csv  455,566 行 × 22 列，16 个城市，挂牌数据（不是成交数据）
#   北京 73,685 行，是本项目第一版（43,213 行）的 1.7 倍
#
# 相比第一版**多出来的字段**（这正是换数据的目的）：
#   朝向 orientation、楼层位置 floor_level、装修 decoration、小区 community、
#   板块 neighborhood、环线（从 p2_desc 解析）、近公园（从 tags 解析）、
#   30 日带看量 visit_30_days
#
# 相比第一版**失去的字段**：
#   经纬度、精确地铁步行距离（北京子集全空）、成交日期
#
# 重要口径差异：本数据是**挂牌价**（卖方要价），第一版是**成交价**。
# 两者不可直接比数值高低，但建模方法论完全一致。
CRAWL_YEAR = 2026      # 数据抓取年份，用于算房龄（build_year 最大 2026）

# --------------------------------------------------------------------------
# 清洗阈值
# --------------------------------------------------------------------------
# 与第一版同样的哲学：定得很宽，只剔录入错误，不用来做数据「整形」。
# 实测：单价最低 4,576（保留 1 行），>30 万只有 17 行；面积最低 3.4 ㎡（明显是车位/储物间）。
PRICE_MIN_V2, PRICE_MAX_V2 = 5_000, 300_000
AREA_MIN_V2, AREA_MAX_V2 = 10.0, 1_000.0

# --------------------------------------------------------------------------
# 字段
# --------------------------------------------------------------------------
# 数值特征。近地铁/满五/近公园 是从 tags 里拆出来的 0/1，所以也放数值列。
NUMERIC = ["面积", "室", "厅", "卫", "总层数", "房龄", "关注人数", "带看30天",
           "近地铁", "满五", "近公园"]

# 类别特征。板块 270 个取值、朝向 11 个、环线 6 个（有序）。
CATEGORICAL = ["朝向", "楼层位置", "装修", "板块", "环线"]

# 小区：**只用于 GroupKFold 分组，绝不进特征**。
# 6,852 个小区、平均 10.8 套；当特征用等于 target encoding ——
# 「这个小区的均价」天然含本行信息，与第一版的 communityAverage 是同一个陷阱。
GROUP_COL = "小区"

# 环线由内到外，是有序的
RING_LEVELS = ["二环内", "二环至三环", "三环至四环", "四环至五环", "五环至六环", "六环外"]

# 绝不允许进入特征的列
DROP_ALWAYS_V2 = [
    "unit_price",        # 目标本身
    "total_price(w)",    # 泄露：× 10000 ÷ 面积 就是目标（实测相关 0.999979）
    "id", "house_id",    # 标识符
    "city",              # 常量（已筛成北京）
    "community",         # 小区：只作分组，见 GROUP_COL
    "p1_desc", "p2_desc",  # 冗余文本（环线已从中解析出来）
    "subway_info",       # 北京子集 100% 为空
    "tags",              # 已拆成 近地铁/满五/近公园
    "neighborhood",      # 已重命名为「板块」
]

# 与第一版共用同一套判据阈值，但列名黑名单要加上本数据集的实际列名
LEAK_NAME_PATTERN_V2 = re.compile(r"价格|总价|单价|首付|万元|_万$|price|Price|total_price")


def feature_columns() -> list[str]:
    return NUMERIC + CATEGORICAL


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_V2, low_memory=False)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_V2, low_memory=False)


def load_xy(df: pd.DataFrame | None = None):
    """返回 (X, y, groups)。

    X **严格由 `feature_columns()` 决定**，绝不写成「除目标外的所有列」——
    建模表里还有一列「小区」，那样写会让它作为特征悄悄混进模型，
    而「小区」是 target encoding 级别的泄露（同小区的成交价天然含本行信息）。
    """
    df = load_clean() if df is None else df
    X = df[feature_columns()].copy()
    y = df[TARGET].copy()
    groups = df[GROUP_COL].astype(str).copy()
    return X, y, groups


# --------------------------------------------------------------------------
# 特征工程
# --------------------------------------------------------------------------
def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_tags(s) -> set:
    """tags 形如 '["满五年","近地铁"]'，解析失败就当没有标签。"""
    if not isinstance(s, str):
        return set()
    try:
        v = json.loads(s)
        return set(v) if isinstance(v, list) else set()
    except (ValueError, TypeError):
        return set()


def merge_rare(s: pd.Series, min_count: int = MIN_CATEGORY_COUNT):
    """稀有类别两步走 —— 与第一版完全相同的处理。

    为什么要两步：只做第一步（并成「其他」）不管用。第一版里「建筑类型=平房」
    只有 1 行，并成「其他」之后这个桶本身还是 1 行，照样拿到 +10,460 元/㎡ 的系数 ——
    换个名字并没有解决问题。所以并完还要检查桶够不够大，不够大就按缺失处理。
    """
    vc = s.value_counts()
    rare = [k for k, n in vc.items() if n < min_count]
    if not rare:
        return s, {}, {}
    merged = {k: int(vc[k]) for k in rare}
    n_rare = int(sum(vc[k] for k in rare))
    if n_rare < min_count:
        return s.where(~s.isin(rare), np.nan), {}, merged
    return s.where(~s.isin(rare), OTHER), merged, {}


def build_features(raw: pd.DataFrame):
    """从原始 22 列构造建模表。返回 (特征表, 被剔除的行, 统计)。"""
    d = raw.copy()
    out = pd.DataFrame(index=d.index)

    # ---- 目标与泄露候选 ----
    out[TARGET] = _num(d["unit_price"])
    out["总价_万"] = _num(d["total_price(w)"])      # 只用于泄露演示

    # ---- 房屋属性 ----
    out["面积"] = _num(d["area_sqm"])
    out["室"] = _num(d["room_num"])
    out["厅"] = _num(d["hall_num"])
    out["卫"] = _num(d["bathroom_num"])
    out["总层数"] = _num(d["floor_total"])

    # 房龄 = 抓取年 - 建成年。build_year 实测 1945~2026，没有越界值。
    out["房龄"] = CRAWL_YEAR - _num(d["build_year"])

    out["关注人数"] = _num(d["attention_count"])
    out["带看30天"] = _num(d["visit_30_days"])

    # ---- tags → 三个 0/1 ----
    tag_sets = d["tags"].map(_parse_tags)
    out["近地铁"] = tag_sets.map(lambda s: int("近地铁" in s))
    out["满五"] = tag_sets.map(lambda s: int("满五年" in s))
    out["近公园"] = tag_sets.map(lambda s: int("近公园" in s))

    # ---- 类别 ----
    out["朝向"] = d["orientation"].astype("string").str.strip()
    out["楼层位置"] = d["floor_level"].astype("string").str.strip()
    out["装修"] = d["decoration"].astype("string").str.strip()
    out["板块"] = d["neighborhood"].astype("string").str.strip()

    # 环线藏在 p2_desc 的最后一段：「德胜门 新街口外大街甲8号院 · 二环至三环」
    ring = d["p2_desc"].astype("string").str.rsplit("·", n=1).str[-1].str.strip()
    out["环线"] = ring.where(ring.isin(RING_LEVELS), pd.NA)

    # ---- 小区：只作分组 ----
    out[GROUP_COL] = d["community"].astype("string").str.strip()

    # ---- 清洗 ----
    n0 = len(out)
    keep = (out[TARGET].between(PRICE_MIN_V2, PRICE_MAX_V2)
            & out["面积"].between(AREA_MIN_V2, AREA_MAX_V2))
    dropped = out[~keep].copy()
    out = out[keep].copy()
    n1 = len(out)

    # ---- 稀有类别 ----
    merged_all, missing_all = {}, {}
    for c in CATEGORICAL:
        out[c], merged, miss = merge_rare(out[c])
        if merged:
            merged_all[c] = merged
        if miss:
            missing_all[c] = miss

    stats = {"原始": n0, "清洗后": n1, "剔除": len(dropped),
             "稀有类别": merged_all, "样本不足": missing_all}
    return out, dropped, stats


# --------------------------------------------------------------------------
# 预处理器
# --------------------------------------------------------------------------
# 类别列在输出里的位置是固定的（数值列之后），GBDT 要按索引声明类别特征
CAT_IDX = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))


class ToCategory(BaseEstimator, TransformerMixin):
    """把类别列转成 pandas category dtype。

    三个 GBDT 库对类别特征的约定各不相同，实测下来 **pandas category dtype**
    是唯一三家通吃的写法（LGBM 自动识别、XGBoost 要 enable_categorical=True、
    CatBoost 按索引声明 cat_features）。写在 Pipeline 里，三种模型就能共用同一套预处理。
    """

    def __init__(self, n_numeric: int = len(NUMERIC)):
        self.n_numeric = n_numeric

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for i in range(self.n_numeric, X.shape[1]):
            X.iloc[:, i] = X.iloc[:, i].astype("category")
        return X


def make_preprocessor(one_hot: bool = True, scale_numeric: bool = True):
    """两种预处理，对应两类模型：

    * ``one_hot=True``  —— 给线性模型（线性回归 / 岭回归 / KNN）。
      数值列中位数填补 + 标准化；类别列众数填补 + one-hot。
    * ``one_hot=False`` —— 给 GBDT。数值列中位数填补（树不需要标准化）；
      类别列众数填补后转成 category dtype，交给各自的**原生类别处理**。

    为什么 GBDT 不用 one-hot：板块有 270 个取值，one-hot 之后维度爆炸，而树模型
    原生类别分裂比 one-hot 更能利用「同一个板块」这个信息。CatBoost 更是靠
    有序目标统计吃饭的，喂 one-hot 等于废掉它的看家本领。
    """
    if one_hot:
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUMERIC),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), CATEGORICAL),
        ])
    return Pipeline([
        ("imp", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL),
        ])),
        ("tocat", ToCategory()),
    ])


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
def make_models() -> dict:
    """第一版讲过的线性模型（做对照）+ 文献推荐的 GBDT 家族 + 堆叠集成。

    GBDT 超参在对比阶段一律用同一套保守值，**留到 14_tune 再调** ——
    否则「哪个模型好」和「谁调得更狠」两件事会混在一起，比出来的结果不可信。
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor

    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def build(model, one_hot=True, scale=True):
        return Pipeline([("pre", make_preprocessor(one_hot, scale)), ("model", model)])

    tree = dict(n_estimators=400, learning_rate=0.06, random_state=SEED)

    return {
        # ---- 第一版用过的，做对照 ----
        "均值基线 DummyRegressor": build(DummyRegressor(strategy="mean"), scale=False),
        "线性回归 LinearRegression": build(LinearRegression()),
        "岭回归 Ridge": build(Ridge(alpha=1.0)),
        "K近邻 KNN(k=5)": build(KNeighborsRegressor(n_neighbors=5)),

        # ---- GBDT 家族 ----
        # sklearn 自带的 HistGradientBoosting，无需第三方依赖，作为 GBDT 的下限基准
        "直方图梯度提升 HistGBR": build(
            HistGradientBoostingRegressor(
                categorical_features=CAT_IDX, random_state=SEED, **tree)),
        "LightGBM": build(LGBMRegressor(verbose=-1, **tree)),
        "XGBoost": build(XGBRegressor(enable_categorical=True, tree_method="hist",
                                      verbosity=0, **tree)),
        "CatBoost": build(CatBoostRegressor(verbose=0, cat_features=CAT_IDX,
                                            random_seed=SEED, **tree)),
    }


def make_stack(estimators: dict):
    """堆叠集成：三个 GBDT 的输出喂给一个 Ridge 元学习器。

    文献里一致的做法（CatBoost + XGBoost + LightGBM + Ridge meta）——
    多个研究都报告堆叠比任何单一模型都好。用 `cv=make_cv()` 做内部交叉验证
    生成元特征，避免基模型在训练数据上「背答案」再喂给元学习器（那是另一种泄露）。
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge

    return StackingRegressor(
        estimators=[(k, v) for k, v in estimators.items()],
        final_estimator=Ridge(alpha=1.0),
        cv=make_cv(), n_jobs=-1)


# --------------------------------------------------------------------------
# 分组交叉验证
# --------------------------------------------------------------------------
def eta2(df: pd.DataFrame, group: str, target: str = TARGET) -> float:
    """组间方差 / 总方差 —— 分组变量对目标的一元解释力。

    第一版里这个函数在 02_clean.py 和 03_eda.py 各写了一份，是不必要地重复。
    第二版只在这里定义一次。
    """
    grand = df[target].mean()
    vc = df[group].value_counts()
    ssb = sum(g * (df.loc[df[group] == k, target].mean() - grand) ** 2
              for k, g in vc.items() if pd.notna(k))
    return float(ssb / ((df[target] - grand) ** 2).sum())


def make_group_cv():
    """按「小区」分组的 5 折交叉验证。

    这是第一版 README §十一 #1 承认但没修的乐观偏差：同一个小区可能有多套房，
    普通 KFold 会把它们分到训练集和验证集两侧，等于让模型「见过类似样本」。
    第一版没有小区标识列，只能算了；**本数据集有 community，所以这一条现在能修**。
    """
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=5)


def group_cv_scores(model, X, y, groups):
    """返回按小区分组的 5 折 R² 数组。"""
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(model, X, y, cv=make_group_cv(), groups=groups,
                        scoring="r2", error_score="raise", n_jobs=1)
    assert not np.isnan(s).any(), "分组交叉验证返回了 NaN，说明有折拟合失败"
    return s
