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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 与数据集无关的规则，全部沿用第一版，不重新定义
from common import (DATA, MIN_CATEGORY_COUNT, OTHER, SEED, TARGET, _num, cv_scores,
                    make_cv)

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

# 类别特征。清洗后的取值数：板块 129（合并稀有类别之前 270）、
# 朝向 11、装修 4、楼层位置 4、环线 6（有序）。实测 η² 见 README §13.6。
CATEGORICAL = ["朝向", "楼层位置", "装修", "板块", "环线"]

# 小区：**只用于 GroupKFold 分组，绝不进特征**。
# 6,833 个小区、平均 10.8 套；当特征用等于 target encoding ——
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


class GbdtFrame(TransformerMixin, BaseEstimator):
    """GBDT 的预处理：逐列填补，并把类别列转成 pandas ``category`` dtype。

    **为什么不用 ColumnTransformer**：它把各列 hstack 成一个数组，float64 与 object
    混在一起会被统一提升成 object —— 连数值列也变成对象类型。三个 GBDT 库拿到这种
    数组，轻则把数字当字符串处理，重则直接报错。这里逐列构造 DataFrame，dtype 才守得住。

    **为什么是 category dtype**：三个库对类别特征的约定各不相同，实测下来
    pandas category 是唯一三家通吃的写法（LightGBM 自动识别、XGBoost 要
    ``enable_categorical=True``、CatBoost 按索引声明 ``cat_features``）。
    写成同一个 Transformer，三种模型就能共用同一套预处理。

    注意 ``transform`` 里必须用 ``X[c] = ...`` 这种**标签赋值**。写成
    ``X.iloc[:, i] = X.iloc[:, i].astype("category")`` 是无效的：iloc 赋值是
    就地写入已有的 object 块，pandas 会把 Categorical 还原成原值，dtype 一点没变，
    而且不报任何错 —— 这个坑真的踩过一次。

    **类别表必须在 fit 时定死**，这是第二个踩过的坑。写成
    ``s.astype("category")`` 的话，类别表是按**当前这一份数据**现推的：
    训练折推一套、验证折又推另一套，顺序未必相同。三个库里
    XGBoost 是按 ``cat.codes``（类别在类别表里的**下标**）读数据的，
    下标一错位，整列的含义就全变了 —— 它不报错，只是把「南北」当成「东」，
    于是 GroupKFold 的 R² 掉成负数。症状是「换了个 CV 方案模型就崩了」，
    根因却在编码。LightGBM 和 CatBoost 按类别**取值**处理，所以没事。
    """

    def __init__(self, numeric: list[str] | None = None,
                 categorical: list[str] | None = None):
        self.numeric = NUMERIC if numeric is None else numeric
        self.categorical = CATEGORICAL if categorical is None else categorical

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.num_fill_ = {c: pd.to_numeric(X[c], errors="coerce").median()
                          for c in self.numeric}
        self.cat_fill_, self.categories_ = {}, {}
        for c in self.categorical:
            m = X[c].astype("string").mode()
            fill = m.iloc[0] if len(m) else pd.NA
            self.cat_fill_[c] = fill
            # 类别表在 fit 时**定死**，transform 不再重新推断。理由见类文档。
            vc = X[c].astype("string").where(lambda s: s.notna(), fill).value_counts()
            self.categories_[c] = list(vc.index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c in self.numeric:
            v = pd.to_numeric(X[c], errors="coerce")
            out[c] = v.fillna(self.num_fill_[c]).astype("float64")
        for c in self.categorical:
            s = X[c].astype("string")
            s = s.where(s.notna(), self.cat_fill_[c])
            # 固定类别表：验证折里没在训练折出现过的取值 → NaN（按缺失处理），
            # 而不是让 pandas 按本折数据另排一套类别序号
            out[c] = pd.Categorical(s, categories=self.categories_[c])
        return out


def make_preprocessor(one_hot: bool = True):
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
    return GbdtFrame()


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class CatBoostCategorical(RegressorMixin, BaseEstimator):
    """CatBoost，但把 ``cat_features`` 挪到 ``fit`` 里声明。

    注意继承顺序是 ``(RegressorMixin, BaseEstimator)`` —— **mixin 必须写在前面**。
    写成 ``(BaseEstimator, RegressorMixin)`` 时 ``BaseEstimator.__sklearn_tags__``
    排在 MRO 前面且不调用 ``super()``，于是 ``RegressorMixin`` 那版永远轮不到，
    ``estimator_type`` 一直是 ``None``，``is_regressor()`` 返回 False，
    堆叠集成会拒绝它：「The estimator Pipeline should be a regressor.」
    同样报错信息完全指不到继承顺序上，只能靠 ``is_regressor`` 逐个体检才看得出来。

    **为什么不能直接在构造器里传**：``CatBoostRegressor.get_params()`` 返回的
    ``cat_features`` 是列表的**副本**，不是传进去的那个对象；而 ``sklearn.clone``
    有一条硬断言 —— 构造器必须把参数原样存回来（``param1 is not param2`` 就报错）。
    clone 是 ``cross_val_score`` 和 ``StackingRegressor`` 每次都要走的路径，
    所以只要做交叉验证就一定会撞上：

        RuntimeError: Cannot clone object CatBoostRegressor(...),
        as the constructor either does not set or modifies parameter cat_features

    把声明挪进 ``fit``，构造器参数就只剩普通标量，clone 自然通过。
    另有一个实测事实：CatBoost **不会**从 pandas category dtype 自动识别类别列，
    不声明就直接报错（"has dtype 'category' but is not in cat_features list"），
    所以这一步不能省。

    这里的参数都是显式列出的，不能用 ``**kwargs`` —— ``get_params`` 只认
    ``__init__`` 的具名参数，写成 ``**kwargs`` 的话 clone 出来的新对象会把
    ``n_estimators`` 之类的设置全丢掉，而且不报错。
    """

    def __init__(self, cat_features=None, n_estimators: int = 400,
                 learning_rate: float = 0.06, random_seed: int = SEED,
                 verbose: int = 0):
        self.cat_features = cat_features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.verbose = verbose

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            cat_features=self.cat_features, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, random_seed=self.random_seed,
            verbose=self.verbose)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_models() -> dict:
    """第一版讲过的线性模型（做对照）+ 文献推荐的 GBDT 家族 + 堆叠集成。

    GBDT 超参在对比阶段一律用同一套保守值，**留到 14_tune 再调** ——
    否则「哪个模型好」和「谁调得更狠」两件事会混在一起，比出来的结果不可信。
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor

    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    def build(model, one_hot=True):
        return Pipeline([("pre", make_preprocessor(one_hot)), ("model", model)])

    # 三个库的随机种子参数名不一样（CatBoost 是 random_seed），所以不放进公共字典，
    # 各自显式传，避免重复传参或传了个不被识别的名字被静默忽略。
    tree = dict(n_estimators=400, learning_rate=0.06)

    return {
        # ---- 第一版用过的，做对照 ----
        # 线性模型吃 one-hot：它们只能在线性组合里找信号，类别必须展开成 0/1 列。
        "均值基线 DummyRegressor": build(DummyRegressor(strategy="mean")),
        "线性回归 LinearRegression": build(LinearRegression()),
        "岭回归 Ridge": build(Ridge(alpha=1.0)),
        "K近邻 KNN(k=5)": build(KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),

        # ---- GBDT 家族：一律 one_hot=False，走原生类别处理 ----
        #
        # 这一条必须显式写。默认值 True 会让它们全都拿到 one-hot 后的 numpy 数组，
        # 而 CatBoost 会当场报错（「data 是浮点数组，但你声明了 cat_features」）；
        # 另外三家**不报错**，只是默默按普通数值列处理那 165 维 0/1 列 ——
        # 等于把 CatBoost 的有序目标统计、LightGBM 的类别分裂全部废掉，
        # 「哪个模型更强」比出来的就是「谁更抗折腾」，毫无意义。
        "直方图梯度提升 HistGBR": build(
            # sklearn 自带的 HistGBR，无需第三方依赖，作为 GBDT 的下限基准。
            # 注意它的迭代次数参数叫 max_iter，不是另外三个库的 n_estimators。
            HistGradientBoostingRegressor(categorical_features=CAT_IDX,
                                          max_iter=tree["n_estimators"],
                                          learning_rate=tree["learning_rate"],
                                          random_state=SEED),
            one_hot=False),
        "LightGBM": build(LGBMRegressor(verbose=-1, random_state=SEED, **tree),
                          one_hot=False),
        "XGBoost": build(XGBRegressor(enable_categorical=True, tree_method="hist",
                                      verbosity=0, random_state=SEED, **tree),
                         one_hot=False),
        "CatBoost": build(CatBoostCategorical(cat_features=CAT_IDX,
                                              random_seed=SEED, **tree),
                          one_hot=False),
    }


def make_stack(estimators: dict):
    """堆叠集成：三个 GBDT 的输出喂给一个 Ridge 元学习器。

    文献里一致的做法（CatBoost + XGBoost + LightGBM + Ridge meta）——
    多个研究都报告堆叠比任何单一模型都好。用 `cv=make_cv()` 做内部交叉验证
    生成元特征，避免基模型在训练数据上「背答案」再喂给元学习器（那是另一种泄露）。

    ⚠️ **一处诚实的保留**：内部这层 CV 用的是普通 KFold，不是 GroupKFold。
    因为 ``StackingRegressor`` 没有把 ``groups`` 透传给内部划分器的接口
    （`fit` 不接受 groups 参数），做不到按小区分组。
    影响有多大要说清楚：元特征是在**训练折内部**生成的，外层评估用的
    验证折小区与外层训练折完全不重叠，所以**不会**把验证折的小区信息漏进去；
    受影响的只是「元学习器学到的权重略微乐观」这一点，
    量级远小于基模型层面的同小区泄露。
    要彻底修掉得自己用 GroupKFold 生成元特征再拟合 Ridge，属于可以但不值得的复杂度。
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


def shuffle_once(X, y, groups, seed: int = SEED):
    """按固定种子打乱行序，返回打乱后的 (X, y, groups)。

    **为什么必须先打乱**：``GroupKFold`` **不打乱**，它按「小区第一次出现的顺序」
    依次把小区分到各折。而原始 CSV 是按板块/环线排好序的，于是每一折恰好落在
    连续的地理区块上 —— 这样测出来的落差里，「没见过的小区」和「没见过的区域」
    两件事混在一起，说不清是谁造成的。普通 KFold 是 ``shuffle=True`` 的，
    拿一个打乱的方案去比一个没打乱的方案，差多少都不能归因。

    打乱行序只改变「哪些小区进哪一折」，**不破坏小区完整性**，
    分组约束依然成立，所以这是纯粹的实验设计修正，不是放水。
    """
    perm = np.random.RandomState(seed).permutation(len(X))
    return (X.iloc[perm].reset_index(drop=True),
            y.iloc[perm].reset_index(drop=True),
            groups.iloc[perm].reset_index(drop=True))


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
