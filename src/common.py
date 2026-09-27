# -*- coding: utf-8 -*-
"""
公共模块 —— 路径、特征工程、防泄露校验、绘图样式。

所有脚本都要用到同一套特征工程和同一条防泄露规则。
这些东西只在这里定义一次：尤其是「哪些列绝对不能进模型」，
如果有两份定义，早晚会有一份先过期。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
for _d in (DATA, FIG, RES):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA / "lianjia_bj_raw.csv"        # 01 产出：全量 318,851 行
CLEAN_CSV = DATA / "lianjia_bj_clean.csv"    # 02 产出：2017 年建模表

SEED = 42
TARGET = "单价"

# --------------------------------------------------------------------------
# 市场窗口
# --------------------------------------------------------------------------
# 原始数据跨 2002–2018，这期间北京单价中位数从 3.8 万涨到 6.6 万（1.75 倍）。
# 全量混在一起建模，模型学到的很大一部分会是「这套房是哪年卖的」，
# 而不是「这套房值多少」—— 那不是一个估价模型。
#
# 所以只取一个市场水位基本一致的窗口。2017 全年 43,217 条：
#   - 是作业要求的样本量下限（10,000 条）的 4.3 倍
#   - 单一日历年，年内单价中位数波动 1.17 倍（56,961 ~ 66,449），
#     比 2016–2017 的 1.75 倍干净得多
#   - 因此不需要把「成交年月」放进特征，模型纯粹由房屋属性解释价格
#
# 想换窗口改这一个常量即可：(2016, 2017) → 134,046 行；None → 全量 318,851 行。
WINDOW_YEARS = (2017, 2017)

# --------------------------------------------------------------------------
# 编码 → 中文。这些映射**不是从文档抄的**，是从数据本身实证出来的，
# 证据一并写在下面（README 里也有）。原始数据没有附带码表。
# --------------------------------------------------------------------------
# 证据：按 district 分组算平均经纬度，与北京各区真实地理位置、真实房价排序对照；
# 再用 10 个公开地标反查（金融街/国贸/中关村/天通苑/果园/亦庄/古城/良乡/顺义/门头沟），
# 每个地标 2km 内最近的 60 套房源 60/60 全部落在预期编码上。
DISTRICT_NAMES = {
    1: "东城", 2: "丰台", 3: "亦庄开发区", 4: "大兴", 5: "房山", 6: "昌平",
    7: "朝阳", 8: "海淀", 9: "石景山", 10: "西城", 11: "通州", 12: "门头沟", 13: "顺义",
}

# 证据：code 1 总层数中位 21、98.0% 有电梯（塔楼）；code 4 总层数中位 6、
# 26.6% 有电梯（多层板楼）；code 3 中位 17 层、94.6% 有电梯（板塔结合）；
# code 2 只有 1 条、总层数 1（平房）。层数和电梯率各自独立地指向同一套标签。
# 注意：网上流传的「1=板楼、4=平房」与数据矛盾（4 有 6 层、27% 装电梯），是错的。
BUILDING_TYPE_NAMES = {1: "塔楼", 2: "平房", 3: "板塔结合", 4: "板楼"}

# 这一组证据弱得多，只在 README 里标明，模型里按原样保留代码语义：
# code 4 房龄最轻（13 年）、电梯率最高（65.3%）；code 2 单价最低（57,824）。
# 装修对单价的影响本身很小（四档中位数 57,824 ~ 63,326，约 9%）。
RENOVATION_NAMES = {1: "毛坯", 2: "简装", 3: "精装", 4: "豪华"}

FLOOR_LEVELS = ["底", "低", "中", "高", "顶", "未知"]

# 稀有类别并成「其他」的样本量下限。
#
# 为什么必须做这件事：one-hot + 线性模型会给每个类别一个自由系数，
# 而系数是拿该类别的样本估出来的。实测原始取值里，「建筑类型=平房」只有 1 行，
# 却拿到了 +11,363 元/㎡ 的系数 —— 页面上谁勾了「平房」，这个数字就凭空冒出来。
# 这不是模型学到的知识，是拿一行样本当成了规律。
#
# 怎么定的 200：单看均值的标准误约 std/√n。本数据单价标准差约 24,000，
# n=200 时标准误 ≈ 1,700 元/㎡，已经和要估的效应同量级；再小就没有意义了。
# 这个阈值刚好把 平房(1)、结构1(30)、结构3(13)、结构5(37) 四档并掉，
# 而 13 个区县里最少的门头沟有 267 行，全部保留 —— 区县是用户必须能选的维度，
# 不能因为样本少就并成「其他」，那样页面上就没有门头沟这个选项了。
MIN_CATEGORY_COUNT = 200

OTHER = "其他"

# --------------------------------------------------------------------------
# 防泄露：绝不能进入特征列的字段
# --------------------------------------------------------------------------
DROP_ALWAYS = [
    "单价",             # 目标本身
    "price",            # 目标在原始数据里的列名
    "totalPrice",       # 总价（万元）= 单价 × 面积 / 10000，与目标互为函数关系
    "communityAverage", # 小区均价。它是同一批数据里同小区房源的均价，天然含本行信息。
                        # 名字里没有「价格」二字，只看列名拦不住 —— 靠下面的 R² 筛查抓
    "url", "id", "Cid", # 标识符，无信息
    "tradeTime",        # 只用来切窗口，不留作特征（见 WINDOW_YEARS）
    "DOM",              # 挂牌天数：成交后才知道，而且「卖得慢」本身可能是价格太高的结果
    "Lng", "Lat",       # 经纬度：是「区县」的更细粒度版本。放进线性模型会得到一张
                        # 光滑的价格平面，区县系数变得不可读，而本练习要看懂系数
    "kitchen",          # 42,829/43,217 都是 1，零方差
]

# 列名层面兜底：任何名字像价格的列都拦下来
LEAK_NAME_PATTERN = re.compile(r"价格|总价|单价|首付|万元|_万$|price|Price")

# 单特征交叉验证 R² 超过这个值就判定为泄露。
#
# 为什么设一道 R² 关，而不是只看相关系数：实测「小区均价」与目标的相关系数
# 0.932、「总价_万」只有 +0.473，用 0.95 的相关性阈值**一个都拦不住**；
# 但让它们单独预测单价，CV R² 分别是 0.868 和 0.2234（见 results/leak_screen.csv），
# 前者一测就露馅。而所有合法的房屋属性特征 R² 都在 +0.11 以内（最强的「近地铁」
# 0.1056），所以这条线对**这一份数据**分得很干净。
#
# ⚠️ 但这条线不是保证，别把它当成安全网：
#   1. 「总价_万」在 0.2234，**远在阈值之下**，R² 这一关根本拦不住它 ——
#      它是靠列名/语义那一层（LEAK_NAME_PATTERN 里的 `price`）拦下来的。
#   2. 同一个「总价_万」，换到第二版数据上 R² 是 0.3522，刚刚越过 0.35 ——
#      同一套阈值、同一个泄露列，换份数据结论就**反过来**。
#      判别力本身依赖数据，不能当成保证。
#   3. 数值判据看不见比值型泄露，根因是**线性模型自己造不出「比值」这种
#      非线性组合**：判据的有效性依赖于测试它用的模型类别。
#
# 所以三层是递进关系，优先级从高到低：**列名/语义 → DROP_ALWAYS 清单 → R² 筛查**。
# 前两层是能依赖的，第三层是兜底。
LEAK_R2_THRESHOLD = 0.35

# --------------------------------------------------------------------------
# 绘图样式（配色取自 dataviz 技能里已通过校验器的调色板）
# --------------------------------------------------------------------------
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_YELLOW, C_MAGENTA = "#eda100", "#e87ba4"
C_CRITICAL = "#d03b3b"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, MIDPOINT = "#e1e0d9", "#c3c2b7", "#f0efec"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# 分类色序（固定顺序，不循环使用）
CAT_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#7d5ec7", "#0f9bbd", "#c96a2b"]


def setup_chinese_font() -> None:
    """让 matplotlib 正常显示中文和负号。所有画图脚本的第一句。"""
    import matplotlib
    matplotlib.use("Agg")  # 只存文件，不弹窗
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
    # SimHei 里没有 U+2212（排版减号）。matplotlib 默认用它，于是所有负数刻度
    # 都会显示成空白方块 —— 而本练习的 R² 经常是负的，正好全中。必须关掉。
    plt.rcParams["axes.unicode_minus"] = False
    # 注意：SimHei 没有粗体，不要用 fontweight="bold"，否则每个文本都触发 findfont 警告。
    # 注意：SimHei 也没有上标 ²（U+00B2），图里一律写 R^2 / eta^2，不要写 R²。
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.dpi": 200, "savefig.bbox": "tight", "figure.dpi": 120,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": 11, "axes.titlesize": 14,
    })


def seq_cmap():
    """顺序色阶（单一蓝色，浅→深），用于量级编码。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def div_cmap():
    """发散色阶（蓝↔红，灰中点对齐 0），用于相关矩阵这种有极性的量。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "div_br", ["#0d366b", "#2a78d6", "#9ec5f4", MIDPOINT, "#f0a3a2", "#e34948", "#a01f1f"])


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """去掉所有字符串列的首尾空白。

    坑：pandas 3 起字符串列的 dtype 是 'str'，**不再等于 object**。
    写成 `if df[c].dtype == object` 会静默跳过所有字符串列，strip 完全失效。
    """
    for c in df.columns:
        if pdt.is_string_dtype(df[c]) or pdt.is_object_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
    return df


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise SystemExit(f"找不到 {RAW_CSV}，请先运行：uv run python src/01_download.py")
    return _strip_strings(pd.read_csv(RAW_CSV, encoding="utf-8", low_memory=False))


def load_clean() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise SystemExit(f"找不到 {CLEAN_CSV}，请先运行：uv run python src/02_clean.py")
    return pd.read_csv(CLEAN_CSV, encoding="utf-8")


# --------------------------------------------------------------------------
# 特征工程
# --------------------------------------------------------------------------
NUMERIC = ["面积", "室", "厅", "卫", "总层数", "房龄", "梯户比",
           "关注人数", "电梯", "满五", "近地铁"]
CATEGORICAL = ["区县", "楼层位置", "装修", "建筑类型", "建筑结构"]

# 清洗阈值。2017 年单价中位数 62,331，5,000 ~ 200,000 是个很宽的带，
# 只用来剔除明显是录入错误的记录（原始数据里最小 136 元/㎡）。
PRICE_MIN, PRICE_MAX = 5_000, 200_000
AREA_MIN, AREA_MAX = 10.0, 1_000.0
YEAR_MIN, YEAR_MAX = 1949, 2017
LADDER_MAX = 10.0        # 梯户比 >10 的视为录入错误（实测最大 10,009,400）


def _num(s) -> pd.Series:
    """能转数字的转数字，转不了（'#NAME?' 这类 Excel 损坏值）变 NaN。"""
    return pd.to_numeric(s, errors="coerce")


def build_features(raw: pd.DataFrame, years: tuple[int, int] | None = WINDOW_YEARS):
    """从原始 26 列构造建模表。默认只取 WINDOW_YEARS 指定的年份窗口。

    返回 (特征表, 被剔除的泄露列演示表)。第二项只用于展示泄露有多危险。
    """
    d = raw.copy()
    d["_year"] = d["tradeTime"].astype(str).str[:4].astype(float)

    out = pd.DataFrame(index=d.index)

    # ---- 目标与泄露候选 ----
    out["单价"] = _num(d["price"])
    out["总价_万"] = _num(d["totalPrice"])            # 只用于泄露演示
    out["小区均价"] = _num(d["communityAverage"])      # 只用于泄露演示

    # ---- 房屋属性 ----
    out["面积"] = _num(d["square"])
    out["室"] = _num(d["livingRoom"])
    out["厅"] = _num(d["drawingRoom"])
    out["卫"] = _num(d["bathRoom"])

    # floor 形如「中 18」「高 26」「顶 6」：前缀是楼层位置，末尾数字是楼栋总层数。
    # 前缀「未知」的 4 条当缺失处理，交给 SimpleImputer。
    floor = d["floor"].astype(str)
    out["总层数"] = _num(floor.str.extract(r"(\d+)$")[0])
    level = floor.str.extract(r"^(\D+?)\s*\d+$")[0]
    out["楼层位置"] = level.where(level.isin(FLOOR_LEVELS[:-1]),
                                 pd.Series(np.nan, index=d.index))

    out["房龄"] = d["_year"] - _num(d["constructionTime"])
    # 原始数据里有 2018 年之后建成的记录（录入错误），房龄成了负数
    out.loc[~out["房龄"].between(0, d["_year"].max() - YEAR_MIN), "房龄"] = np.nan

    ladder = _num(d["ladderRatio"])
    out["梯户比"] = ladder.where(ladder.between(0, LADDER_MAX))

    out["关注人数"] = _num(d["followers"])
    out["电梯"] = _num(d["elevator"])
    out["满五"] = _num(d["fiveYearsProperty"])
    out["近地铁"] = _num(d["subway"])

    # ---- 类别 ----
    out["区县"] = _num(d["district"]).map(DISTRICT_NAMES)
    out["装修"] = _num(d["renovationCondition"]).map(RENOVATION_NAMES)
    out["建筑类型"] = _num(d["buildingType"]).map(BUILDING_TYPE_NAMES)
    out["建筑结构"] = _num(d["buildingStructure"]).map(lambda v: f"结构{int(v)}" if pd.notna(v) else np.nan)

    out["_year"] = d["_year"]

    # ---- 切窗口 + 清洗 ----
    n0 = len(out)
    if years is not None:
        out = out[out["_year"].between(*years)]
    n1 = len(out)

    keep = (out["单价"].between(PRICE_MIN, PRICE_MAX)
            & out["面积"].between(AREA_MIN, AREA_MAX)
            & out["总价_万"].notna() & (out["总价_万"] > 0))
    dropped = out[~keep].copy()
    out = out[keep].copy()
    n2 = len(out)

    # ---- 稀有类别：先并成「其他」，并完还不够就按缺失处理 ----
    #
    # 放在切窗口和清洗**之后**：样本量要按真正进入建模的数据算，
    # 否则会被窗口外那些不参与建模的行撑大。
    #
    # 两步走是有必要的，只做第一步会留下同样的坑：本数据「建筑类型」只有
    # 平房(1 行) 一个稀有取值，并成「其他」之后这个桶本身还是 1 行，
    # 照样拿到 +10,460 元/㎡ 的系数 —— 换个名字并没有解决问题。
    # 所以还要检查桶够不够大；不够大就按缺失处理，交给 Pipeline 里的
    # SimpleImputer(strategy="most_frequent") 归进最常见的那一档。
    merged: dict[str, dict] = {}
    to_missing: dict[str, dict] = {}
    for c in CATEGORICAL:
        vc = out[c].value_counts()
        rare = [k for k, n in vc.items() if n < MIN_CATEGORY_COUNT]
        if not rare:
            continue
        merged[c] = {k: int(vc[k]) for k in rare}
        n_rare = int(sum(vc[k] for k in rare))
        if n_rare < MIN_CATEGORY_COUNT:
            out[c] = out[c].where(~out[c].isin(rare), np.nan)
            to_missing[c] = merged.pop(c)
        else:
            out[c] = out[c].where(~out[c].isin(rare), OTHER)

    stats = {"原始": n0, "窗口内": n1, "清洗后": n2, "窗口": years, "剔除": len(dropped),
             "稀有类别": merged, "样本不足": to_missing}
    return out.drop(columns=["_year"]), dropped, stats


def feature_columns() -> list[str]:
    return NUMERIC + CATEGORICAL


# --------------------------------------------------------------------------
# 防泄露校验
# --------------------------------------------------------------------------
def screen_single_features(X: pd.DataFrame, y: pd.Series, thresh: float = LEAK_R2_THRESHOLD):
    """逐个特征单独做交叉验证，R² 过高的判为泄露。

    单看相关系数是不够的：「小区均价」与目标的相关系数 0.932、
    「总价_万」只有 +0.473，用 0.95 的相关性阈值一个都拦不住。
    但让它们单独预测单价，CV R² 分别是 0.868 和 0.2234 —— 一测就露馅。

    **但它只拦得住前者。**「总价_万」的 0.2234 远在阈值 0.35 之下，
    这道关放它过去了，真正拦住它的是列名/语义那一层（正则里的 `price`）。
    原因见 LEAK_R2_THRESHOLD 上方那段注释：线性模型造不出「总价 ÷ 面积」
    这种比值，所以看不见比值型泄露。调用方不能只靠这一道关。

    只筛数值列：类别列要做成 one-hot 才能测，而 one-hot 本身维度很高，
    单独测出来的 R² 受维度影响，不适合用同一个阈值判断。
    """
    cv = make_cv()
    rows = []
    for c in X.columns:
        if not pdt.is_numeric_dtype(X[c]):
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("model", Ridge(alpha=1.0))])
        r2 = cross_val_score(pipe, X[[c]], y, cv=cv, scoring="r2", error_score="raise").mean()
        rows.append({"特征": c, "单特征CV_R2": round(float(r2), 4),
                     "判定": "泄露" if r2 > thresh else "正常"})
    return pd.DataFrame(rows).sort_values("单特征CV_R2", ascending=False).reset_index(drop=True)


def guard_no_leakage(X: pd.DataFrame, y: pd.Series, screen: pd.DataFrame | None = None) -> None:
    """建模前必须调用。两道关：列名黑名单 + 单特征 R² 筛查。"""
    banned = [c for c in X.columns if LEAK_NAME_PATTERN.search(c)]
    if banned:
        raise ValueError(f"特征里出现了疑似价格的列，会与目标泄露：{banned}")

    hit = [c for c in X.columns if c in DROP_ALWAYS and c != TARGET]
    if hit:
        raise ValueError(f"特征里出现了必须排除的列：{hit}")

    flagged = screen if screen is not None else screen_single_features(X, y)
    bad = flagged[flagged["判定"] == "泄露"] if "判定" in flagged.columns else flagged
    if not bad.empty:
        raise ValueError(f"以下特征可单独预测目标，判定为泄露：\n{bad.to_string(index=False)}")


# --------------------------------------------------------------------------
# 建模
# --------------------------------------------------------------------------
def make_cv():
    """5 折交叉验证。打乱顺序，固定随机种子。

    注意：这里是普通 KFold。严格来说同一个小区可能有多套房，但它们没有
    小区标识列可用（原始数据只有经纬度），所以先用 KFold，
    README 里说明了这个乐观偏差。
    """
    return KFold(n_splits=5, shuffle=True, random_state=SEED)


def make_preprocessor(scale_numeric: bool = True, sparse_output: bool = False) -> ColumnTransformer:
    """数值列中位数填补 + 标准化；类别列众数填补 + one-hot。

    handle_unknown='ignore' 仍然必要：13 个区县在每一折里都可能有个别取值
    没出现在训练集，用默认的 'error' 会直接崩。

    坑：这里用 Pipeline([...]) 而不是 make_pipeline(...)。
    make_pipeline 在 sklearn 1.9 里**不接受 (名字, 估计器) 元组**，它会照单全收
    把整个元组当成一个「估计器」，名字取类型名，于是 step 变成
    ('tuple-1', ('imp', SimpleImputer(...))) 这种废东西，ColumnTransformer
    接着报 "All estimators should implement fit and transform"。
    要自定义 step 名字，只能用 Pipeline + 显式列表。
    """
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=sparse_output))]),
         CATEGORICAL),
    ])


class DegreeOnNumeric(BaseEstimator, TransformerMixin):
    """只对数值列做多项式展开，one-hot 列原样保留。

    教程 §6.1 的写法是 make_pipeline(PolynomialFeatures(2), Ridge())，
    多项式加在**整个**设计矩阵上。本数据有几十个 one-hot 列，做完 2 次多项式
    维度会翻好几倍，而且 one-hot 相乘没有意义（两个 0/1 相乘还是 0/1）。
    所以这里把多项式限制在数值列内。
    """
    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X, y=None):
        from sklearn.preprocessing import PolynomialFeatures
        self.pf_ = PolynomialFeatures(self.degree, include_bias=False)
        self.pf_.fit(np.asarray(X)[:, :len(NUMERIC)])
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.hstack([self.pf_.transform(X[:, :len(NUMERIC)]), X[:, len(NUMERIC):]])


def make_models() -> dict:
    """教程 §6.1 的回归模型，外加一个均值基线做对照。"""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline

    def build(model, degree: int = 1, scale: bool = True):
        steps = [("pre", make_preprocessor(scale_numeric=scale))]
        if degree > 1:
            steps.append(("poly", DegreeOnNumeric(degree)))
        steps.append(("model", model))
        return Pipeline(steps)

    return {
        "均值基线 DummyRegressor": build(DummyRegressor(strategy="mean"), scale=False),
        "线性回归 LinearRegression": build(LinearRegression()),
        "岭回归 Ridge(alpha=1)": build(Ridge(alpha=1.0)),
        "岭回归+多项式 Ridge+Poly2": build(Ridge(alpha=1.0), degree=2),
        "K近邻回归 KNN(k=5)": build(KNeighborsRegressor(n_neighbors=5)),
    }


def cv_scores(model, X, y):
    """返回 5 折的 R² 数组。

    error_score='raise'：默认值是 np.nan，pipeline 出错时会**静默**返回 nan，
    坏掉的模型看起来只是「效果差」。宁可让它直接报错。
    """
    s = cross_val_score(model, X, y, cv=make_cv(), scoring="r2", error_score="raise")
    assert not np.isnan(s).any(), "交叉验证返回了 NaN，说明有折拟合失败"
    return s


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))
