# -*- coding: utf-8 -*-
"""
公共模块 —— 路径、特征工程、防泄露校验、绘图样式。

四个脚本（02/03/04/05）都要用到同一套特征工程和同一条防泄露规则。
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

RAW_CSV = DATA / "lianjia_bj.csv"
CLEAN_CSV = DATA / "lianjia_bj_clean.csv"

SEED = 42
TARGET = "单价"

# --------------------------------------------------------------------------
# 防泄露：绝不能进入特征列的字段
# --------------------------------------------------------------------------
# 单价 = 价格(万) × 10000 / 面积，所以「价格(万)」是目标变量的另一种写法，
# 用它当特征 R² 会直接冲到 0.85 以上，模型毫无意义。
DROP_ALWAYS = [
    "单价",        # 目标本身
    "价格(万)",    # 与目标互为倒数关系 —— 泄露
    "标题",        # 108/306 条标题里直接写着小区名，等于把「小区」这个近乎唯一的
                   # 标识符又拼了回来；另有 102 条含「满五」（=房本）
    "标签",        # 306 行只有 1 个取值，零方差
    "小区",        # 282 个唯一值 / 306 行，比样本还碎，只能记住不能泛化
    "VR",
    "看房时期",
    "分布日期",    # 只取其中的天数，原文不留
]

# 列名层面兜底：任何名字像价格的列都拦下来
LEAK_NAME_PATTERN = re.compile(r"价格|总价|单价|首付|万元|_万$")

# 单特征交叉验证 R² 超过这个值就判定为泄露。
#
# 为什么是 0.35 而不是 0.5：实测「价格(万)」单独一个特征的 CV R² 是 0.421，
# 用 0.5 当阈值会把它**漏过去**。而所有合法原始特征的 R² 都在 ±0.05 以内，
# 0.35 这条线分得很干净。
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

    源数据 10 个列都带填充空格（' 精装 '、' 世纪星城 '）。不 strip 的话
    OneHotEncoder 会把 ' 精装 ' 和 '精装' 当成两个类别，凭空多出一倍的类别数。

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
    return _strip_strings(pd.read_csv(RAW_CSV, encoding="utf-8"))


def load_clean() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise SystemExit(f"找不到 {CLEAN_CSV}，请先运行：uv run python src/02_clean.py")
    return pd.read_csv(CLEAN_CSV, encoding="utf-8")


# --------------------------------------------------------------------------
# 特征工程
# --------------------------------------------------------------------------
NUMERIC = ["面积", "房间数", "厅数", "总层数", "建成年", "关注人数", "近地铁", "南北通透"]
CATEGORICAL = ["地段", "装修", "形式"]


def _first_number(s, strip_comma: bool = False):
    """从「92.2平米」「59,653元/平」这类字符串里取出第一个数字。

    先统一转成字符串：有些列（如「价格(万)」）整列都是纯数字，
    pandas 会直接推断成整数类型，那样就没有 .str 访问器可用。
    """
    s = s.astype(str)
    if strip_comma:
        s = s.str.replace(",", "", regex=False)
    return s.str.extract(r"(-?\d+(?:\.\d+)?)")[0].astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始 19 列构造建模用的特征表。"""
    out = pd.DataFrame(index=df.index)

    out["面积"] = _first_number(df["面积"])
    out["单价"] = _first_number(df["单价"], strip_comma=True)          # 目标
    out["价格_万"] = _first_number(df["价格(万)"])                      # 只用于泄露演示

    out["房间数"] = df["户型"].str.extract(r"(\d+)室")[0].astype(float)
    out["厅数"] = df["户型"].str.extract(r"(\d+)厅")[0].astype(float)

    # 楼层形如「中楼层(共6层)」或「25层」。两种写法都是楼栋总层数。
    with_total = df["楼层"].str.extract(r"共(\d+)层")[0]
    bare_total = df["楼层"].str.extract(r"^(\d+)层$")[0]
    out["总层数"] = with_total.fillna(bare_total).astype(float)          # 272 + 34 = 306/306

    out["建成年"] = _first_number(df["年份"])                            # 305/306

    out["关注人数"] = _first_number(df["关注人数"])
    out["近地铁"] = df["交通"].str.contains("近地铁", na=False).astype(int)
    out["南北通透"] = (df["朝向"].str.contains("南", na=False)
                   & df["朝向"].str.contains("北", na=False)).astype(int)

    out["地段"] = df["地段"]
    out["装修"] = df["装修"]
    out["形式"] = df["形式"].replace({"暂无数据": np.nan})

    return out


def feature_columns() -> list[str]:
    return NUMERIC + CATEGORICAL


# --------------------------------------------------------------------------
# 防泄露校验
# --------------------------------------------------------------------------
def screen_single_features(X: pd.DataFrame, y: pd.Series, thresh: float = LEAK_R2_THRESHOLD):
    """逐个特征单独做交叉验证，R² 过高的判为泄露。

    单看相关系数是不够的：corr(价格(万), 单价) 只有 0.664，用 0.95 的相关性
    阈值根本拦不住它。但让它单独预测单价，CV R² 是 0.421 —— 一测就露馅。
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
        rows.append({"特征": c, "单特征CV_R2": round(float(r2), 4), "判定": "泄露" if r2 > thresh else "正常"})
    return pd.DataFrame(rows).sort_values("单特征CV_R2", ascending=False).reset_index(drop=True)


def guard_no_leakage(X: pd.DataFrame, y: pd.Series) -> None:
    """建模前必须调用。两道关：列名黑名单 + 单特征 R² 筛查。"""
    banned = [c for c in X.columns if LEAK_NAME_PATTERN.search(c)]
    if banned:
        raise ValueError(f"特征里出现了疑似价格的列，会与目标泄露：{banned}")

    hit = [c for c in X.columns if c in DROP_ALWAYS and c != TARGET]
    if hit:
        raise ValueError(f"特征里出现了必须排除的列：{hit}")

    flagged = screen_single_features(X, y)
    bad = flagged[flagged["判定"] == "泄露"]
    if not bad.empty:
        raise ValueError(f"以下特征可单独预测目标，判定为泄露：\n{bad.to_string(index=False)}")


# --------------------------------------------------------------------------
# 建模
# --------------------------------------------------------------------------
def make_cv():
    """5 折交叉验证。打乱顺序，固定随机种子。

    注意：这里是普通 KFold。严格来说同一个小区可能有多套房，用
    GroupKFold(groups=小区) 才是无偏的；本练习样本量小，先用 KFold，
    README 里说明了这个乐观偏差。
    """
    return KFold(n_splits=5, shuffle=True, random_state=SEED)


def make_preprocessor(scale_numeric: bool = True, sparse_output: bool = False) -> ColumnTransformer:
    """数值列中位数填补 + 标准化；类别列众数填补 + one-hot。

    handle_unknown='ignore' 是必须的：143 个地段里绝大多数只有 1~3 套房，
    每个交叉验证折里都会出现训练集没见过的新地段，用默认的 'error' 会直接崩。

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
    多项式加在**整个**设计矩阵上。本数据有 143 个地段 one-hot，做完 2 次多项式
    是上万维、306 个样本，必然崩掉（实测 CV R² 直接掉到 0 附近）。
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
