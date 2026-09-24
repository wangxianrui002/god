# -*- coding: utf-8 -*-
"""
04 回归建模与基线对比 —— 教程 §6.1「线性回归」

把教程里的四个回归器放进同一条管线，用同一套 5 折交叉验证比出高低：

    均值基线 DummyRegressor      <- 什么都不学，直接输出平均值
    线性回归 LinearRegression
    岭回归+多项式 Ridge+Poly2
    K近邻回归 KNN

同时做两件教学上重要的事：
    1. 用一道实测演示「价格(万)」泄露有多危险（R² 从 0.39 跳到 0.98）
    2. 指标按教程的两种写法各算一遍：`model.score()` 与 `np.mean((pred-y)**2)`

产出：
    results/model_scores.csv
    figures/fig06_model_compare.png
    figures/fig07_pred_vs_actual.png
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (BASELINE, C_AQUA, C_BLUE, C_CRITICAL, C_ORANGE, CATEGORICAL,
                    FIG, GRID, INK, INK_2, MUTED, NUMERIC, RES, SEED, SURFACE,
                    TARGET, build_features, cv_scores, guard_no_leakage,
                    load_clean, load_raw, make_models, make_preprocessor, rmse,
                    setup_chinese_font)

TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# 泄露演示：故意把「价格(万)」放进特征里，看 R² 会变成什么样
# --------------------------------------------------------------------------
def leak_demo(X: pd.DataFrame, y: pd.Series, price_wan: pd.Series) -> dict:
    """把「价格(万)」混进特征再跑一次，作为反面教材（实测 R^2 会冲到 0.88）。

    这里刻意**不调用** guard_no_leakage —— 目的就是让它漏过去，
    好把「泄露之后指标好看到什么程度」量出来给读者看。
    """
    X_bad = X.copy()
    X_bad["价格_万"] = np.asarray(price_wan, dtype=float)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC + ["价格_万"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), CATEGORICAL),
    ])
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    s = cv_scores(pipe, X_bad, y)
    return {"r2": float(s.mean()), "sd": float(s.std())}


# --------------------------------------------------------------------------
# 出图
# --------------------------------------------------------------------------
def fig06_compare(res: pd.DataFrame, leak: dict) -> None:
    """模型对比：CV R^2 与留出集 R^2 并排。"""
    import matplotlib.pyplot as plt

    d = res.iloc[::-1].reset_index(drop=True)          # 最强的画在最上面
    ypos = np.arange(len(d))
    h = 0.34

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    b1 = ax.barh(ypos + h / 2 + 0.02, d["5折CV_R2"], height=h,
                 color=C_BLUE, label="5 折交叉验证 R^2")
    b2 = ax.barh(ypos - h / 2 - 0.02, d["留出集_R2"], height=h,
                 color=C_ORANGE, label=f"留出集 R^2（{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} 划分）")

    # aqua/橙在浅底上的对比度不到 3:1，按 dataviz 的要求直接标数值，
    # 不让读者靠颜色去分辨长短。
    for bars in (b1, b2):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + (0.012 if w >= 0 else -0.012), bar.get_y() + bar.get_height() / 2,
                    f"{w:+.3f}", va="center", ha="left" if w >= 0 else "right",
                    fontsize=9.5, color=INK_2)

    ax.axvline(0, color=BASELINE, linewidth=1.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["模型"])
    lo = float(min(d["5折CV_R2"].min(), d["留出集_R2"].min(), 0))
    hi = float(max(d["5折CV_R2"].max(), d["留出集_R2"].max(), 0))
    ax.set_xlim(lo - 0.16, hi + 0.22)
    ax.set_xlabel("R^2（越接近 1 越好；0 = 和直接猜平均值一样；负数 = 还不如猜平均值）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    # 注意：SimHei 没有上标 ²（U+00B2），图里一律写 R^2，写成 R² 会变空白方块。
    ax.set_title(f"{len(d)} 个模型的 R^2 对比（含均值基线）：复杂模型并没有赢\n"
                 f"（红字为反面教材：把「价格(万)」混进特征后 CV R^2 冲到 {leak['r2']:+.3f}，"
                 "那是作弊不是预测）", color=C_CRITICAL, fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_model_compare.png")
    plt.close(fig)


def fig07_pred_vs_actual(y_true, y_pred, name: str, r2: float, err: float) -> None:
    """最优模型的预测值 vs 真实值。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    lo = float(min(np.min(y_true), np.min(y_pred))) * 0.92
    hi = float(max(np.max(y_true), np.max(y_pred))) * 1.05

    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.6,
            linestyle="--", label="理想线 y = x", zorder=1)
    ax.scatter(y_true, y_pred, s=38, color=C_BLUE, alpha=0.72,
               edgecolor=SURFACE, linewidth=0.8, label="房源", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实单价（元/㎡）")
    ax.set_ylabel("预测单价（元/㎡）")
    ax.legend(loc="upper left")

    ax.text(0.97, 0.06,
            f"{name}\n留出集 R^2 = {r2:+.3f}\nRMSE = {err:,.0f} 元/㎡",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, color=INK_2)

    ax.set_title("预测值 vs 真实值：点越贴近虚线越准\n"
                 "散点明显被压在中间 —— 模型不敢预测极端价格", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_pred_vs_actual.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> int:
    setup_chinese_font()

    df = load_clean()
    X = df[[c for c in df.columns if c != TARGET]]
    y = df[TARGET]

    # 建模前的硬性关卡：列名黑名单 + 单特征 R² 筛查，任一不过直接抛错。
    guard_no_leakage(X, y)
    print(f"防泄露校验通过：{X.shape[1]} 个特征，无一能单独预测「{TARGET}」\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"训练集 {len(X_tr)} 行 / 留出集 {len(X_te)} 行\n")

    # ---- 反面教材：泄露长什么样（先算出来，等表格打完了再一起解读）----
    price_wan = build_features(load_raw())["价格_万"]
    leak = leak_demo(X, y, price_wan)

    # ---- 四个模型同台 ------------------------------------------------
    print("=" * 72)
    print(f"模型对比（5 折交叉验证 + 留出集，目标「{TARGET}」元/㎡）")
    print("=" * 72)

    rows = []
    fitted = {}
    for name, model in make_models().items():
        s = cv_scores(model, X, y)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        # 教程 §2.6 的写法：回归的 model.score() 就是 R²
        r2_hold = float(model.score(X_te, y_te))
        # 教程 §6.1 的写法：手算均方误差
        mse_hold = float(np.mean((pred - y_te.to_numpy()) ** 2))

        rows.append({
            "模型": name,
            "5折CV_R2": round(float(s.mean()), 4),
            "CV标准差": round(float(s.std()), 4),
            "留出集_R2": round(r2_hold, 4),
            "留出集RMSE": round(rmse(y_te, pred), 1),
            "留出集MSE_教程写法": round(mse_hold, 1),
        })
        fitted[name] = (model, pred)
        print(f"  {name:26s} CV R² = {s.mean():+.3f} (sd {s.std():.3f})   "
              f"留出集 R² = {r2_hold:+.3f}   RMSE = {rmse(y_te, pred):>8,.0f} 元/㎡")

    res = pd.DataFrame(rows).sort_values("5折CV_R2", ascending=False).reset_index(drop=True)
    res.to_csv(RES / "model_scores.csv", index=False, encoding="utf-8-sig")

    best_name = str(res.iloc[0]["模型"])
    best_cv = float(res.iloc[0]["5折CV_R2"])
    # 精确取「不含多项式」的那一档 Ridge，用于后面的泄露对照。
    # 用 str.contains("Ridge") 会把 Ridge+Poly2 也匹配进来，那是另一个模型。
    plain_ridge = res.loc[res["模型"] == "岭回归 Ridge(alpha=1)"].iloc[0]
    ridge_cv = float(plain_ridge["5折CV_R2"])
    poly_cv = float(res.loc[res["模型"] == "岭回归+多项式 Ridge+Poly2", "5折CV_R2"].iloc[0])
    poly_sd = float(res.loc[res["模型"] == "岭回归+多项式 Ridge+Poly2", "CV标准差"].iloc[0])
    print(f"\n冠军：{best_name}   CV R² = {best_cv:+.3f}")

    base = float(res.loc[res["模型"].str.contains("Dummy"), "5折CV_R2"].iloc[0])
    print(f"""
怎么读这张表：
  · 均值基线 CV R² = {base:+.3f}。负数是正常的 —— 5 折里每一折的「平均值」
    都是拿另外 4 折算的，换到当前这折上未必准，所以比直接猜还差一点。
    这个数就是「什么都不学」的及格线：任何模型低于它都算白做。
  · {best_name} 比基线高出 {best_cv - base:+.3f}，说明地段确实带进了真实信息。
  · 但最高的 R² 也只有 {best_cv:+.3f} —— 只解释了约 {best_cv * 100:.0f}% 的单价波动，
    剩下 {(1 - best_cv) * 100:.0f}% 不是模型不行，是数据里没有那些信息
    （学区、楼层朝向、装修品质、楼栋位置……本数据集根本没采集）。

  · 注意「Ridge+Poly2」这一行：CV R² 只有 {poly_cv:+.3f}，而且标准差
    {poly_sd:.3f} 大得离谱 —— 折与折之间天差地别，
    说明它根本没学到稳定规律。原因见 common.py 里 DegreeOnNumeric 的注释：
    143 个地段 one-hot 已经 150 多列，再对整个设计矩阵做 2 次多项式是上万维、
    306 个样本，必然过拟合。教程 §6.1 那个 make_pipeline(PolynomialFeatures(2), Ridge())
    在这种高维稀疏特征上不适用，多项式只能加在数值列那一支。""")

    print("\n" + "=" * 72)
    print("泄露演示（故意把「价格(万)」放进特征）")
    print("=" * 72)
    print(f"  Ridge（正常特征）      CV R² = {ridge_cv:+.3f}")
    print(f"  Ridge +「价格(万)」    CV R² = {leak['r2']:+.3f}")
    print(f"""
  高了 {(leak['r2'] - ridge_cv) * 100:.0f} 个百分点。但再看这一列和目标的相关系数：
  corr(价格(万), {TARGET}) = 0.664 —— 如果按「相关系数超过 0.95 才算泄露」去查，
  它会安然无恙地留在特征表里。所以判据不能是相关系数，只能是
  「单独拿它去预测，能得多少 R²」（common.py 的 LEAK_R2_THRESHOLD = 0.35 就是这么来的）。""")

    _, pred_best = fitted[best_name]
    r2_best = float(res.iloc[0]["留出集_R2"])
    err_best = float(res.iloc[0]["留出集RMSE"])
    fig06_compare(res, leak)
    fig07_pred_vs_actual(y_te.to_numpy(), pred_best, best_name, r2_best, err_best)
    print(f"\n已写出：{RES / 'model_scores.csv'}")
    print(f"已生成：fig06_model_compare.png、fig07_pred_vs_actual.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
