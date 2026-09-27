# -*- coding: utf-8 -*-
"""
04 回归建模与基线对比 —— 教程 §6.1「线性回归」

把教程里的四个回归器放进同一条管线，用同一套 5 折交叉验证比出高低：

    均值基线 DummyRegressor      <- 什么都不学，直接输出平均值
    线性回归 LinearRegression
    岭回归+多项式 Ridge+Poly2
    K近邻回归 KNN

同时做两件教学上重要的事：
    1. 用一道实测演示「总价_万」泄露有多危险 —— 而且它是数值判据拦不住的那一个
    2. 指标按教程的两种写法各算一遍：`model.score()` 与 `np.mean((pred-y)**2)`

产出：
    results/model_scores.csv
    figures/fig07_model_compare.png
    figures/fig08_pred_vs_actual.png
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

from common import (BASELINE, C_BLUE, C_CRITICAL, C_ORANGE, CATEGORICAL,
                    FIG, INK_2, NUMERIC, RES, SEED, SURFACE,
                    TARGET, build_features, cv_scores, guard_no_leakage,
                    load_clean, load_raw, make_models, rmse,
                    setup_chinese_font)

TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# 泄露演示：故意把「价格(万)」放进特征里，看 R² 会变成什么样
# --------------------------------------------------------------------------
def leak_demo(X: pd.DataFrame, y: pd.Series, total_wan: pd.Series) -> dict:
    """把「总价_万」混进特征再跑一次，作为反面教材。

    这一列是**数值判据拦不住**的那一个：它的相关系数只有 0.47、单特征 CV R²
    只有 0.223，两道数值判据都会放它过去（见 03 的 fig05）。但它其实是目标的
    恒等重建 —— 单价 = 总价×10000÷面积，而线性模型自己造不出「比值」这个
    非线性组合，所以单看它一个，R² 才显得那么低。

    这里刻意**不调用** guard_no_leakage —— 目的就是让它漏过去，
    好把「泄露之后指标好看到什么程度」量出来给读者看。
    """
    X_bad = X.copy()
    X_bad["总价_万"] = np.asarray(total_wan, dtype=float)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC + ["总价_万"]),
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
def fig07_compare(res: pd.DataFrame, leak: dict) -> None:
    """模型对比：CV R^2 与留出集 R^2 并排。"""
    import matplotlib.pyplot as plt

    # 泄露那一行单独挂在最上面，用红色，且**只有 CV 一根柱子** ——
    # 泄露演示只算了交叉验证，硬凑一根留出集柱子等于编数字。
    # 它不进这张表的排序，是反面参照物，所以中间多留半行空白和真实模型隔开。
    leak_row = 0.0
    d = res.iloc[::-1].reset_index(drop=True)          # 最强的画在最上面
    ypos = np.arange(len(d)) + 1.15
    h = 0.34

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    b1 = ax.barh(ypos + h / 2 + 0.02, d["5折CV_R2"], height=h,
                 color=C_BLUE, label="5 折交叉验证 R^2")
    b2 = ax.barh(ypos - h / 2 - 0.02, d["留出集_R2"], height=h,
                 color=C_ORANGE, label=f"留出集 R^2（{int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} 划分）")

    ax.barh([leak_row], [leak["r2"]], height=h, color=C_CRITICAL,
            label="泄露演示：Ridge +「总价_万」（同一套 CV，故意漏过去）")
    ax.text(leak["r2"] + 0.012, leak_row, f"{leak['r2']:+.3f}",
            va="center", ha="left", fontsize=9.5, color=C_CRITICAL)
    ax.text(0.012, leak_row - 0.42,
            "这才是「把答案抄进去」的样子 —— 判据拦不住它，只有列名/语义能拦",
            va="center", ha="left", fontsize=9.5, color=C_CRITICAL)

    # aqua/橙在浅底上的对比度不到 3:1，按 dataviz 的要求直接标数值，
    # 不让读者靠颜色去分辨长短。
    for bars in (b1, b2):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + (0.012 if w >= 0 else -0.012), bar.get_y() + bar.get_height() / 2,
                    f"{w:+.3f}", va="center", ha="left" if w >= 0 else "right",
                    fontsize=9.5, color=INK_2)

    ax.axvline(0, color=BASELINE, linewidth=1.4)
    ax.set_yticks(list(ypos) + [leak_row])
    ax.set_yticklabels(list(d["模型"]) + ["Ridge ＋「总价_万」\n（故意泄露，仅 CV）"],
                       color=INK_2)
    # 把泄露那一行的标签涂红，和柱子的红对上
    ax.get_yticklabels()[-1].set_color(C_CRITICAL)
    lo = float(min(d["5折CV_R2"].min(), d["留出集_R2"].min(), 0))
    hi = float(max(d["5折CV_R2"].max(), d["留出集_R2"].max(), leak["r2"], 0))
    ax.set_xlim(lo - 0.16, hi + 0.22)
    ax.set_ylim(-0.9, len(d) + 1.6)
    ax.set_xlabel("R^2（越接近 1 越好；0 = 和直接猜平均值一样；负数 = 还不如猜平均值）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    # 注意：SimHei 没有上标 ²（U+00B2），图里一律写 R^2，写成 R² 会变空白方块。
    ax.set_title(f"{len(d)} 个模型的 R^2 对比（含均值基线）\n"
                 f"最上面那根红柱是反面教材：把「总价_万」混进特征，CV R^2 冲到 "
                 f"{leak['r2']:+.3f} —— 那是作弊不是预测",
                 color=C_CRITICAL, fontsize=12.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig07_model_compare.png")
    plt.close(fig)


def fig08_pred_vs_actual(y_true, y_pred, name: str, r2: float, err: float) -> None:
    """最优模型的预测值 vs 真实值。

    留出集有八千多个点，用 s=38 会糊成一团黑，所以点调小、透明度调低。
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true.min(), y_pred.min())) * 0.92
    hi = float(max(y_true.max(), y_pred.max())) * 1.05

    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.6,
            linestyle="--", label="理想线 y = x", zorder=1)
    ax.scatter(y_true, y_pred, s=7, color=C_BLUE, alpha=0.16,
               edgecolor="none", label=f"房源（{len(y_true):,} 套）", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实单价（元/㎡）")
    ax.set_ylabel("预测单价（元/㎡）")
    ax.legend(loc="upper left")

    ax.text(0.97, 0.06,
            f"{name}\n留出集 R^2 = {r2:+.3f}\nRMSE = {err:,.0f} 元/㎡",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, color=INK_2,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=3))

    ax.set_title("预测值 vs 真实值：点越贴近虚线越准\n"
                 "散点被压在中间 —— 模型不敢预测极端价格", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_pred_vs_actual.png")
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
    feats, _, _ = build_features(load_raw())
    feats = feats.dropna(subset=[TARGET]).reset_index(drop=True)
    assert len(feats) == len(df), f"行数对不上：feats {len(feats)} vs clean {len(df)}"
    leak = leak_demo(X, y, feats["总价_万"])

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
    knn_cv = float(res.loc[res["模型"].str.startswith("K近邻"), "5折CV_R2"].iloc[0])
    print(f"""
怎么读这张表：
  · 均值基线 CV R² = {base:+.3f}。负数是正常的 —— 5 折里每一折的「平均值」
    都是拿另外 4 折算的，换到当前这折上未必准，所以比直接猜还差一点。
    这个数就是「什么都不学」的及格线：任何模型低于它都算白做。
  · 冠军是 {best_name}，比基线高出 {best_cv - base:+.3f} —— 区县确实带进了真实信息
    （03 里量过：区县对单价的方差解释率 eta^2 = 0.627）。
  · 但最高的 R² 也只有 {best_cv:+.3f} —— 只解释了约 {best_cv * 100:.0f}% 的单价波动，
    剩下 {(1 - best_cv) * 100:.0f}% 不是模型不行，是数据里没有那些信息
    （学区、楼栋朝向、小区品质、临街与否……本数据集根本没采集）。

  · KNN 的 CV R² = {knn_cv:+.3f}，比线性模型低 {(best_cv - knn_cv) * 100:.0f} 个百分点。
    原因见 05 的验证曲线那一节：类别列 one-hot 之后空间维度高、样本分布稀疏，
    「距离」在这种空间里没有意义 —— 这是维度灾难，调 k 治不好。

  · 注意「Ridge+Poly2」这一行：CV R² {poly_cv:+.3f}（标准差 {poly_sd:.3f}），
    比不加多项式的 Ridge 高 {(poly_cv - ridge_cv) * 100:+.1f} 个百分点 —— 它赢了，
    虽然赢得很小。**这个结论和样本量强相关**：同一段代码在只有 306 行时
    CV R² 是 +0.005（等于没学到东西），样本量涨到 {len(df):,} 行之后才翻过来。
    数值列只有 {len(NUMERIC)} 个，平方和两两乘积把数值那一支从 {len(NUMERIC)} 维
    撑到 {len(NUMERIC) * (len(NUMERIC) + 3) // 2} 维，多出来的维度要靠样本量喂饱。

    所以「多项式会不会炸」不是算法的固有属性，而是
    「模型复杂度 ÷ 样本量」的函数 —— 这正是教程 §2.4 讲的过拟合。
    另外这里仍然把多项式限制在数值列内（见 common.py 的 DegreeOnNumeric）：
    教程 §6.1 的 make_pipeline(PolynomialFeatures(2), Ridge()) 是把多项式加在
    **整个**设计矩阵上，连 one-hot 列也会被乘起来，而两个 0/1 相乘还是 0/1，
    那些乘积项纯粹是白占维度。""")

    print("\n" + "=" * 72)
    print("泄露演示（故意把「总价_万」放进特征）")
    print("=" * 72)
    leak_corr = float(feats["总价_万"].corr(y))
    print(f"  Ridge（正常特征）      CV R² = {ridge_cv:+.3f}")
    print(f"  Ridge +「总价_万」     CV R² = {leak['r2']:+.3f}")
    print(f"""
  高了 {(leak['r2'] - ridge_cv) * 100:.0f} 个百分点。而这一列在两道数值判据下的样子是：
  corr(总价_万, {TARGET}) = {leak_corr:+.3f}（离 0.95 的常规阈值远得很），
  单特征 CV R² = +0.223（离 0.35 的判据线也远得很）—— 两道都拦不住它。

  可它是实打实的泄露。「总价 × 10000 ÷ 面积」和单价的相关系数是
  0.9999999998，本来就是同一个数的两种写法。数值判据看不见它，是因为
  判据用的线性模型**自己造不出「比值」**这个非线性组合；换个能学比值的模型，
  这一列的 R² 立刻就是 1.0。所以判据的有效性依赖于测试它用的模型类别 ——
  这就是为什么 common.py 里既有列名黑名单（LEAK_NAME_PATTERN，按语义拦），
  又有单特征 R² 筛查（按统计拦），两层缺一不可：
  「总价_万」只有语义层拦得住，「小区均价」只有统计层拦得住。
  完整对照见 03 的 fig05。""")

    _, pred_best = fitted[best_name]
    r2_best = float(res.iloc[0]["留出集_R2"])
    err_best = float(res.iloc[0]["留出集RMSE"])
    fig07_compare(res, leak)
    fig08_pred_vs_actual(y_te.to_numpy(), pred_best, best_name, r2_best, err_best)
    print(f"\n已写出：{RES / 'model_scores.csv'}")
    print("已生成：fig07_model_compare.png、fig08_pred_vs_actual.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
