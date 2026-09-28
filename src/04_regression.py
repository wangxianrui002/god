"""比较三种回归模型，保存量化结果、图表和静态展示页。"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from common import (
    FEATURES, FIGURES, RESULTS, SEED, TARGET, WEB,
    load_clean, make_models, setup_plot,
)


def make_figures(scores: pd.DataFrame, y_test: pd.Series, prediction: np.ndarray,
                 winner: str) -> None:
    import matplotlib.pyplot as plt

    setup_plot()
    FIGURES.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(scores))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - 0.18, scores["交叉验证R2"], width=0.36, label="训练集 5 折 CV", color="#2a78d6")
    ax.bar(x + 0.18, scores["测试集R2"], width=0.36, label="独立测试集", color="#ed6a36")
    ax.set_xticks(x, scores["模型"])
    ax.set_ylim(-0.08, max(0.9, float(scores["测试集R2"].max()) + 0.08))
    ax.set(ylabel="R²", title="三种回归模型与均值基线")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_model_scores.png")
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    selected = rng.choice(len(y_test), size=min(2_500, len(y_test)), replace=False)
    actual = y_test.to_numpy()[selected]
    predicted = prediction[selected]
    fig, ax = plt.subplots(figsize=(6.2, 5.7))
    ax.scatter(actual, predicted, s=8, alpha=0.23, color="#2a78d6", edgecolors="none")
    upper = max(float(actual.max()), float(predicted.max()))
    ax.plot([0, upper], [0, upper], color="#ed6a36", linewidth=1.5, label="理想预测")
    ax.set(xlabel="真实单价（元/平米）", ylabel="预测单价（元/平米）",
           title=f"测试集预测效果：{winner}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_predictions.png")
    plt.close(fig)


def make_page(scores: pd.DataFrame, n_rows: int, n_train: int, n_test: int,
              winner: pd.Series) -> None:
    """单页静态汇报；数字直接取本次运行的模型结果。"""
    WEB.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr><th scope='row'>{}</th><td>{:+.3f}</td><td>{:+.3f}</td>"
        "<td>{:,.0f}</td><td>{:,.0f}</td></tr>".format(
            escape(str(r["模型"])), r["交叉验证R2"], r["测试集R2"],
            r["测试集MAE"], r["测试集RMSE"]
        )
        for _, r in scores.iterrows()
    )
    page = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>北京二手房单价预测 · 综合实验</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", system-ui, sans-serif; }
    body { margin: 0; color: #17233b; background: #f5f7fb; line-height: 1.65; }
    main { max-width: 980px; padding: 36px 22px 64px; margin: auto; }
    h1 { margin-bottom: 4px; font-size: clamp(28px, 4vw, 42px); }
    h2 { margin-top: 38px; border-left: 4px solid #2a78d6; padding-left: 12px; }
    p { margin: 10px 0; }
    .sub { color: #546078; margin-bottom: 26px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
    .card, section { background: white; border-radius: 12px; box-shadow: 0 2px 14px #17233b0b; }
    .card { padding: 18px 20px; }
    .card strong { display: block; font-size: 25px; color: #1764b8; }
    .card span { color: #546078; font-size: 14px; }
    section { padding: 8px 22px 24px; margin-top: 20px; }
    img { width: 100%; height: auto; display: block; }
    .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 18px; }
    figure { margin: 0; background: white; border: 1px solid #e5eaf1; border-radius: 8px; overflow: hidden; }
    figcaption { padding: 9px 14px 14px; font-size: 14px; color: #546078; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid #e5eaf1; }
    th:first-child, td:first-child { text-align: left; }
    thead { background: #eaf2fd; }
    .note { color: #546078; font-size: 14px; }
  </style>
</head>
<body><main>
  <h1>北京二手房单价预测</h1>
  <p class="sub">应用综合实验 · 2017 年成交数据 · 预测目标：每平方米成交价格</p>
  <div class="cards">
    <div class="card"><strong>__ROWS__</strong><span>清洗后成交记录</span></div>
    <div class="card"><strong>3 种</strong><span>回归模型</span></div>
    <div class="card"><strong>__BEST_R2__</strong><span>最佳模型测试集 R²</span></div>
  </div>
  <section>
    <h2>1. 数据与做法</h2>
    <p>使用公开的北京链家二手房成交数据，选取 2017 年记录；剔除明显错误的面积和单价。输入包含面积、户型、房龄、区县及房屋设施等 16 项属性。总价、小区均价和成交时间不作为预测特征。</p>
    <p>按固定随机种子划分 __TRAIN__ 条训练记录与 __TEST__ 条测试记录。在训练集上进行 5 折交叉验证，比较线性回归、KNN 回归和决策树回归，再用独立测试集评估。缺失值填补和类别编码均放在模型管线内。</p>
    <div class="charts">
      <figure><img src="../figures/fig01_price_distribution.png" alt="2017 年北京二手房单价分布"><figcaption>单价分布与中位数</figcaption></figure>
      <figure><img src="../figures/fig02_district_median.png" alt="各区县成交单价中位数"><figcaption>区县之间的价格差异</figcaption></figure>
    </div>
  </section>
  <section>
    <h2>2. 模型对比</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>模型</th><th>5 折 CV R²</th><th>测试集 R²</th><th>测试集 MAE</th><th>测试集 RMSE</th></tr></thead>
      <tbody>__TABLE__</tbody>
    </table></div>
    <p class="note">MAE 和 RMSE 的单位均为元/㎡；均值基线只作参照，不计入三种机器学习模型。</p>
    <div class="charts">
      <figure><img src="../figures/fig03_model_scores.png" alt="交叉验证和测试集 R² 对比"><figcaption>同一数据划分下的模型表现</figcaption></figure>
      <figure><img src="../figures/fig04_predictions.png" alt="最佳模型预测单价与真实单价对比"><figcaption>最佳模型：__WINNER__；散点越接近斜线，预测越准确</figcaption></figure>
    </div>
  </section>
  <section>
    <h2>3. 结论与局限</h2>
    <p>按训练集交叉验证 R² 选择，表现最高的是<strong>__WINNER__</strong>；它在测试集上的 R² 为 <strong>__BEST_R2__</strong>，MAE 为 <strong>__BEST_MAE__ 元/㎡</strong>。</p>
    <p>本实验描述 2017 年成交记录，不能直接用于判断当前北京房价。原数据没有可靠的小区分组标识，随机划分的结果可能高估对全新区域房源的预测能力。</p>
  </section>
</main></body></html>
"""
    replacements = {
        "__ROWS__": f"{n_rows:,}",
        "__TRAIN__": f"{n_train:,}",
        "__TEST__": f"{n_test:,}",
        "__BEST_R2__": f"{winner['测试集R2']:+.3f}",
        "__BEST_MAE__": f"{winner['测试集MAE']:,.0f}",
        "__WINNER__": escape(str(winner["模型"])),
        "__TABLE__": rows,
    }
    for token, value in replacements.items():
        page = page.replace(token, value)
    (WEB / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    df = load_clean()
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"均值基线": DummyRegressor(strategy="mean"), **make_models()}

    rows = []
    predictions = {}
    for name, model in models.items():
        folds = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="r2", error_score="raise"
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        row = {
            "模型": name,
            "交叉验证R2": float(folds.mean()),
            "折间标准差": float(folds.std()),
            "训练集R2": float(model.score(X_train, y_train)),
            "测试集R2": float(r2_score(y_test, predicted)),
            "测试集MAE": float(mean_absolute_error(y_test, predicted)),
            "测试集RMSE": float(np.sqrt(mean_squared_error(y_test, predicted))),
        }
        rows.append(row)
        print(f"{name:8s} CV R²={row['交叉验证R2']:+.4f} "
              f"测试 R²={row['测试集R2']:+.4f} "
              f"MAE={row['测试集MAE']:,.0f} 元/㎡")

    scores = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS / "model_scores.csv", index=False, encoding="utf-8-sig")
    winner = scores.loc[scores["模型"] != "均值基线"].sort_values(
        "交叉验证R2", ascending=False
    ).iloc[0]
    make_figures(scores, y_test, predictions[str(winner["模型"])], str(winner["模型"]))
    make_page(scores, len(df), len(X_train), len(X_test), winner)
    print(f"最佳模型：{winner['模型']}；已更新 results、figures 和 web/index.html")


if __name__ == "__main__":
    main()
