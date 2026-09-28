"""两张图说明预测目标的分布和区县差异。"""
from __future__ import annotations

import numpy as np

from common import FIGURES, TARGET, load_clean, setup_plot


def main() -> None:
    import matplotlib.pyplot as plt

    setup_plot()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_clean()

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(df[TARGET], bins=np.arange(0, 160_001, 5_000), color="#2a78d6", edgecolor="white")
    ax.axvline(df[TARGET].median(), color="#ed6a36", linewidth=2,
               label=f"中位数 {df[TARGET].median():,.0f} 元/平米")
    ax.set(title="2017 年北京二手房成交单价分布", xlabel="单价（元/平米）", ylabel="房源数量")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_price_distribution.png")
    plt.close(fig)

    district = df.groupby("区县")[TARGET].agg(["median", "count"]).sort_values("median")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    bars = ax.barh(district.index, district["median"], color="#2a78d6")
    for bar, count in zip(bars, district["count"]):
        ax.text(bar.get_width() + 1_000, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():,.0f}（n={count:,}）", va="center", fontsize=8)
    ax.set_xlim(0, district["median"].max() * 1.40)
    ax.set(title="各区县成交单价中位数", xlabel="单价（元/平米）")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_district_median.png")
    plt.close(fig)

    print(f"样本 {len(df):,} 条；单价中位数 {df[TARGET].median():,.0f} 元/㎡")
    print("已生成单价分布图和区县对比图。")


if __name__ == "__main__":
    main()
