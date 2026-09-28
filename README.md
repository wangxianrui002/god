# 北京二手房单价预测

《应用综合实验》的房价预测题目。使用 **2017 年北京二手房成交数据**，以每平方米成交价为预测目标，比较线性回归、KNN 回归和决策树回归三种模型。清洗后共有 **43,213 条记录**，满足课程对 1 万条以上数据和三种以上模型的要求。

## 运行

安装 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/) 后，在本目录运行：

```powershell
uv sync
uv run python src/03_eda.py
uv run python src/04_regression.py
```

项目已附带清洗后的 `data/lianjia_bj_clean.csv`，因此以上命令可以直接重画图和重新评估模型。需要从原始数据完整重建时，在绘图前再运行：

```powershell
uv run python src/01_download.py
uv run python src/02_clean.py
```

原始数据来自 [sni13/HousingPrice_Beijing](https://github.com/sni13/HousingPrice_Beijing)，由第三方采集自链家北京二手房成交页面，仅供课程学习。下载脚本会将原始 GB18030 编码转换为 UTF-8。

## 结果

训练集为 34,570 条，测试集为 8,643 条；交叉验证仅在训练集上进行。最佳模型是**决策树回归**：5 折 CV R² 为 **0.711**，测试集 R² 为 **0.720**，MAE 为 **9,462 元/㎡**。完整指标见 [实验报告](REPORT.md) 和 [结果表](results/model_scores.csv)。

四张图位于 `figures/`；双击 [展示页](web/index.html) 可查看简短汇报。展示页由 `src/04_regression.py` 根据本次结果生成，无需服务器。

数据是 2017 年的历史成交记录，不能当作当前北京房价报价。
