# 模型选型与文献依据

> 本文回答两个问题：**为什么是这几个模型**，以及**为什么用这种验证方式**。
> 实测数字在 `results/model_scores_v2.csv`，结论在 `README.md`；本文只讲「凭什么这么选」。
> 最后更新：2026-09-27。

## 一、结论先说

| 决策 | 依据 |
|---|---|
| 线性模型（线性回归 / 岭回归）留作**对照**，不当主模型 | 文献一致：在表格型房价数据上，线性模型系统性地输给梯度提升树；但它便宜、可解释，是必要的下限参照 |
| 主模型选 **GBDT 家族**（LightGBM / XGBoost / CatBoost / HistGBR） | 多份对比研究报告 GBDT 稳定优于线性模型与随机森林；且本项目 75% 的特征是类别型的（板块清洗后 129 个取值），树模型的原生类别处理正好对上 |
| 用**四个** GBDT 而不是一个 | CatBoost / XGBoost / LightGBM 谁第一**随数据集变**，没有普适冠军。选一个等于赌，选四个才能用实测说话 |
| 堆叠 **Stacking + Ridge 元学习器**作为**待检验对象**，而非预设的最终方案 | 文献一致报告堆叠优于任何单一基模型（R² 约 0.933 vs 单模型 0.929~0.931），Ridge 元学习器是标准配置 —— **所以它值得一试**。但本项目实测**没有复现这个结论**（见 §三） |
| 交叉验证用 **GroupKFold（按小区分组）**，并保留普通 KFold 作对照 | 空间自相关会让随机划分的 CV **系统性高估**精度；分组/分块 CV 的估计更接近真实样本外误差 |
| 不做分类 / 聚类 / PCA / 神经网络 | 目标是连续价格；文献里这类任务的 SOTA 全是树模型集成，上深度网络属于用大炮打蚊子，且失去可解释性 |

## 二、GBDT 家族：文献怎么说

多份独立研究在房价数据上对比过这三个库，结论高度一致但不完全重合：

| 研究 | 数据集 | 第一名 | 关键数字 |
|---|---|---|---|
| Robust Stacked Ensemble (PPASP) | Ames 房价 1,460 行 | XGBoost | R² 0.9298，CatBoost 0.9293、LightGBM 0.9225 紧随 |
| Billah & Sarker | Ames 房价 1,460 行 | CatBoost | log-RMSE 0.1275，R² 0.9129 |
| U.S. Housing Data (Mundo FESC) | 美国住房数据 | CatBoost | 测试 R² 0.86 |
| Nature Sci. Reports 基线 | 房价数据 | XGBoost | R² 98.25%，CatBoost 98.18%、LightGBM 97.82% |
| Madrid 市场 (UPM 硕士论文) | 马德里 | Random Forest | RF 0.8966 反而最高，CatBoost 0.8139 |

**这张表最重要的一行是最后一行**：在马德里数据上，随机森林反而赢了三个 GBDT。说明「哪个库
最好」**依赖数据集和特征工程，没有普适冠军**。

所以本项目不采用「文献说 CatBoost 好，那就用 CatBoost」这种论证 —— 那是把别人的数据集上的
结论硬套到自己的数据上。本项目的做法是：**四个都放进去，用自己这份数据实测，用 GroupKFold
的结果说话。** 文献在这里的作用是划定候选范围（GBDT 家族 + 堆叠），不是替我下结论。

**为什么类别的处理方式很关键**：本数据集的类别特征里，「板块」清洗后有 **129** 个取值
（合并稀有类别之前 270 个，`README §九` 的 η² 数字用的是 129 这一列）。对线性模型这是
灾难 —— one-hot 之后 129 维稀疏列，且「板块 A 与板块 B 相邻」这类信息完全丢失。三个 GBDT 库
各有各的原生类别处理（CatBoost 的有序目标统计、LightGBM 的类别分裂、XGBoost 的
`enable_categorical`），能直接用上「同一个板块」这个信息。

**但三家对类别特征的编码约定不同，这是本项目踩坑最多的地方**：XGBoost 按类别在类别表里的
**下标**读数据，训练折和验证折的类别表一旦不同，下标错位，整列含义就变了 —— 它不报错，
静默给出错误结果。详见 `CLAUDE.md` §五「已知坑」。

## 三、堆叠集成：文献怎么说

PPASP 那篇的做法与本项目几乎一致（XGBoost + CatBoost + LightGBM 作基学习器，Ridge 元学习器
α=1.0，用 out-of-fold 预测当元特征避免泄露）：

| 模型 | R² | RMSLE |
|---|---|---|
| 加权平均集成 | 0.9334 | 0.1138 |
| **Stacking + Ridge 元学习器** | **0.9330** | **0.1132** |
| RF + CatBoost 混合 | 0.9313 | — |

要点：堆叠**优于任何单一基模型**（单模型最好 0.9298）；元特征必须用 **out-of-fold 预测**
生成，直接用基模型在训练集上的拟合值再喂给元学习器，等于把答案抄进去 —— 这是堆叠里最容易
犯的泄露，与第一版「总价_万」是同一类错误。sklearn 的 `StackingRegressor` 默认就是这么做的
（`cv` 参数控制），本项目沿用，没有自己手搓。

另一个研究报告：混合 Stacking Regressor **优于所有单体算法**，且 CatBoost 单体误差最低。

**这两个说法在本项目上都没有复现，必须写清楚**：

| 文献结论 | 本项目实测（GroupKFold 口径） |
|---|---|
| 堆叠优于所有单一基模型 | **未复现**。堆叠 +0.8124，低于 HistGBR +0.8135（README §九）—— 它在普通 KFold 上第一，在诚实的 GroupKFold 口径上掉到第二 |
| CatBoost 单体误差最低 | **未复现**。CatBoost +0.7790，四库里垫底 |

对堆叠不成立的原因有一个具体解释：它的四个基模型**太像了** —— HistGBR 与 LightGBM 的
GroupKFold 分数只差 0.0023，元学习器没有互补信息可用，多出来的一层反而多了一层过拟合。
文献里堆叠能赢，前提是基模型之间**有差异**。

对 CatBoost 则**不能**下「它就是差」的结论：迭代数扫描显示它只是没训够（400 → 1600 轮单调
升到 +0.7954，仍未收敛），而主表里所有模型一律 400 轮。详见 `README.md` §九。

**这两条是本项目与文献不一致的地方，也是本报告比「照抄文献结论」更有价值的地方**：文献用来
划定候选范围（GBDT 家族 + 堆叠），**用哪把尺子量、量出来是多少，得自己跑**。

## 四、验证方式：为什么必须用 GroupKFold

这是本项目方法学上最值得讲的一点，而且有明确的文献支撑。

**问题**：房价数据有强烈的**空间自相关**（Tobler 第一定律：近的房子比远的更像）。普通 KFold
**随机**划分，训练集和验证集里都会有空间上紧邻的房子，于是模型可以「从空间上邻近的验证样本
那里拿到信息」—— 留一/随机 CV 所依赖的「残差独立」假设被打破了。

Deppner 在法兰克福公寓租金数据上做了直接对比：**非空间（随机）CV 的误差系统性偏乐观**
（RF 和 XGBoost 都是）；**空间 CV 的误差略偏悲观，但可靠得多**，接近真实样本外误差。作者的
结论是：随机 CV 的误差应当被视为真实误差的**下界**，空间 CV 的误差视为**上界**（更保守）。

还有一层机制值得注意：当模型缺少位置/邻里特征时，空间结构会被**其他恰好按空间排列的属性**
「顶替」着学进去（overfitted to covarying but non-causal regressors）—— 这正好解释了本项目
里 KNN 的落差为什么最大：KNN 完全靠「附近样本的均值」预测，它对空间结构最敏感。

**本项目的落地**：第一版 README §十 #1 承认了这个乐观偏差，但当时**没有小区标识列**，
修不了。第二版数据带 `community` 列（6,833 个小区，平均 10.8 套/小区），于是这一条从
「已知局限」升级成「已量化」：普通 KFold 与 GroupKFold（整个小区只进一侧）各跑一遍，差值即
乐观偏差；两个方案**都先按固定种子打乱行序** —— `GroupKFold` 本身不打乱，而原始 CSV 按
板块/环线排序，直接跑会让「没见过的小区」和「没见过的区域」两件事混在一起，落差就说不清是
谁造成的。打乱只改变「哪些小区进哪一折」，不破坏小区完整性。见 `common_v2.shuffle_once()`。

BORG 那个 R 包做的正是这件事：把「同组同时出现在训练和测试」定义为硬违规，并提供
`borg_compare_cv()` 把随机 CV 和分块 CV **并排跑**，再用配对 t 检验量化指标虚高了多少 ——
与本项目的做法一致。

**一个必须写清楚的边界**：按小区分组只挡住了「同小区泄露」，**没有**挡住「相邻小区泄露」——
空间自相关依然存在于组与组之间。所以 GroupKFold 的结果**仍然偏乐观**，只是乐观得少一些；
它比普通 KFold 更接近真实样本外误差，但**不等于**真实样本外误差；真正严格的做法是按
**地理区块**分折（spatial blocking），而第二版数据源本身不含经纬度列，做不了。
这条边界要写进报告，不能宣称「偏差已消除」。

## 五、参考文献

- [A Robust Stacked Ensemble Approach for House Price Prediction (PPASP)](http://www.ppaspk.org/index.php/PPAS-A/article/download/1660/1117/6635)
- [Accounting for Spatial Autocorrelation in Algorithm-Driven Hedonic Models: A Spatial Cross-Validation Approach (ERES)](https://eres.architexturez.net/system/files/P_20220131104853_8417.pdf)
- [Deppner 博士论文：空间自相关下的房价/租金模型评估](https://epub.uni-regensburg.de/54649/1/Dissertation_Deppner_Juergen_Pflichtexemplar.pdf)
- [BORG: Bounded Outcome Risk Guard for Model Evaluation（CRAN）](https://cran.ma.ic.ac.uk/web/packages/BORG/refman/BORG.html)
- [Spatial+: A new cross-validation method to evaluate geospatial machine learning models](https://core.ac.uk/works/143419251/)
- [A hybrid machine learning approach for housing price prediction: the stacking regressor method (ScienceDirect)](https://www.sciencedirect.com/org/science/article/abs/pii/S1753827026000051)
- [Enhancing House Price Prediction Using Ensemble Machine Learning Models (Mundo FESC)](https://www.fesc.edu.co/Revistas/OJS/index.php/mundofesc/article/view/1885)
