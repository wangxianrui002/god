# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 一、实验目标与要求（便于回忆与检查）

本仓库是《Python机器学习教程》第 7 章「综合练习」的完整实现。题目原文只说「应用上述算法
思想对股票、房价进行预测」，**没给数据、没指定算法、没给评分标准**，所以选题、取数、建模、
交付全部自定。

> 综合实验要求：三种以上机器学习模型、数据量 1W+、模型选用合理、计算过程科学、
> 结果分析量化、展示效果良好、实验讲解清晰。

| 要求 | 本项目如何满足 | 证据 |
|---|---|---|
| 三种以上模型 | 第一版 **5 个**（`LinearRegression` / `Ridge` / `Ridge+Poly2` / `KNN` 取自教程 §6.1，外加 `DummyRegressor` 均值基线当及格线）；第二版 **9 个**（4 个教程模型 + `HistGBR` / `LightGBM` / `XGBoost` / `CatBoost` + `StackingRegressor`） | `make_models()`；`results/model_scores.csv`、`model_scores_v2.csv` |
| 数据量 1W+ | 第一版 **43,213** 行（4.3 倍）；第二版 **73,625** 行（7.4 倍） | `data/*_clean.csv`；README §四、§九 |
| 模型选用合理 | 第一版只挑教程讲过的算法；第二版**按文献调研**选 GBDT 家族 + 堆叠（`MODEL_SELECTION.md`），且四个 GBDT 全部实测、用自己数据说话而非照搬文献结论；部署模型刻意选 Ridge 而非 CV 更高的 Poly2，理由是外推风险 | `MODEL_SELECTION.md`；README §八、§九 |
| 计算过程科学 | 5 折交叉验证 + **GroupKFold 分组交叉验证**（量化乐观偏差 0.0402）+ 网格/随机搜索 + 三道防泄露关卡 + 幂等脚本 + node 双重校验；**搜索口径 = 汇报口径**（第二版都在 GroupKFold 上） | `make_group_cv()`；README §五、§七、§八、§九 |
| 结果分析量化 | R² / RMSE / 标准差 / η²=0.627 / 相关系数 / 辛普森悖论 / 泄露对照（+0.678 → +0.868）/ **「换数据 +0.0917」与「换模型 +0.0840」分开算** | README §三、§五、§七、§九；`results/*.csv` |
| 展示效果良好 | 13 张图 + **自包含汇报网页**（双击即开、离线可用；九节正文 + 10 张内联 SVG + 内嵌的 13 张实测图 + 结论卡片，末尾接一个 16 输入项的估价器） | `figures/fig01~13`；`web/index.html` |
| 实验讲解清晰 | README 十一节正文 + `MODEL_SELECTION.md`（选型依据）；关键取舍都写在源码注释里 | `README.md`、`MODEL_SELECTION.md` |

**最有价值的结论（答辩时的主线）**：目标是单价而非总价，所以面积几乎不含信息
（`corr(面积, 单价) = -0.219`，而 `corr(面积, 总价) = +0.688`）；单价六成以上由区县/板块决定
（η² = 0.627 / 0.641）；R² 停在 0.70 是因为数据没采集学区/朝向/楼层。

**第二版把「缺的是数据不是算法」验证了，也修正了它**：换数据确实涨 9 个点（+0.0917），
但换模型同样涨 8 个点（+0.0840）—— 因为线性模型**吃不下特征交互**，而新字段（朝向/装修/
楼层）的一元解释力低得惊人（η² 0.002~0.037），价值全在组合里。**两件事必须一起做。**

---

## 二、常用命令

依赖用 `uv` 管理（Python 3.11 + scikit-learn 1.9.1 + pandas 3）。所有脚本的路径基于
`__file__` 解析，**在哪个目录调用都一样**。

```bash
uv sync                                # 按 uv.lock 还原依赖

uv run python src/01_download.py       # 下载数据，约 60 MB，需联网；已存在则跳过（--force 重下）
uv run python src/02_clean.py          # 切窗口 + 清洗 + 特征工程 + 泄露筛查   秒级
uv run python src/03_eda.py            # 探索性分析 → fig01~fig06             秒级
uv run python src/04_regression.py     # 5 个模型对比 → fig07/08              秒级
uv run python src/05_tune.py           # 网格搜索调参 → fig09                 ★ 几分钟
uv run python src/06_export.py         # 导出参数 + 渲染网页 + node 校验       秒级
```

第二版（换数据 + GBDT，README §九）：

```bash
uv run python src/10_download_v2.py    # 下载挂牌数据集，约 26 MB，需联网
uv run python src/11_prepare_v2.py     # 清洗 + 特征工程 → data/house_v2_clean.csv
uv run python src/12_model_compare.py  # 9 个模型 × 2 套 CV                   ★ 约 10 分钟
uv run python src/12_model_compare.py --figs-only   # 只重画图，读已落盘的结果表
uv run python src/13_tune_v2.py        # GroupKFold 口径随机搜索   ★ 8 分钟（LightGBM 376 s）
uv run python src/13_tune_v2.py --figs-only         # 只重画图，读已落盘的搜索结果
uv run python src/14_eda_v2.py         # 新字段一元解释力 η²                 秒级
uv run python src/15_cat_iter_scan.py  # CatBoost 迭代数单向扫描   ★ 约 10 分钟（附录，非必跑）
```

`12_model_compare.py --figs-only` 除了重画 fig10/fig11，还会重算**排除性验证**
（`results/unseen_plate.csv`）—— 那一步只做分折数学、不拟合模型，秒级完成，
所以专门放在 `figs_only` 分支之外，不必为它等 10 分钟建模。

- **`05_tune.py` 和 `12_model_compare.py` 是慢的**。前者在 43,213 行上跑 KNN 网格搜索，每次
  预测都要算一遍全表距离，已开 `n_jobs=-1`，仍需几分钟；后者里的堆叠集成要跑内层 5 折 ×
  4 个基模型 × 外层 5 折，约 10 分钟（CatBoost 一个就 20 秒上下）。**别用默认 2 分钟超时去跑它们**。
  距离次数由 `knn_distances()` 按实际行数算出（≈1.8×10¹⁰），**不写死**。
- 所有脚本都**幂等**，随便重复跑。
- `06_export.py` 的其中两道校验需要 **node**；没装时它们打印 `[跳过]` 并继续，但模板引用那道
  照跑，页脚也如实写「本次导出没跑 node 校验」（**这条路径靠 monkeypatch 掉 `verify_with_node`
  模拟跑过**，因为本机永远有 node —— 页脚现在由 `report.verify.text` 这个**字符串**驱动，
  取不到值会让整页数字都不填）。
- 没有测试框架、没有 lint 配置。唯一的自动化校验是 `06_export.py` 里的**三道**检查，失败即
  `assert`，拒绝发布：① 模板里的数字路径与 `$("id")` 元素引用是否都存在；② node 编译页面
  脚本（只查语法）；③ node 用 `predict.js` 逐行比对 43,213 行预测值。
- `figures/` 和 `results/` **进版本库**，所以重跑脚本会让工作区变脏 —— 这是预期的，
  提交时把图一起带上。

---

## 三、架构

### 3.1 两套流水线：第一版（`01`~`06`）与第二版（`10`~`15`）

每个脚本只做一件事，产出物落盘，下一个脚本读盘。因此任何一段都可以单独重跑。
**用 `1x` 编号是为了让两套流水线并存但不混淆**：第一版的产出物一个都没动，
「换数据」和「换模型」各值多少才算得出来。

```
第一版 —— 链家成交数据 + 教程里的线性模型，README §一~§八
01_download  → data/lianjia_bj_raw.csv     318,851×26   (59 MB，不入库)
02_clean     → data/lianjia_bj_clean.csv    43,213×17   (3.8 MB，入库；不含任何价格字段)
             → results/leak_screen.csv, results/dropped_rows.csv
03_eda       → figures/fig01~fig06, results/simpson_age.csv  (fig06 的三个 r，网页要引用)
04_regression→ figures/fig07~fig08, results/model_scores.csv
             → results/leak_demo.csv   (真把「总价_万」放进特征再跑一遍 Ridge：0.678 → 0.868)
05_tune      → figures/fig09, results/tuning_results.csv
             → results/val_curve_v1.csv   (验证曲线上 10 个读数，网页第七节整节的论据)
06_export    → web/model.json, web/index.html (4.1 MB), results/app_test_cases.csv
             ← 读 results/ 下 12 张表，把数注入模板；自己不产生实验结论数字

第二版 —— Kaggle 挂牌数据 + GBDT 家族，README §九
10_download_v2 → data/house_v2_raw.csv      73,685×22   (26 MB，不入库)
11_prepare_v2  → data/house_v2_clean.csv    73,625×18   (7.5 MB，入库；16 特征 + 单价 + 小区)
               → results/leak_screen_v2.csv, results/dropped_rows_v2.csv
12_model_compare→ figures/fig10~fig11, results/model_scores_v2.csv, model_folds_v2.csv
               → results/unseen_plate.csv   (排除性验证：落差不是「整片板块没见过」)
13_tune_v2     → figures/fig12, results/tuning_results_v2.csv
14_eda_v2      → figures/fig13, results/eta2_v2.csv
15_cat_iter_scan → results/cat_iter_scan.csv  (附录：CatBoost 只扫迭代数)
```

**`src/common.py` 与 `src/common_v2.py` 的分工**：与数据集**无关**的规则（随机种子、
防泄露三道关、交叉验证、RMSE、绘图配色）只在 `common.py` 定义一次，`common_v2.py`
**导入**它们，绝不重新定义 —— 那套规则只能有一份，有两份早晚会有一份先过期。
`common.py` 是**唯一真源**，被 02~06 全部导入（第二版经 `common_v2.py` 间接复用）。
`common_v2.py` 只放第二版特有的：字段映射、清洗阈值、`GbdtFrame` 预处理器、
`make_group_cv()`、`CatBoostCategorical`。

`common.py` 同时定义：路径常量、`SEED = 42`、`TARGET = "单价"`、`WINDOW_YEARS`、
编码 → 中文的映射表（**从数据实证得出，不是抄文档**）、`NUMERIC` / `CATEGORICAL` 特征清单与
`build_features()`、防泄露规则（`DROP_ALWAYS` / `LEAK_NAME_PATTERN` / `LEAK_R2_THRESHOLD` /
`guard_no_leakage()`）、`make_preprocessor()` / `make_models()` / `make_cv()` / `cv_scores()`、
绘图配色常量与中文字体设置。

**要改特征、改窗口、改模型清单，只改 `common.py`。**

### 3.2 网页是生成物，不是源码

`web/index.html`（4.1 MB）由 `06_export.py` 渲染生成。它**一份文件兼两个用途**：九节课堂
汇报（§01 题目与做法 / §02 防泄露 / §03 第一版 5 模型 / §04 第二版 9 模型 × 2 套 CV /
§05 换数据 vs 换模型 / §06 η² / §07 调参 / §08 局限 / §09 现场演示），末尾接一个可现场操作的
估价器。

```
web/template.html  ─┐
web/predict.js     ─┼→ 06_export.py 替换 __PREDICT_JS__ / __MODEL_JSON__ / __DATA_JSON__ → web/index.html
web/model.json     ─┘
```

- **改了 `template.html` 或 `predict.js` 必须重跑 `06_export.py`**，否则改动不会出现在页面上。
- `index.html` 仍进版本库，是为了 clone 下来双击就能用。
- **汇报页上的数字一个都不许手抄。** 正文里写 `<span data-n="report.a.b:格式">`，
  `06_export._report()` 从 `results/*.csv` 读出来注入，`fillNumbers()` 初始化时解析，
  **取不到值就抛错**（不静默留破折号）。格式串在 `FMT` 里，R² 的**差值**要用 `pp1`/`pp0`
  （「9.2 个百分点」），不能用 `pct1`（会读成「相对涨了 9.2%」）。
  为这条规矩，`03_eda.py` 把本来只打印的辛普森三个 r 落盘成 `results/simpson_age.csv`。
- 汇报部分的图有**两种**，各有各的用处，别混为一谈：
  ① **10 张内联 SVG**，由注入的 JSON 现画 —— 用于模型对比这类需要跟着页面配色走的图，
  也会随主题换色；**估价器那三张会响应输入**。
  ② **13 张实测分析图**（`figures/fig01~13`，`06_export.figures()` 读成 `data:` URI 内嵌），
  是报告里的原图，浅底白卡片呈现，不随主题换色（PNG 是死的）。**文件 4.1 MB 基本都是它们**。
  之所以内嵌而不是写 `<img src="figures/...">`：网页的交付形态是**单个文件、双击即开**，
  写相对路径的话把 `index.html` 单独拷走图就全裂了。模板里用 `__FIG_<stem>__` 占位，
  `render()` 会双向断言（模板引用了的图必须存在、读到的图必须被模板引用），
  免得图裂成一行占位符字符串而校验还打 ✓。
- `predict.js` 是**一份实现、两处使用**：网页和 node 校验脚本共用它，所以「页面显示的数」和
  「Python 算的数」不会各写一份然后慢慢漂移。它把 Python 管线的每一步在 JS 里重放：
  `SimpleImputer(median)` → `StandardScaler` → `SimpleImputer(most_frequent)` →
  `OneHotEncoder(handle_unknown="ignore")` → `Ridge` 线性组合。
- **导出参数一律按 `float64` 原样写，不做任何四舍五入。** 曾经为了「让 JSON 好看」留 6 位
  小数，一万多的 one-hot 系数乘上去误差到 `1e-2` 量级，被 node 校验抓出来。当前实测最大
  差值 5.8e-11 元/㎡。
- **估价器部署的是 `Ridge(alpha=0.178)`，不是 CV 更高的 `Ridge+Poly2`**：差距 1.8 个百分点，
  但二次函数在训练区间外会掉头向下（「关注人数填 500」能算出负单价）。网页输入域开放，这一分
  精度不值得换外推风险。同理 `FORM_RANGES` 给每个输入框卡了范围。**第二版的 GBDT 结果只以
  图表形式汇报、不接进估价器** —— Ridge 39 个系数能逐行搬到 JS，GBDT 集成不能。

### 3.3 数据窗口

原始数据跨 2002–2018，这期间北京单价中位数从 3.8 万涨到 6.6 万。全量混在一起建模，模型学到
的大半会是「这套房是哪年卖的」而不是「这套房值多少」。所以 `WINDOW_YEARS` 默认
`(2017, 2017)` → 43,217 行，清洗后 **43,213** 行；单一年份因此**不需要把「成交年月」放进
特征**，价格纯粹由房屋属性解释。改这个常量即换窗口（`(2016,2017)` → 134,046 行；
`None` → 全量 318,851 行）。

---

## 四、不可违反的约定

这几条是本项目的正确性核心，动代码前先读 `README.md` §五、§六：

1. **防泄露三关**，建模前必须过 `guard_no_leakage()`：① 列名黑名单 `LEAK_NAME_PATTERN`
   （抓 `总价_万`）；② `DROP_ALWAYS` 显式排除（抓 `totalPrice` / `communityAverage` /
   经纬度 / `DOM` / 标识符）；③ 单特征 CV R² > 0.35 判泄露（抓 `小区均价`，它的列名里没有
   「价格」二字，只有 R² 拦得住）。**两道数值判据都拦不住 `总价_万`**（corr 只有 0.473、
   单特征 R² 只有 0.223），但 `总价_万 × 10000 ÷ 面积` 与单价的相关是 0.9999999998 ——
   所以列名/语义那一层不能省。加进去 Ridge 的 CV R² 会从 0.678 虚高到 0.868。
   `02_clean.py` 有断言：建模表里不允许出现 `DROP_ALWAYS` 的任何一列。
   **不要为了「特征更多」而把价格派生量、按区县/小区分组的均价加回来**（target encoding 泄露）。
2. **缺失值一律保留 NaN**，交给 Pipeline 里的 `SimpleImputer`。**不要手工先填成中位数** ——
   那会让「填补」这一步躲开交叉验证，等于训练时偷看验证集的中位数。剩余缺失：房龄 793、
   建筑类型 478、建筑结构 80、楼层位置 4、梯户比 1。
3. **异常值不删。** IQR 之外的点是核心城区与远郊的真实价差，不是错误值；按 IQR 硬删会切掉
   最贵的核心区样本，等于把最有效的预测依据删了。清洗阈值（单价 5,000~200,000、面积
   10~1,000 ㎡）定得很宽，只剔录入错误（原始数据最小单价 136 元/㎡），实测只剔 4 行。
4. **稀有类别两步走**（`MIN_CATEGORY_COUNT = 200`）：先并成「其他」，并完桶还不够 200 行就
   设为 NaN 交给 `SimpleImputer(most_frequent)`。只做第一步不管用 —— `建筑类型=平房` 只有
   1 行，并成「其他」后这个桶还是 1 行，照样拿到 +10,460 元/㎡ 的系数。阈值刚好并掉
   平房(1)/结构1(30)/结构3(13)/结构5(37)，而 13 个区县最少的门头沟 267 行**全部保留**
   （区县是页面上必须能选的维度，不能并成「其他」）。
5. **README 里的数字是实测写死的**（0.696 / 13,306 / 62.7% / 5.8e-11 / 43,213 …）。
   **改动数据或模型后重跑，若数字变了，必须同步更新 README 对应小节**，否则文档与产出物
   互相矛盾。同理 `05_tune.py` 的子图副标题是**按实测缝隙算出来的，不是写死的** ——
   曾经照抄教科书的「alpha 太小 → 过拟合」，跑完才发现图上根本没有那道缝。
6. **图表用中文标签 + 浅色底，配色复用 `common.py` 的常量**（取自 dataviz 技能里已通过校验器
   的调色板），不要在脚本里新写颜色字面量。

---

## 五、已知的坑（都实际踩过）

**scikit-learn**

- `make_pipeline(('name', est), ...)` 在 1.9 起**不再接受 `(名字, 估计器)` 元组** —— 它会把
  整个元组当成一个「估计器」，step 变成 `('tuple-1', ('imp', SimpleImputer(...)))`，
  `ColumnTransformer` 接着报 `All estimators should implement fit and transform`。
  **要自定义 step 名字，只能 `Pipeline([...])` + 显式列表**，见 `make_preprocessor()`。
- `cross_val_score` 的 `error_score` 默认是 `np.nan`，pipeline 出错会**静默**返回 nan，
  坏掉的模型看起来只是「效果差」。`cv_scores()` 里用 `error_score="raise"` + 断言。
- `OneHotEncoder` 必须 `handle_unknown="ignore"`：13 个区县在每一折里都可能有个别取值没有
  出现在训练集，默认的 `"error"` 会直接崩。
- **自定义估计器的继承顺序必须是 `(RegressorMixin, BaseEstimator)`，mixin 在前。** 写反了
  `BaseEstimator.__sklearn_tags__` 排在 MRO 前面且不调用 `super()`，`RegressorMixin` 那版
  永远轮不到，`estimator_type` 一直是 `None`，`is_regressor()` 返回 False，堆叠集成报
  **"The estimator Pipeline should be a regressor."** —— 报错信息完全指不到继承顺序上。
  见 `common_v2.CatBoostCategorical`。

**第二版 GBDT（`common_v2.py`，三个坑都不报错或报错指不到根因）**

- **`ColumnTransformer` 会把数值列也变成 `object`。** 它把各列 hstack 成一个数组，float64
  与 object 混在一起会被统一提升成 object。改用 `GbdtFrame` 逐列构造 DataFrame。
- **`X.iloc[:, i] = X.iloc[:, i].astype("category")` 是无效赋值。** iloc 是就地写入已有的
  object 块，pandas 会把 Categorical 还原成原值，dtype 一点没变，**不报任何错**。
  必须用标签赋值 `X[c] = ...`。
- **类别表必须在 `fit` 时定死，不能在 `transform` 里现推。** 训练折和验证折各推一套，顺序
  未必相同；XGBoost 按 `cat.codes`（类别在类别表里的**下标**）读数据，下标一错位整列含义就
  变了 —— 不报错，只是把「南北」当成「东」。症状是 **GroupKFold 的 R² 掉成 -0.1461（负数）
  而 KFold 一切正常**，根因却在编码。LightGBM / CatBoost 按类别**取值**处理，没这个问题。
- **`CatBoostRegressor.get_params()` 返回 `cat_features` 的副本**，而 `sklearn.clone` 有
  「构造器必须原样保存参数」的硬断言，`cross_val_score` / `StackingRegressor` 必崩：
  `Cannot clone object ... as the constructor either does not set or modifies parameter`。
  修法见 `CatBoostCategorical`（把声明挪进 `fit`）。另外 CatBoost **不会**从 pandas
  category dtype 自动识别类别列，不声明直接报错。
- **`GroupKFold` 不打乱。** 原始 CSV 按板块/环线排序，直接跑会让每一折落在连续的地理区块上，
  「没见过的小区」和「没见过的区域」混在一起。必须先 `shuffle_once()` 打乱行序（只改变哪些
  小区进哪一折，不破坏分组完整性），才能和 `shuffle=True` 的 `KFold` 对比。

**pandas 3**

- **字符串列的 dtype 是 `str` 而不再是 `object`**。写 `if df[c].dtype == object` 会**静默跳过
  所有字符串列**，`.strip()` 完全失效。本数据的字符串列带填充空格（`' 精装 '`），不 strip
  会让 OneHotEncoder 把 `' 精装 '` 和 `'精装'` 当成两个类别。
- `groupby(...).apply(...)` 需要 `include_groups=False`，否则分组列同时出现在参数和结果里。

**数据源**

- 编码是 **GB18030 不是 UTF-8**，直接按 UTF-8 读会在字节 0 就 `UnicodeDecodeError`。下载后
  `.decode('gb18030')` 再存 UTF-8，让编码问题只在 `01` 里出现一次。
- GitHub 个人仓库链接会失效，`SOURCES` 列表放主/备地址逐个重试并退避，全失败时打印手动
  下载指引。
- 原始数据没有附带码表，`district` 等字段是数字代码。**不要猜编码含义** —— 现有映射是从数据
  实证的（按代码分组算平均经纬度 + 10 个公开地标反查 60/60 命中；建筑类型靠层数与电梯率两个
  独立证据交叉验证）。网上流传的「1=板楼、4=平房」**与数据矛盾**，是错的。

**发布链路**

- `06_export.py` 有**三道**校验，缺一不可：① 模板里的 `<span data-n="...">` 路径与
  `$("id")` 元素引用是否都存在（`verify_template_refs`）；② node 的 `vm.Script` 编译页面里的
  内联脚本（只查语法）；③ `predict.js` 整表跑 43,213 行与 Python 逐行比对。**每一道都是被一次
  真实事故补上的**：模板里多写一个 `}`，页面打开是空表而 ③ 仍打印 ✓（它验的是 `predict.js`
  里的预测函数，和页面脚本是两份代码）→ 补了 ②；改模板时删掉了一个被 `<span>` 指向的元素、
  `fillNumbers` 抛 TypeError 导致整页数字全不填，而 ②③ 都只验预测逻辑 → 补了 ①。
- **`verify_template_refs` 的扫描范围最容易写反**：`$("id")` 全写在 `<script>` 里，被引用的
  id 定义在标签里，所以要在**整份**模板里找 `$(`、在**去掉 `<script>` 后**找 `id=`。写反了
  这项检查就是在空跑（曾经扫出「0 处元素引用」还打 ✓），所以它末尾有 `assert refs` 兜底。
- **校验顺序不能随便调**：模板里有一处 `<span data-n="report.verify.text">` 引用的是
  **校验结果本身**，所以 node 比对必须排在模板检查**前面**。顺序反了会报「模板引用了不存在的
  东西」，而真正的问题是顺序。
- 比较用的期望值**不能取整**：写 `round(..., 6)` 光期望值自己就带 5e-7 误差，会淹没真差异。
- **模板引用的数一律走注入，禁止在正文里手抄。** `report.*` 的字段在 `_report()` 里定义，
  要加一个就在那儿加；`results/` 里没有的数（比如辛普森的组内 r）先让产出脚本落盘成 CSV，
  再读进来 —— 不要在 `06_export.py` 里现算一份，那会绕开「产出脚本是唯一真源」这条。

---

## 六、提交与协作

- 提交信息用**中文 conventional commits**：`feat(data): 换用 4.3 万行数据集 —— 样本量从 306
  提到 43,213`。正文常带具体数字。
- `data/lianjia_bj_raw.csv`（59 MB）与教材 PDF **不入库**（见 `.gitignore` 内的说明），
  清洗后的建模表入库。
- README 是实验报告主体，只保留结论与数字；**结构与脚本的对应关系看本文件 §3.1**
  （README 已压到 100 行以内，不再逐脚本展开）。改代码时按 §3.1 的产出物表找要同步的数字。
- 已知局限已诚实写在 README §十：KFold 有乐观偏差（同小区多套房会跨折，严格该用
  `GroupKFold`，但原始数据没有小区标识）、留出集被调参看过、部署模型是在全量上重拟合的因而
  页头 RMSE 来自留出集。**不要把这些说成没有。**
