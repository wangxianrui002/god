# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 一、实验目标与要求（便于回忆与检查）

本仓库是《Python机器学习教程》第 7 章「综合练习」的完整实现，对应课程的综合实验任务：

> 基于三种以上机器学习模型的房价等综合实验，设计模型并进行数据、结果的量化分析。
> 综合实验要求：数据量充分（1W+），模型选用合理，计算过程科学，结果分析量化，
> 展示效果良好，实验讲解清晰。

**选题**：北京二手房单价（元/㎡）预测。题目原文只说「应用上述算法思想对股票、房价进行
预测」，没给数据、没指定算法、没给评分标准，所以选题、取数、建模、交付全部自定。

逐条对照（复核时按这张表点文件）：

| 要求 | 本项目如何满足 | 证据 |
|---|---|---|
| 三种以上模型 | **5 个**：`LinearRegression` / `Ridge` / `Ridge+Poly2` / `KNN` 四个取自教程 §6.1，外加 `DummyRegressor` 均值基线当及格线（R²≈0，任何模型低于它都算白做） | `src/common.py` `make_models()`；`results/model_scores.csv` |
| 数据量 1W+ | 原始 **318,851** 行 → 建模表 **43,213** 行，是要求的 4.3 倍 | `data/lianjia_bj_clean.csv`；README §3.2 |
| 模型选用合理 | 只挑教程讲过的回归算法，不引入教程外模型；部署模型刻意选 Ridge 而非 CV 更高的 Poly2，理由是外推风险 | README §9.2 |
| 计算过程科学 | 5 折交叉验证 + `GridSearchCV` 网格搜索 + 三道防泄露关卡 + 幂等脚本 + node 双重校验 | `src/common.py` `guard_no_leakage()`；README §五、§7.2、§9.3 |
| 结果分析量化 | R² / RMSE / 标准差 / η²=0.627 / 相关系数 / 辛普森悖论 / 泄露对照（+0.678 → +0.868） | README §四、§五、§七；`results/*.csv` |
| 展示效果良好 | 9 张图 + **自包含预测网页**（双击即开、离线可用、16 个输入项、3 张实时图） | `figures/fig01~09`；`web/index.html` |
| 实验讲解清晰 | README 十二节正文；关键取舍都写在 `src/common.py` 的注释里，不另设文档 | `README.md` |

**最有价值的结论（答辩时的主线）**：目标是单价而非总价，所以面积几乎不含信息
（`corr(面积, 单价) = -0.219`，而 `corr(面积, 总价) = +0.688`）；单价六成以上由区县决定
（η² = 0.627）；R² 停在 0.70 是因为数据没采集学区/朝向/楼层，**缺的是数据不是算法**。

---

## 二、常用命令

依赖用 `uv` 管理（Python 3.11 + scikit-learn 1.9.1 + pandas 3）。所有脚本的路径基于
`__file__` 解析，**在哪个目录调用都一样**。

```bash
uv sync                                # 按 uv.lock 还原依赖

uv run python src/01_download.py       # 下载数据，约 60 MB，需联网；已存在则跳过
uv run python src/01_download.py --force   # 强制重下
uv run python src/02_clean.py          # 切窗口 + 清洗 + 特征工程 + 泄露筛查   秒级
uv run python src/03_eda.py            # 探索性分析 → fig01~fig06             秒级
uv run python src/04_regression.py     # 5 个模型对比 → fig07/08              秒级
uv run python src/05_tune.py           # 网格搜索调参 → fig09                 ★ 几分钟
uv run python src/06_export.py         # 导出参数 + 渲染网页 + node 校验       秒级
```

- **`05_tune.py` 是唯一慢的**：43,213 行上跑 KNN 网格搜索，每次预测都要算一遍全表距离，
  已开 `n_jobs=-1` 并行，仍需几分钟。**别用默认 2 分钟超时去跑它**，会中途被杀。
  （README §二与源码注释对距离次数的说法不一致 —— 源码注释更贴近实测。）
- 六个脚本都**幂等**，随便重复跑。
- `06_export.py` 需要 **node**；没装时两道校验都打印 `[跳过]` 并继续，但会明确提示
  「网页逻辑未经验证」。
- 没有测试框架、没有 lint 配置。唯一的自动化校验是 `06_export.py` 里的两道 node 检查
  （页面脚本语法 + 逐行比对 43,213 行预测值），失败即 `assert`，拒绝发布。
- `figures/` 和 `results/` **进版本库**，所以重跑脚本会让工作区变脏 —— 这是预期的，
  提交时把图一起带上。

---

## 三、架构

### 3.1 六段式流水线

每个脚本只做一件事，产出物落盘，下一个脚本读盘。因此任何一段都可以单独重跑。

```
01_download  → data/lianjia_bj_raw.csv     318,851×26   (59 MB，不入库)
02_clean     → data/lianjia_bj_clean.csv    43,213×17   (3.8 MB，入库；不含任何价格字段)
             → results/leak_screen.csv, results/dropped_rows.csv
03_eda       → figures/fig01~fig06
04_regression→ figures/fig07~fig08, results/model_scores.csv
05_tune      → figures/fig09, results/tuning_results.csv
06_export    → web/model.json, web/index.html, results/app_test_cases.csv
```

**`src/common.py` 是唯一真源**，被 02~06 全部导入。它同时定义：

- 路径常量、`SEED = 42`、`TARGET = "单价"`；
- `WINDOW_YEARS` —— 取哪一年的数据；
- 编码 → 中文的映射表（`DISTRICT_NAMES` 等，**从数据实证得出，不是抄文档**）；
- `NUMERIC` / `CATEGORICAL` 特征清单与 `build_features()` 特征工程；
- **防泄露规则**（`DROP_ALWAYS` / `LEAK_NAME_PATTERN` / `LEAK_R2_THRESHOLD` / `guard_no_leakage()`）；
- `make_preprocessor()` / `make_models()` / `make_cv()` / `cv_scores()`；
- 绘图配色常量与中文字体设置。

**要改特征、改窗口、改模型清单，只改 `common.py`。** 这套规则绝不能有两份定义 ——
早晚会有一份先过期。

### 3.2 网页是生成物，不是源码

`web/index.html`（78 KB）由 `06_export.py` 渲染生成：

```
web/template.html  ─┐
web/predict.js     ─┼→ 06_export.py 替换 __PREDICT_JS__ / __MODEL_JSON__ / __DATA_JSON__ → web/index.html
web/model.json     ─┘
```

- **改了 `template.html` 或 `predict.js` 必须重跑 `06_export.py`**，否则改动不会出现在页面上。
- `index.html` 仍进版本库，是为了 clone 下来双击就能用。
- `predict.js` 是**一份实现、两处使用**：网页和 node 校验脚本共用它，所以「页面显示的数」
  和「Python 算的数」不会各写一份然后慢慢漂移。它把 Python 管线的每一步在 JS 里重放：
  `SimpleImputer(median)` → `StandardScaler` → `SimpleImputer(most_frequent)` →
  `OneHotEncoder(handle_unknown="ignore")` → `Ridge` 线性组合。
- **导出参数一律按 `float64` 原样写，不做任何四舍五入。** 曾经为了「让 JSON 好看」留 6 位
  小数，一万多的 one-hot 系数乘上去误差到 `1e-2` 量级，被 node 校验抓出来。当前实测最大
  差值 5.8e-11 元/㎡。
- 网页部署的是 `Ridge(alpha=0.178)`，**不是 CV 更高的 `Ridge+Poly2`**：差距约 0.6%，但二次
  函数在训练区间外会掉头向下（「关注人数填 500」能算出负单价）。网页输入域开放，这一分精度
  不值得换外推风险。同理 `FORM_RANGES` 给每个输入框卡了范围。

### 3.3 数据窗口

原始数据跨 2002–2018，这期间北京单价中位数从 3.8 万涨到 6.6 万。全量混在一起建模，
模型学到的大半会是「这套房是哪年卖的」而不是「这套房值多少」。所以 `WINDOW_YEARS`
默认 `(2017, 2017)` → 43,217 行，清洗后 **43,213** 行；单一年份因此**不需要把「成交年月」
放进特征**，价格纯粹由房屋属性解释。改这个常量即换窗口（`(2016,2017)` → 134,046 行；
`None` → 全量 318,851 行）。

---

## 四、不可违反的约定

这几条是本项目的正确性核心，动代码前先读 `README.md` §五、§六：

1. **防泄露三关**，建模前必须过 `guard_no_leakage()`：① 列名黑名单
   `LEAK_NAME_PATTERN`（抓 `总价_万`）；② `DROP_ALWAYS` 显式排除（抓 `totalPrice` /
   `communityAverage` / 经纬度 / `DOM` / 标识符）；③ 单特征 CV R² > 0.35 判泄露
   （抓 `小区均价`，它的列名里没有「价格」二字，只有 R² 拦得住）。
   **两道数值判据都拦不住 `总价_万`**（corr 只有 0.473、单特征 R² 只有 0.223），
   但 `总价_万 × 10000 ÷ 面积` 与单价的相关是 0.9999999998 —— 所以列名/语义那一层不能省。
   加进去 Ridge 的 CV R² 会从 0.678 虚高到 0.868。
   `02_clean.py` 有断言：建模表里不允许出现 `DROP_ALWAYS` 的任何一列。
   **不要为了「特征更多」而把价格派生量、按区县/小区分组的均价加回来**（target encoding 泄露）。
2. **缺失值一律保留 NaN**，交给 Pipeline 里的 `SimpleImputer`。**不要手工先填成中位数**
   —— 那会让「填补」这一步躲开交叉验证，等于训练时偷看验证集的中位数。剩余缺失：
   房龄 793、建筑类型 478、建筑结构 80、楼层位置 4、梯户比 1。
3. **异常值不删。** IQR 之外的点是核心城区与远郊的真实价差，不是错误值；按 IQR 硬删会切掉
   最贵的核心区样本，等于把最有效的预测依据删了。清洗阈值（单价 5,000~200,000、
   面积 10~1,000 ㎡）定得很宽，只剔录入错误（原始数据最小单价 136 元/㎡），实测只剔 4 行。
4. **稀有类别两步走**（`MIN_CATEGORY_COUNT = 200`）：先并成「其他」，并完桶还不够 200 行
   就设为 NaN 交给 `SimpleImputer(most_frequent)`。只做第一步不管用 —— `建筑类型=平房`
   只有 1 行，并成「其他」后这个桶还是 1 行，照样拿到 +10,460 元/㎡ 的系数。阈值刚好并掉
   平房(1)/结构1(30)/结构3(13)/结构5(37)，而 13 个区县最少的门头沟 267 行**全部保留**
   （区县是页面上必须能选的维度，不能并成「其他」）。
5. **README 里的数字是实测写死的**（0.696 / 13,306 / 62.7% / 5.8e-11 / 43,213 …）。
   **改动数据或模型后重跑，若数字变了，必须同步更新 README 对应小节**，否则文档与产出物
   互相矛盾。同理 `05_tune.py` 的子图副标题是**按实测缝隙算出来的，不是写死的** ——
   曾经照抄教科书的「alpha 太小 → 过拟合」，跑完才发现图上根本没有那道缝。
6. **图表用中文标签 + 浅色底，配色复用 `common.py` 的常量**（取自 dataviz 技能里已通过
   校验器的调色板），不要在脚本里新写颜色字面量。

---

## 五、已知的坑（都实际踩过）

**scikit-learn**

- `make_pipeline(('name', est), ...)` 在 1.9 起**不再接受 `(名字, 估计器)` 元组** ——
  它会把整个元组当成一个「估计器」，step 变成 `('tuple-1', ('imp', SimpleImputer(...)))`，
  `ColumnTransformer` 接着报 `All estimators should implement fit and transform`。
  **要自定义 step 名字，只能 `Pipeline([...])` + 显式列表**，见 `make_preprocessor()`。
- `cross_val_score` 的 `error_score` 默认是 `np.nan`，pipeline 出错会**静默**返回 nan，
  坏掉的模型看起来只是「效果差」。`cv_scores()` 里用 `error_score="raise"` + 断言。
- `OneHotEncoder` 必须 `handle_unknown="ignore"`：13 个区县在每一折里都可能有个别取值
  没出现在训练集，默认的 `"error"` 会直接崩。

**pandas 3**

- **字符串列的 dtype 是 `str` 而不再是 `object`**。写 `if df[c].dtype == object` 会
  **静默跳过所有字符串列**，`.strip()` 完全失效。本数据的字符串列带填充空格（`' 精装 '`），
  不 strip 会让 OneHotEncoder 把 `' 精装 '` 和 `'精装'` 当成两个类别。
- `groupby(...).apply(...)` 需要 `include_groups=False`，否则分组列同时出现在参数和结果里。

**数据源**

- 编码是 **GB18030 不是 UTF-8**，直接按 UTF-8 读会在字节 0 就 `UnicodeDecodeError`。
  下载后 `.decode('gb18030')` 再存 UTF-8，让编码问题只在 `01` 里出现一次。
- GitHub 个人仓库链接会失效，`SOURCES` 列表放主/备地址逐个重试并退避，全失败时打印手动
  下载指引。
- 原始数据没有附带码表，`district` 等字段是数字代码。**不要猜编码含义** —— 现有映射是从
  数据实证的（按代码分组算平均经纬度 + 10 个公开地标反查 60/60 命中；建筑类型靠层数与
  电梯率两个独立证据交叉验证）。网上流传的「1=板楼、4=平房」**与数据矛盾**，是错的。

**发布链路**

- `06_export.py` 有**两道**校验，缺一不可：第一道用 node 的 `vm.Script` 编译页面里的内联
  脚本（只查语法），第二道用 `predict.js` 整表跑 43,213 行与 Python 逐行比对。曾经模板里
  多写一个 `}`，页面打开是空表，而第二道仍打印 ✓ —— 因为它验的是 `predict.js` 里的预测
  函数，和页面脚本是两份代码。
- 比较用的期望值**不能取整**：写 `round(..., 6)` 光期望值自己就带 5e-7 误差，会淹没真差异。

---

## 六、提交与协作

- 提交信息用**中文 conventional commits**：`feat(data): 换用 4.3 万行数据集 —— 样本量从 306 提到 43,213`。
  正文常带具体数字。
- `data/lianjia_bj_raw.csv`（59 MB）与教材 PDF **不入库**（见 `.gitignore` 内的说明），
  清洗后的建模表入库。
- README 十二节是实验报告主体，结构与脚本一一对应（§3.1→01、§3.2/3.3→02、§四→03、
  §7.1→04、§7.2→05、§九→06）。改代码时按这个对应关系找要同步的章节。
- 已知局限已诚实写在 README §十一：KFold 有乐观偏差（同小区多套房会跨折，严格该用
  `GroupKFold`，但原始数据没有小区标识）、留出集被调参看过、部署模型是在全量上重拟合的
  因而页头 RMSE 来自留出集。**不要把这些说成没有**。
