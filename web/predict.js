/**
 * 纯预测函数 —— 把 Python 里拟合好的 Ridge 管线照搬到 JavaScript。
 *
 * 页面和 node 校验脚本共用这一份实现，所以「网页上显示的数字」和
 * 「Python 算出来的数字」必然是同一套逻辑，不会各写一份然后慢慢漂移。
 *
 * 对应 Python 侧的等价物：
 *     pre = ColumnTransformer([
 *         ("num", Pipeline([("imp", SimpleImputer(median)),
 *                           ("sc",  StandardScaler())]),  NUMERIC),
 *         ("cat", Pipeline([("imp", SimpleImputer(most_frequent)),
 *                           ("oh",  OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
 *     ])
 *     y = Ridge(alpha).fit(pre.transform(X), y).predict(...)
 */
(function (root) {
  "use strict";

  /**
   * @param {object} model  web/model.json 里的模型参数
   * @param {object} input  {面积: 91.5, 房间数: 2, ..., 地段: "望京", 装修: "精装", 形式: "板楼"}
   * @returns {number}      预测单价（元/㎡）
   */
  function predictUnitPrice(model, input) {
    var x = [];
    var i, j, v;

    // ---- 数值列：缺失填训练集中位数，再标准化 ----
    for (i = 0; i < model.numeric_cols.length; i++) {
      v = input[model.numeric_cols[i]];
      if (v === null || v === undefined || v === "" || isNaN(Number(v))) {
        v = model.impute_median[i];
      }
      x.push((Number(v) - model.scaler_mean[i]) / model.scaler_scale[i]);
    }

    // ---- 类别列：缺失填训练集众数，再 one-hot ----
    // handle_unknown="ignore" 的等价行为：取值不在训练集类别里时，整组输出全 0。
    for (i = 0; i < model.cat_cols.length; i++) {
      v = input[model.cat_cols[i]];
      if (v === null || v === undefined || v === "") {
        v = model.cat_fill[i];
      }
      var levels = model.categories[i];
      for (j = 0; j < levels.length; j++) {
        x.push(v === levels[j] ? 1 : 0);
      }
    }

    // ---- 线性组合 ----
    var y = model.intercept;
    for (i = 0; i < x.length; i++) {
      y += model.coef[i] * x[i];
    }
    return y;
  }

  root.predictUnitPrice = predictUnitPrice;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { predictUnitPrice: predictUnitPrice };
  }
})(typeof window !== "undefined" ? window : globalThis);
