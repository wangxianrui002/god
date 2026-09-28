const $ = (selector) => document.querySelector(selector);
const format = (number) => Math.round(number).toLocaleString("zh-CN");
const signed = (number) => `${number >= 0 ? "+" : "−"}${format(Math.abs(number))}`;
const form = $("#prediction-form");
let meta = null;
let selected = null;
let requestNumber = 0;

function field(name) {
  return form.elements.namedItem(name);
}

function fillSelect(name, values) {
  const select = field(name);
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    select.append(option);
  }
}

function showAllFeatures(example) {
  const list = $("#all-features");
  list.replaceChildren();
  for (const [name, value] of Object.entries(example.all_features)) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = name;
    detail.textContent = value == null ? "缺失" : String(value);
    row.append(term, detail);
    list.append(row);
  }
}

function readFeatures() {
  return {
    "区县": field("区县").value,
    "面积": Number(field("面积").value),
    "室": Number(field("室").value),
    "房龄": Number(field("房龄").value),
    "近地铁": Number(field("近地铁").value),
    "装修": field("装修").value,
  };
}

function setCase(example) {
  selected = example;
  for (const [name, value] of Object.entries(example.editable)) {
    field(name).value = value;
  }
  for (const button of document.querySelectorAll(".case-tab")) {
    button.classList.toggle("active", Number(button.dataset.id) === example.id);
    button.setAttribute("aria-pressed", String(Number(button.dataset.id) === example.id));
  }
  showAllFeatures(example);
  calculate();
}

function renderCaseTabs() {
  const tabs = $("#case-tabs");
  tabs.replaceChildren();
  for (const example of meta.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-tab";
    button.dataset.id = example.id;
    const title = document.createElement("strong");
    title.textContent = example.label;
    const subtitle = document.createElement("small");
    subtitle.textContent = `${example.editable["面积"]} ㎡ · ${example.editable["室"]} 室`;
    button.append(title, subtitle);
    button.addEventListener("click", () => setCase(example));
    tabs.append(button);
  }
}

function renderExamples() {
  const grid = $("#examples-grid");
  grid.replaceChildren();
  for (const example of meta.examples) {
    const card = document.createElement("article");
    card.className = "example-card";
    const heading = document.createElement("h3");
    heading.textContent = example.label;
    card.append(heading);
    const pairs = [
      ["真实单价", format(example.actual)],
      ["模型预测", format(example.prediction)],
      ["预测误差", signed(example.prediction - example.actual)],
    ];
    for (const [index, [label, value]] of pairs.entries()) {
      const row = document.createElement("div");
      row.className = `example-row${index === 2 ? " example-diff" : ""}`;
      const name = document.createElement("span");
      const amount = document.createElement("strong");
      name.textContent = label;
      amount.textContent = value;
      row.append(name, amount);
      card.append(row);
    }
    const track = document.createElement("div");
    track.className = "mini-track";
    const bar = document.createElement("div");
    bar.style.width = `${Math.min(100, example.prediction / 130000 * 100)}%`;
    track.append(bar);
    card.append(track);
    grid.append(card);
  }
}

function renderScores() {
  const container = $("#model-scores");
  container.replaceChildren();
  for (const score of meta.scores) {
    const item = document.createElement("article");
    item.className = `model-score${score.model === "决策树回归" ? " best" : ""}`;
    const title = document.createElement("h3");
    title.textContent = score.model;
    const value = document.createElement("div");
    value.className = "score-value";
    value.textContent = score.test_r2 < 0.001 ? "≈ 0" : score.test_r2.toFixed(3);
    const label = document.createElement("div");
    label.className = "score-label";
    label.textContent = "测试集 R²";
    const track = document.createElement("div");
    track.className = "score-track";
    const bar = document.createElement("div");
    bar.style.width = `${Math.max(0, score.test_r2) / 0.75 * 100}%`;
    track.append(bar);
    const mae = document.createElement("div");
    mae.className = "score-mae";
    const maeLabel = document.createElement("span");
    maeLabel.textContent = "测试 MAE";
    const maeNumber = document.createElement("strong");
    maeNumber.textContent = `${format(score.mae)} 元/㎡`;
    mae.append(maeLabel, maeNumber);
    item.append(title, value, label, track, mae);
    container.append(item);
  }
}

function showResult(result) {
  $("#predicted-price").textContent = format(result.prediction);
  $("#prediction-value").textContent = format(result.prediction);
  const status = $("#result-status");
  status.textContent = result.changed ? "条件已修改" : "真实测试样本";
  status.classList.toggle("modified", result.changed);

  let reference;
  if (result.changed) {
    reference = result.original_prediction;
    $("#reference-label").textContent = "原样本预测";
    $("#reference-value").textContent = format(reference);
    $("#result-explanation").textContent =
      `相对原样本预测变化 ${signed(result.delta)} 元/㎡。修改后的条件没有对应的真实成交价。`;
    $("#result-footnote").textContent = "条件变化只表示模型输出变化，不代表因果影响";
  } else {
    reference = result.actual;
    $("#reference-label").textContent = "真实成交";
    $("#reference-value").textContent = format(reference);
    $("#result-explanation").textContent =
      `这条测试样本的真实单价为 ${format(result.actual)} 元/㎡；预测误差为 ${signed(result.error)} 元/㎡。`;
    $("#result-footnote").textContent = "这条房源来自固定测试集，未参与模型训练";
  }
  const scale = Math.max(reference, result.prediction, 1000) * 1.13;
  $("#reference-bar").style.width = `${reference / scale * 100}%`;
  $("#prediction-bar").style.width = `${result.prediction / scale * 100}%`;
}

async function calculate() {
  if (!selected) return;
  const requestId = ++requestNumber;
  const button = $("#predict-button");
  button.disabled = true;
  $("#result-status").textContent = "模型计算中";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({case: selected.id, features: readFeatures()}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "预测失败");
    if (requestId === requestNumber) showResult(data);
  } catch (error) {
    if (requestId === requestNumber) {
      $("#result-status").textContent = "无法预测";
      $("#result-explanation").textContent = error.message;
    }
  } finally {
    if (requestId === requestNumber) button.disabled = false;
  }
}

async function start() {
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) throw new Error("无法读取演示数据");
    meta = await response.json();
    $("#stat-rows").textContent = format(meta.metrics.rows);
    $("#stat-r2").textContent = meta.metrics.r2.toFixed(3);
    $("#stat-mae").textContent = format(meta.metrics.mae);
    fillSelect("区县", meta.districts);
    fillSelect("装修", meta.renovations);
    renderCaseTabs();
    renderExamples();
    renderScores();
    setCase(meta.examples[0]);
  } catch (error) {
    $("#result-status").textContent = "启动失败";
    $("#result-explanation").textContent = `${error.message}；请确认 Python 演示程序仍在运行。`;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (form.reportValidity()) calculate();
});
$("#reset-button").addEventListener("click", () => {
  if (selected) setCase(selected);
});
start();
