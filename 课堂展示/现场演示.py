"""课堂网页：前端经本机 JSON API 调用实验中的决策树模型。"""
from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web_demo"
PROJECT_SRC = ROOT.parent / "src"
if (PROJECT_SRC / "common.py").is_file():
    # 在原项目中运行时，直接调用实验程序的特征配置和模型管线。
    sys.path.insert(0, str(PROJECT_SRC))
    from common import FEATURES, SEED, TARGET, load_clean, make_models

    def make_demo_model():
        return make_models()["决策树回归"]
else:
    # 只拷贝展示文件夹时，使用同配置的独立快照。
    from demo_model import FEATURES, SEED, TARGET, load_clean, make_demo_model

CASES = {49: "朝阳两居", 69: "西城两居", 709: "房山两居"}
EDITABLE = ("区县", "面积", "室", "房龄", "近地铁", "装修")
RANGES = {"面积": (10, 300), "室": (0, 8), "房龄": (0, 70), "近地铁": (0, 1)}
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def plain(value):
    """将 pandas / numpy 标量转换为合法 JSON 值。"""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return plain(value.item())
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def number(value, key):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字") from exc
    low, high = RANGES[key]
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise ValueError(f"{key} 需要在 {low}～{high} 之间")
    if key in ("室", "近地铁"):
        if not parsed.is_integer():
            raise ValueError(f"{key} 必须是整数")
        return int(parsed)
    return parsed


def prepare():
    """使用与正式实验相同的划分、预处理和决策树参数。"""
    df = load_clean()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED
    )
    if not set(CASES).issubset(x_test.index):
        raise RuntimeError("固定演示样例不在测试集中，请检查数据文件")
    model = make_demo_model().fit(x_train, y_train)
    test_predictions = model.predict(x_test)
    examples = []
    for ident, label in CASES.items():
        original = x_test.loc[ident]
        examples.append({
            "id": ident, "label": label,
            "actual": float(y_test.loc[ident]),
            "prediction": float(model.predict(x_test.loc[[ident]])[0]),
            "editable": {key: plain(original[key]) for key in EDITABLE},
            "all_features": {key: plain(original[key]) for key in FEATURES},
        })
    score_table = pd.read_csv(ROOT / "data" / "model_scores.csv", encoding="utf-8-sig")
    scores = [{
        "model": str(row["模型"]),
        "cv_r2": float(row["交叉验证R2"]),
        "test_r2": float(row["测试集R2"]),
        "mae": float(row["测试集MAE"]),
    } for _, row in score_table.iterrows()]
    return {
        "model": model, "x_test": x_test, "examples": examples,
        "scores": scores,
        "districts": sorted(str(v) for v in df["区县"].dropna().unique()),
        "renovations": sorted(str(v) for v in df["装修"].dropna().unique()),
        "metrics": {
            "rows": len(df), "train": len(x_train), "test": len(x_test),
            "r2": float(r2_score(y_test, test_predictions)),
            "mae": float(mean_absolute_error(y_test, test_predictions)),
        },
    }


def predict(state, request):
    if not isinstance(request, dict):
        raise ValueError("请求必须是 JSON 对象")
    try:
        case = int(request.get("case"))
    except (TypeError, ValueError) as exc:
        raise ValueError("请选择一条演示房源") from exc
    if case not in CASES:
        raise ValueError("房源编号不在固定测试样例中")
    supplied = request.get("features")
    if not isinstance(supplied, dict):
        raise ValueError("请提供房屋属性")

    original = state["x_test"].loc[case]
    changed_row = original.copy()
    for key in ("区县", "装修"):
        allowed = state["districts"] if key == "区县" else state["renovations"]
        value = supplied.get(key, original[key])
        if value not in allowed:
            raise ValueError(f"{key} 不在可选范围内")
        changed_row[key] = value
    for key in RANGES:
        changed_row[key] = number(supplied.get(key, original[key]), key)

    frame = pd.DataFrame([{key: changed_row[key] for key in FEATURES}], columns=FEATURES)
    prediction = float(state["model"].predict(frame)[0])
    example = next(item for item in state["examples"] if item["id"] == case)
    changed = any(changed_row[key] != original[key] for key in EDITABLE)
    return {
        "case": case, "label": CASES[case], "prediction": prediction,
        "original_prediction": example["prediction"],
        "actual": None if changed else example["actual"],
        "error": None if changed else prediction - example["actual"],
        "delta": prediction - example["prediction"], "changed": changed,
        "inputs": {key: plain(changed_row[key]) for key in EDITABLE},
    }


def create_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/meta":
                self.send_json(200, {key: state[key] for key in
                                     ("examples", "districts", "renovations", "metrics", "scores")})
            elif route in FILES:
                filename, content_type = FILES[route]
                self.send_bytes(200, (WEB / filename).read_bytes(), content_type)
            else:
                self.send_json(404, {"error": "页面不存在"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                self.send_json(404, {"error": "接口不存在"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16384:
                    raise ValueError("请求内容为空或过大")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                self.send_json(200, predict(state, request))
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="启动北京二手房单价预测课堂网页")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--port", type=int, default=0, help="端口，默认自动选择空闲端口")
    args = parser.parse_args()
    state = prepare()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), create_handler(state))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"课堂网页已启动：{url}", flush=True)
    print("按 Ctrl+C 结束。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已结束。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
