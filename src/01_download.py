# -*- coding: utf-8 -*-
"""
01 数据采集 —— 教程 §3.1「数据采集」

下载真实的北京链家**成交**数据（2011–2017 年成交记录），解码后存为 UTF-8。

数据来源：
    sni13/HousingPrice_Beijing（GitHub）
    原始数据采集自链家网北京成交频道 https://bj.lianjia.com/chengjiao
    仅供学习使用。

关于编码：源文件是 GB18030，不是 UTF-8（用 utf-8 解码在第 0 字节就失败）。
这一步统一转成 UTF-8，后面的脚本就都可以按默认编码读，
编码问题只在这一个地方处理。

关于国内网络：raw.githubusercontent.com 经常连不上或极慢（实测 20 KB/s）。
SOURCES 里的镜像前缀实测 8 MB/s，58 MB 的文件几秒钟就能下完。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "lianjia_bj_raw.csv"

_REPO = "sni13/HousingPrice_Beijing"
_BRANCH = "main"
_FILE = "housing_price_data_lianjia_2011_2017.csv"
_PATH = f"{_REPO}/{_BRANCH}/{_FILE}"

# 同一个文件的不同入口，逐个尝试。带镜像前缀的放前面 —— 国内网络下它们快得多。
SOURCES = [
    f"https://ghfast.top/https://raw.githubusercontent.com/{_PATH}",
    f"https://raw.githubusercontent.com/{_PATH}",
    f"https://gh-proxy.com/https://raw.githubusercontent.com/{_PATH}",
]

EXPECTED_ROWS = 318_851
EXPECTED_COLS = 26
SRC_ENCODING = "gb18030"
TIMEOUT = 180          # 58 MB，给足时间
RETRIES = 2
MIN_BYTES = 10_000_000  # 这个文件实打实 58 MB，下到几十 KB 肯定是错的


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def download() -> bytes | None:
    for url in SOURCES:
        host = url.split("/")[2]
        for attempt in range(1, RETRIES + 1):
            try:
                t0 = time.time()
                print(f"  尝试 {attempt}/{RETRIES}（{host}）……", end="", flush=True)
                data = fetch(url)
                dt = time.time() - t0
                if len(data) < MIN_BYTES:
                    raise ValueError(
                        f"只拿到 {len(data):,} 字节，远小于预期的 ~58 MB，"
                        "多半是错误页或代理限流")
                print(f" {len(data):,} 字节，{dt:.1f} 秒（{len(data)/dt/1e6:.1f} MB/s）")
                return data
            except Exception as exc:  # noqa: BLE001 - 网络异常种类多，统一重试
                print(f" 失败：{type(exc).__name__}: {exc}")
                if attempt < RETRIES:
                    time.sleep(2 * attempt)
    return None


def verify(text: str) -> list[str]:
    """校验行数/列数，返回列名。行数不符只警告，不中断。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    n_rows = len(lines) - 1
    print(f"  行数（不含表头）：{n_rows:,}；列数：{len(header)}")
    if n_rows != EXPECTED_ROWS:
        print(f"  [警告] 预期 {EXPECTED_ROWS:,} 行，实际 {n_rows:,} 行 —— 数据可能已更新，"
              f"README 里的指标是按 {EXPECTED_ROWS:,} 行得到的")
    if len(header) != EXPECTED_COLS:
        print(f"  [警告] 预期 {EXPECTED_COLS} 列，实际 {len(header)} 列")
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description="下载北京链家二手房成交数据")
    ap.add_argument("--force", action="store_true", help="已存在时也重新下载")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if OUT.exists() and not args.force:
        text = OUT.read_text(encoding="utf-8")
        print(f"已存在，跳过下载：{OUT.relative_to(ROOT)}")
        print(f"  大小 {OUT.stat().st_size / 1e6:.1f} MB，行数 {len(text.splitlines()) - 1:,}")
        print("  （要强制重新下载，用 --force）")
        verify(text)
        return 0

    print("下载中（文件约 58 MB，国内镜像一般几秒到几十秒）……")
    raw = download()
    if raw is None:
        print("\n[失败] 所有地址都下载不到。")
        print("请手动下载后放到下面这个位置，再重新运行本脚本：")
        print(f"  {OUT}")
        print("下载地址（浏览器打开）：")
        for u in SOURCES:
            print(f"  {u}")
        return 1

    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode(SRC_ENCODING)
    except UnicodeDecodeError as exc:
        print(f"[失败] 用 {SRC_ENCODING} 解码失败：{exc}")
        return 1

    header = verify(text)
    print(f"  原始字节 SHA-256：{digest[:16]}…（完整值见下）")
    print(f"  {digest}")
    print(f"  列名：{','.join(header)}")

    OUT.write_text(text, encoding="utf-8", newline="")
    print(f"\n已写入：{OUT.relative_to(ROOT)}"
          f"（{SRC_ENCODING} → UTF-8，{OUT.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
