# -*- coding: utf-8 -*-
"""
01 数据采集 —— 教程 §3.1「数据采集」

下载真实的北京链家二手房挂牌数据，解码后存为 UTF-8，供后续步骤使用。

数据来源：
    yinghaopeng/Housing-Price-with-the-COVID-19-base-on-Beijing（GitHub）
    原始数据采集自链家网北京二手房频道，仅供学习使用。

注意：源文件编码是 GB18030，不是 UTF-8。这一步统一转成 UTF-8，
后面的脚本就都可以按默认编码读，编码问题只在这一个地方处理。
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
OUT = DATA / "lianjia_bj.csv"

# 同一个文件在 GitHub 上的几个地址，逐个尝试。
SOURCES = [
    "https://raw.githubusercontent.com/yinghaopeng/Housing-Price-with-the-COVID-19-base-on-Beijing/"
    "main/%E9%93%BE%E5%AE%B6%E4%BA%8C%E6%89%8B%E6%88%BF%E7%BB%9F%E8%AE%A1%E6%95%B0%E6%8D%AE.csv",
    "https://raw.githubusercontent.com/yinghaopeng/Housing-Price-with-the-COVID-19-base-on-Beijing/"
    "master/%E9%93%BE%E5%AE%B6%E4%BA%8C%E6%89%8B%E6%88%BF%E7%BB%9F%E8%AE%A1%E6%95%B0%E6%8D%AE.csv",
]

EXPECTED_ROWS = 306
EXPECTED_COLS = 19
SRC_ENCODING = "gb18030"
TIMEOUT = 30
RETRIES = 3


def fetch(url: str) -> bytes:
    """下载并返回原始字节。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def download() -> bytes | None:
    """按 SOURCES 顺序尝试下载，成功即返回字节。"""
    for url in SOURCES:
        name = url.rsplit("/", 1)[-1]
        for attempt in range(1, RETRIES + 1):
            try:
                print(f"  尝试 {attempt}/{RETRIES}: {name}")
                data = fetch(url)
                if len(data) < 10_000:
                    raise ValueError(f"返回内容过小（{len(data)} 字节），可能不是数据文件")
                print(f"  下载成功：{len(data):,} 字节")
                return data
            except Exception as exc:  # noqa: BLE001 - 网络异常种类多，统一重试
                print(f"    失败：{type(exc).__name__}: {exc}")
                if attempt < RETRIES:
                    time.sleep(2 * attempt)
    return None


def verify(text: str) -> list[str]:
    """校验行数/列数，返回列名。行数不符只警告，不中断。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    n_rows = len(lines) - 1
    print(f"  行数（不含表头）：{n_rows}；列数：{len(header)}")
    if n_rows != EXPECTED_ROWS:
        print(f"  [警告] 预期 {EXPECTED_ROWS} 行，实际 {n_rows} 行 —— 数据可能已更新，"
              f"README 里的指标是按 {EXPECTED_ROWS} 行得到的")
    if len(header) != EXPECTED_COLS:
        print(f"  [警告] 预期 {EXPECTED_COLS} 列，实际 {len(header)} 列")
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description="下载北京链家二手房数据")
    ap.add_argument("--force", action="store_true", help="已存在时也重新下载")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if OUT.exists() and not args.force:
        text = OUT.read_text(encoding="utf-8")
        print(f"已存在，跳过下载：{OUT.relative_to(ROOT)}")
        print(f"  大小 {OUT.stat().st_size:,} 字节，行数 {len(text.splitlines()) - 1}")
        print("  （要强制重新下载，用 --force）")
        verify(text)
        return 0

    print("下载中……")
    raw = download()
    if raw is None:
        print("\n[失败] 所有地址都下载不到。GitHub 上的个人仓库链接可能会失效。")
        print("请手动下载后放到下面这个位置，再重新运行本脚本：")
        print(f"  {OUT}")
        print("下载地址（浏览器打开）：")
        print(f"  {SOURCES[0]}")
        return 1

    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode(SRC_ENCODING)
    except UnicodeDecodeError as exc:
        print(f"[失败] 用 {SRC_ENCODING} 解码失败：{exc}")
        return 1

    header = verify(text)
    print(f"  原始字节 SHA-256：{digest}")
    print(f"  列名：{','.join(header)}")

    OUT.write_text(text, encoding="utf-8", newline="")
    print(f"\n已写入：{OUT.relative_to(ROOT)}（{SRC_ENCODING} → UTF-8）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
