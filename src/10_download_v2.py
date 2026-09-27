# -*- coding: utf-8 -*-
"""
10 下载第二版数据 —— Kaggle「中国主要城市二手房挂牌数据」

来源：https://www.kaggle.com/datasets/xiaopaohadoop/second-hand-housing-dataset
原始文件：SH-house-dataset.csv，170 MB，455,566 行 × 22 列，16 个城市。

为什么要换数据（第一版 README §7.4 的结论是「缺的是数据不是算法」）：
第一版用链家**成交**数据，只有面积/房间数/楼层/房龄/装修这些粗粒度属性，
小区品质、朝向、精确区位一概没有，R² 卡在 0.70 上不去。
本数据集是**挂牌**数据，多出朝向、楼层位置、小区名、板块、环线、带看量等字段 ——
正是第一版分析里点名缺失的那几类信息。

产出：
    data/house_v2_raw.csv    北京子集 73,685 行 × 22 列（全量 16 城 45 万行不入库）

脚本幂等：已存在就跳过，要重下加 --force。
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from common_v2 import RAW_V2

ZIP_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
           "xiaopaohadoop/second-hand-housing-dataset")
INNER_NAME = "SH-house-dataset.csv"
CITY = "北京"
EXPECTED_ROWS = 73_685
EXPECTED_COLS = 22
TIMEOUT = 300          # 33 MB 的 zip
RETRIES = 3


def download_zip() -> bytes:
    """下载 zip，失败重试并退避。Kaggle 这个数据集的下载接口不需要认证。"""
    last = None
    for i in range(RETRIES):
        try:
            req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                total = int(r.headers.get("Content-Length") or 0)
                buf, got = io.BytesIO(), 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    buf.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  下载中 {got / 1e6:6.1f} / {total / 1e6:.1f} MB "
                              f"({got / total:5.1%})", end="", flush=True)
            print()
            return buf.getvalue()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            print(f"\n  第 {i + 1}/{RETRIES} 次失败：{e}")
            if i < RETRIES - 1:
                time.sleep(3 * (i + 1))
    raise SystemExit(
        f"\n下载失败：{last}\n"
        f"  可手动下载后放到 {RAW_V2.parent}：\n"
        f"    {ZIP_URL}\n"
        f"  解压出 {INNER_NAME}，只保留 city == '{CITY}' 的行，"
        f"存成 {RAW_V2.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载中国主要城市二手房挂牌数据")
    ap.add_argument("--force", action="store_true", help="已存在时也重新下载")
    args = ap.parse_args()

    print("=" * 68)
    print("10 下载第二版数据（Kaggle 二手房挂牌数据）")
    print("=" * 68)

    if RAW_V2.exists() and not args.force:
        df = pd.read_csv(RAW_V2, low_memory=False)
        print(f"\n已存在，跳过下载：{RAW_V2.name}  {len(df):,} 行 × {df.shape[1]} 列")
        print("  （要强制重新下载，用 --force）")
        return 0

    print(f"\n数据源：{ZIP_URL}")
    blob = download_zip()
    print(f"  压缩包 {len(blob) / 1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = INNER_NAME if INNER_NAME in z.namelist() else z.namelist()[0]
        print(f"  解出 {name} …")
        with z.open(name) as f:
            all_city = pd.read_csv(f, low_memory=False)
    print(f"  全量 {len(all_city):,} 行 × {all_city.shape[1]} 列，"
          f"{all_city['city'].nunique()} 个城市")

    df = all_city[all_city["city"] == CITY].reset_index(drop=True)
    print(f"\n只保留 {CITY}：{len(df):,} 行")

    assert len(df) == EXPECTED_ROWS, (
        f"{CITY} 行数变了：期望 {EXPECTED_ROWS:,}，实际 {len(df):,} —— "
        f"上游数据集可能已更新，需要重新核对 README 里的数字")
    assert df.shape[1] == EXPECTED_COLS, f"列数变了：{df.shape[1]} != {EXPECTED_COLS}"

    df.to_csv(RAW_V2, index=False, encoding="utf-8-sig")
    mb = RAW_V2.stat().st_size / 1e6
    print(f"\n已写入：{RAW_V2.name}  {len(df):,} 行 × {df.shape[1]} 列（{mb:.1f} MB）")
    print(f"  列：{', '.join(df.columns)}")
    print(f"\n  注：全量 16 城 45 万行不入库（170 MB）。只保留 {CITY} 是为了和第一版"
          f"\n  的北京数据对照 —— 换数据带来的提升要能和换模型带来的提升分开算。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
