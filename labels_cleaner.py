#!/usr/bin/env python3
"""
build_labels.py
---------------
將 2015 / 2016 的 Aurorasaurus tweets 與 web observations
整理成統一欄位： st_y, st_x, country, time, see_aurora

使用方法：
    python build_labels.py
（預設目錄結構見 README，亦可透過 CLI 參數自訂路徑）
"""

from pathlib import Path
import pandas as pd
import argparse

def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """先用 UTF-8 讀取，失敗就退回 latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1", **kwargs)

def load_tweet_file(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path, low_memory=False)
    df = df[["st_y", "st_x", "location_country", "created_at"]].dropna(subset=["st_y", "st_x"])
    df = df.rename(
        columns={"location_country": "country", "created_at": "time"}
    )
    df["see_aurora"] = 1
    return df


def load_web_file(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path, low_memory=False)
    df = df[["st_y", "st_x", "address_country", "time_start", "see_aurora"]].dropna(subset=["st_y", "st_x"])
    df["see_aurora"] = df["see_aurora"].str.lower().map({"t": 1, "f": 0}).fillna(df["see_aurora"])
    df = df.rename(
        columns={"address_country": "country", "time_start": "time"}
    )
    return df


def main(raw_dir: Path, out_path: Path):
    raw_dir = raw_dir.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 找到四個檔案
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"找不到任何 csv 於 {raw_dir}")

    all_frames = []
    for csv in csv_files:
        if "tweets" in csv.stem:
            all_frames.append(load_tweet_file(csv))
        elif "web_observations" in csv.stem:
            all_frames.append(load_web_file(csv))
        else:
            print(f"[WARN] 未識別檔名，跳過：{csv.name}")

    merged = pd.concat(all_frames, ignore_index=True)

    # 轉換 time 欄位為 ISO8601（若需要）
    merged["time"] = pd.to_datetime(merged["time"], errors="coerce").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    merged.to_csv(out_path, index=False)
    print(f"✅ 已輸出 {len(merged):,} 筆 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified aurora label CSV")
    parser.add_argument(
        "--raw_dir",
        default="data/raw/labels",
        type=Path,
        help="原始四個 csv 所在資料夾（預設 raw/labels）",
    )
    parser.add_argument(
        "--out",
        default="data/interim/labels.csv",
        type=Path,
        help="輸出檔案路徑",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out)
