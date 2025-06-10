#!/usr/bin/env python3
"""
download_gsod_by_label.py  (with not-found list)
-----------------------------------------------
讀 data/interim/labels_isd.csv → 依 isd_usaf+isd_wban+year
下載對應 GSOD 站點 .csv。
若下載失敗，將 year,station_id 寫入 data/raw/gsod_not_found.txt
"""

from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

LABEL_FILE = Path("data/interim/labels_isd.csv")
SAVE_ROOT  = Path("data/raw/gsod")
NOTFOUND   = Path("data/raw/gsod_not_found.txt")
YEARS      = {2015, 2016}

BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station}.csv"

def main():
    df = pd.read_csv(LABEL_FILE, low_memory=False)
    df["year"] = pd.to_datetime(df["time"], errors="coerce").dt.year
    df = df[df["year"].isin(YEARS)]
    df["station_id"] = df["isd_usaf"].astype(str).str.zfill(6) + df["isd_wban"].astype(str).str.zfill(5)

    targets = df[["year", "station_id"]].drop_duplicates()
    print(f"將下載 {len(targets)} 個 station-year 檔案…")

    not_found = []                       # 收集下載失敗

    for year, station in tqdm(targets.itertuples(index=False), total=len(targets)):
        url  = BASE_URL.format(year=year, station=station)
        dest_dir  = SAVE_ROOT / str(year)
        dest_file = dest_dir / f"{station}.csv"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_file.exists():
            continue

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content.strip():
                dest_file.write_bytes(r.content)
            else:
                raise ValueError(f"status={r.status_code}")
        except Exception as e:
            not_found.append(f"{year},{station}")
            print(f"[WARN] 下載失敗 {station} {year}：{e}")

    # 把失敗列表寫檔
    if not_found:
        NOTFOUND.parent.mkdir(parents=True, exist_ok=True)
        NOTFOUND.write_text("\n".join(not_found))
        print(f"⚠️ 下載失敗 {len(not_found)} 筆，已寫入 {NOTFOUND}")
    else:
        if NOTFOUND.exists():
            NOTFOUND.unlink()            # 全成功就刪除舊檔
        print("✅ 所有檔案下載成功")

if __name__ == "__main__":
    main()
