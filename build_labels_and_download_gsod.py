#!/usr/bin/env python3
"""
build_labels_and_download_gsod.py
---------------------------------
1. 讀 labels.csv → 為每筆觀測找距離最近且覆蓋 2015–2016 的 ISD 站
2. 檢查/下載該年 GSOD 檔 (若失敗自動找下一近站)
3. 將選定站資訊寫回 labels_isd.csv
   下載檔放 data/raw/gsod/<year>/<station>.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# ---------- 可自行修改 ----------
LABEL_SRC   = Path("data/interim/labels.csv")
ISD_SRC     = Path("data/raw/isd-history.csv")
LABEL_OUT   = Path("data/interim/labels_isd.csv")
GSOD_DIR    = Path("data/raw/gsod")
YEARS_SET   = {2015, 2016}          # 只處理這些年份
BASE_URL    = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station}.csv"
EARTH_R_KM  = 6371.0
# ---------------------------------

def haversine(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))

def load_isd(isd_path: Path) -> pd.DataFrame:
    df = pd.read_csv(isd_path, low_memory=False)
    # 只保留覆蓋 2015–2016 的站
    df = df[(df["BEGIN"] <= 20150101) & (df["END"] >= 20161231)]
    df = df.dropna(subset=["LAT", "LON"])
    df["USAF_STR"] = df["USAF"].astype(str).str.zfill(6)
    df["WBAN_STR"] = df["WBAN"].astype(str).str.zfill(5)
    df["station_id"] = df["USAF_STR"] + df["WBAN_STR"]
    df["lat_rad"] = np.deg2rad(df["LAT"].astype(float))
    df["lon_rad"] = np.deg2rad(df["LON"].astype(float))
    return df[["station_id","CTRY","LAT","LON","ELEV(M)","BEGIN","END","lat_rad","lon_rad"]]

def download_station_year(station_id: str, year: int) -> bool:
    """下載 GSOD 檔，成功回傳 True；404 或失敗回傳 False"""
    dest_dir  = GSOD_DIR / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{station_id}.csv"
    if dest_file.exists():                       # 已下載過
        return True
    url = BASE_URL.format(year=year, station=station_id)
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and r.content.strip():
            dest_file.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

def main():
    print("📥 讀取資料 …")
    labels = pd.read_csv(LABEL_SRC, low_memory=False)
    isd    = load_isd(ISD_SRC)

    # 預先準備 numpy 陣列 (加速距離計算)
    isd_lat = isd["lat_rad"].to_numpy()
    isd_lon = isd["lon_rad"].to_numpy()

    # for 迴圈準備欄位儲存
    out_cols = {k: [] for k in
        ["isd_usaf","isd_wban","isd_ctry","isd_lat","isd_lon","isd_elev","isd_begin","isd_end"]}

    print("🔍 為每筆觀測挑選最近可用站並下載 GSOD …")
    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        lat = np.deg2rad(float(row["st_y"]))
        lon = np.deg2rad(float(row["st_x"]))
        year = pd.to_datetime(row["time"], errors="coerce").year
        if year not in YEARS_SET:
            # 不在關注年份範圍 → 直接寫空欄位
            for k in out_cols: out_cols[k].append(np.nan)
            continue

        # 依距離排序索引
        dists = haversine(lat, lon, isd_lat, isd_lon)
        nearest_idx = np.argsort(dists)

        chosen = None
        for idx in nearest_idx:
            station_id = isd.iloc[idx]["station_id"]
            if download_station_year(station_id, year):
                chosen = isd.iloc[idx]
                break      # 找到可下載的站 → 結束迴圈

        if chosen is not None:
            out_cols["isd_usaf"].append(chosen["station_id"][:6])
            out_cols["isd_wban"].append(chosen["station_id"][6:])
            out_cols["isd_ctry"].append(chosen["CTRY"])
            out_cols["isd_lat"].append(chosen["LAT"])
            out_cols["isd_lon"].append(chosen["LON"])
            out_cols["isd_elev"].append(chosen["ELEV(M)"])
            out_cols["isd_begin"].append(chosen["BEGIN"])
            out_cols["isd_end"].append(chosen["END"])
        else:
            # 沒有任何站可下載
            for k in out_cols: out_cols[k].append(np.nan)

    # 合併欄位並輸出
    for col, vals in out_cols.items():
        labels[col] = vals

    LABEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABEL_OUT, index=False)
    print(f"✅ 已輸出 {len(labels):,} 筆 → {LABEL_OUT}")

if __name__ == "__main__":
    main()
