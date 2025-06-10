#!/usr/bin/env python3
"""
build_labels_and_download_gsod.py  (final, rows really removed)
---------------------------------------------------------------
• 最近站若不在 8 國白名單 ⇒ 直接捨棄該觀測行
• 最近站在白名單但該國所有站 GSOD 下載失敗 ⇒ 仍捨棄該觀測行
最終 labels_isd.csv 只保留成功者，因此行數會比原始少
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

LABEL_SRC = Path("data/interim/labels.csv")
ISD_SRC   = Path("data/raw/isd-history.csv")
LABEL_OUT = Path("data/interim/labels_isd.csv")
GSOD_DIR  = Path("data/raw/gsod")

YEARS_SET = {2015, 2016}
ALLOWED   = {"CA","US","UK","NO","IC","FI","SW","NL"}
URL_FMT   = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{y}/{sid}.csv"
EARTH_R   = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(a))

def load_isd(path: Path):
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["BEGIN"]<=20150101) & (df["END"]>=20161231)]
    df = df.dropna(subset=["LAT","LON"])
    df["station_id"] = df["USAF"].astype(str).str.zfill(6)+df["WBAN"].astype(str).str.zfill(5)
    df["lat_r"] = np.deg2rad(df["LAT"].astype(float))
    df["lon_r"] = np.deg2rad(df["LON"].astype(float))
    return df

def download_gsod(sid:str, year:int)->bool:
    ddir = GSOD_DIR/str(year); ddir.mkdir(parents=True, exist_ok=True)
    f = ddir/f"{sid}.csv"
    if f.exists(): return True
    try:
        r = requests.get(URL_FMT.format(y=year, sid=sid), timeout=20)
        if r.status_code==200 and r.content.strip():
            f.write_bytes(r.content)
            return True
    except Exception: pass
    return False

def main():
    labels = pd.read_csv(LABEL_SRC, low_memory=False)
    total_before = len(labels)
    isd = load_isd(ISD_SRC)
    isd_lat, isd_lon = isd["lat_r"].to_numpy(), isd["lon_r"].to_numpy()

    kept = []
    for _, rec in tqdm(labels.iterrows(), total=total_before):
        year = pd.to_datetime(rec["time"], errors="coerce").year
        if year not in YEARS_SET:
            continue
        lat_r, lon_r = map(np.deg2rad, [float(rec["st_y"]), float(rec["st_x"])])
        dists = haversine(lat_r, lon_r, isd_lat, isd_lon)
        nearest_idx = np.argsort(dists)

        # 最近站若國籍不在白名單 → 直接丟棄
        first = isd.iloc[nearest_idx[0]]
        if first["CTRY"] not in ALLOWED:
            continue

        # 同國家內依距離嘗試下載
        ok_station = None
        for idx in nearest_idx:
            st = isd.iloc[idx]
            if st["CTRY"] != first["CTRY"]:
                break           # 只在同一國家範圍內試
            if download_gsod(st["station_id"], year):
                ok_station = st
                break

        if ok_station is None:
            continue            # 該觀測行最終淘汰

        # ---- 保留成功行 ----
        out = rec.copy()
        out["isd_usaf"]  = ok_station["station_id"][:6]
        out["isd_wban"]  = ok_station["station_id"][6:]
        out["isd_ctry"]  = ok_station["CTRY"]
        out["isd_lat"]   = ok_station["LAT"]
        out["isd_lon"]   = ok_station["LON"]
        out["isd_elev"]  = ok_station["ELEV(M)"]
        out["isd_begin"] = ok_station["BEGIN"]
        out["isd_end"]   = ok_station["END"]
        kept.append(out)

    # ------------ 輸出 ------------
    if kept:
        out_df = pd.DataFrame(kept)
        LABEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(LABEL_OUT, index=False)
        print(f"原始 {total_before:,} 筆 ➜ 保留 {len(out_df):,} 筆，已寫 {LABEL_OUT}")
    else:
        print(f"全部 {total_before:,} 筆皆被濾除，未輸出結果。")

if __name__ == "__main__":
    main()
