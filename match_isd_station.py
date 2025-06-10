#!/usr/bin/env python3
"""
match_isd_station.py  (v2)
--------------------------
1. 讀取 data/interim/labels.csv
2. 讀取 data/raw/isd-history.csv
3. 僅保留 BEGIN<=20150101 & END>=20161231 的 ISD 站
4. 以 Haversine 找最近站，附回欄位：
   isd_usaf, isd_ctry, isd_lat, isd_lon, isd_elev, isd_begin, isd_end
5. 輸出 data/interim/labels_isd.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

LABELS_IN  = Path("data/interim/labels.csv")
ISD_IN     = Path("data/raw/isd-history.csv")
LABELS_OUT = Path("data/interim/labels_isd.csv")
EARTH_R = 6371.0  # km


# ---------- 工具函式 ----------
def haversine(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def load_isd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["LAT", "LON"])
    
    # ➊ 篩掉沒有完整覆蓋 2015–2016 的站
    df = df[(df["BEGIN"] <= 20150101) & (df["END"] >= 20161231)]
    
    # ➋ 補零 & 弧度轉換
    df["USAF_STR"] = df["USAF"].astype(str).str.zfill(6)
    df["WBAN_STR"] = df["WBAN"].astype(str).str.zfill(5)
    df["lat_rad"] = np.deg2rad(df["LAT"].astype(float))
    df["lon_rad"] = np.deg2rad(df["LON"].astype(float))
    return df


def main():
    labels = pd.read_csv(LABELS_IN, low_memory=False)
    isd = load_isd(ISD_IN)

    obs_lat = np.deg2rad(labels["st_y"].astype(float).values)
    obs_lon = np.deg2rad(labels["st_x"].astype(float).values)

    # 預留欄位
    nearest = {k: [] for k in [
        "isd_usaf", "isd_wban", "isd_ctry", "isd_lat", "isd_lon",
        "isd_elev", "isd_begin", "isd_end"
    ]}

    for lat, lon in zip(obs_lat, obs_lon):
        d = haversine(lat, lon, isd["lat_rad"].values, isd["lon_rad"].values)
        idx = d.argmin()
        st  = isd.iloc[idx]
        nearest["isd_usaf"].append(st["USAF_STR"])
        nearest["isd_wban"].append(st["WBAN_STR"])
        nearest["isd_ctry"].append(st.get("CTRY", ""))
        nearest["isd_lat"].append(st["LAT"])
        nearest["isd_lon"].append(st["LON"])
        nearest["isd_elev"].append(st["ELEV(M)"])
        nearest["isd_begin"].append(st["BEGIN"])
        nearest["isd_end"].append(st["END"])

    for col, vals in nearest.items():
        labels[col] = vals

    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS_OUT, index=False)
    print(f"✅ 完成：{len(labels):,} 筆 → {LABELS_OUT}")


if __name__ == "__main__":
    main()
