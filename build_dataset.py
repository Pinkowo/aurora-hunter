#!/usr/bin/env python3
"""
build_dataset_station_daily.py  (fixed)
---------------------------------------
輸出 data/processed/dataset_station_daily.csv
欄位：
    date, station_id, lat, lon, elev, ctry,
    kp_max, kp_mean, Ap, F107,
    temp, visib, prcp, wdsp,
    see_aurora
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- 路徑 ----------
LABEL_ISD = Path("data/interim/labels_isd.csv")
KP_CSV    = Path("data/interim/kp_index.csv")
GSOD_DIR  = Path("data/raw/gsod")
ISD_SRC   = Path("data/raw/isd-history.csv")
OUT_CSV   = Path("data/processed/dataset.csv")

ALLOWED   = {"CA","US","UK","NO","IC","FI","SW","NL"}
YEARS     = {2015, 2016}
# --------------------------------


def load_kp_daily(path: Path) -> pd.DataFrame:
    kp = pd.read_csv(path)
    kp["date"] = pd.to_datetime(
        kp[["YYYY","MM","DD"]].astype(str).agg("-".join, axis=1)
    ).dt.strftime("%Y-%m-%d")
    kp["kp_max"]  = kp[[f"Kp{i}" for i in range(1, 9)]].max(axis=1)
    kp["kp_mean"] = kp[[f"Kp{i}" for i in range(1, 9)]].mean(axis=1)
    kp.rename(columns={"F10.7obs": "F107"}, inplace=True)
    return kp[["date","kp_max","kp_mean","Ap","F107"]]


def build_station_info(isd_path: Path) -> pd.DataFrame:
    isd = pd.read_csv(isd_path, low_memory=False)
    isd = isd[isd["CTRY"].isin(ALLOWED)]
    isd["station_id"] = (
        isd["USAF"].astype(str).str.zfill(6) + isd["WBAN"].astype(str).str.zfill(5)
    )
    return isd[["station_id","CTRY","LAT","LON","ELEV(M)"]].drop_duplicates("station_id")


def load_gsod_year(year: int) -> pd.DataFrame:
    year_dir = GSOD_DIR / str(year)
    if not year_dir.exists():
        return pd.DataFrame()
    dfs = []
    for file in year_dir.glob("*.csv"):
        sid = file.stem
        df = pd.read_csv(
            file,
            usecols=["DATE","TEMP","VISIB","PRCP","WDSP"],
            low_memory=False,
        )
        df.rename(
            columns={"DATE": "date", "TEMP": "temp", "VISIB": "visib",
                     "PRCP": "prcp", "WDSP": "wdsp"},
            inplace=True,
        )
        df["station_id"] = sid
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    # 1. 讀入各來源
    print("📥 讀取 Kp daily …")
    kp_daily = load_kp_daily(KP_CSV)

    print("📥 讀取站點 info …")
    station_info = build_station_info(ISD_SRC)

    print("📥 讀取 GSOD …")
    gsod_frames = [load_gsod_year(y) for y in YEARS]
    gsod = pd.concat(gsod_frames, ignore_index=True)
    gsod.replace({" ":""}, inplace=True)   # 清理空白字串

    # 2. 標籤處理
    print("📥 讀取 aurora labels …")
    labels = pd.read_csv(LABEL_ISD, low_memory=False)
    labels["date"] = pd.to_datetime(labels["time"]).dt.strftime("%Y-%m-%d")
    labels["station_id"] = (
        labels["isd_usaf"].astype(str).str.zfill(6) +
        labels["isd_wban"].astype(str).str.zfill(5)
    )
    label_tbl = (
        labels.groupby(["station_id","date"], as_index=False)["see_aurora"]
            .max()                # 有 1 就保留 1
    )

    # 3. 合併
    print("🔗 合併 (GSOD + Station + Kp + Label) …")
    df = (
        gsod.merge(station_info, on="station_id", how="left")
            .merge(kp_daily, on="date", how="left")
            .merge(label_tbl, on=["station_id","date"], how="left")
    )

    # 4. see_aurora NaN → 0
    df["see_aurora"] = df["see_aurora"].fillna(0).astype(int)

    # 5. 只留白名單國家
    df = df[df["CTRY"].isin(ALLOWED)]

    # 6. 欄位順序 & 輸出
    out_cols = [
        "date","station_id","LAT","LON","ELEV(M)","CTRY",
        "kp_max","kp_mean","Ap","F107",
        "temp","visib","prcp","wdsp",
        "see_aurora"
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"✅ 完成！原始 GSOD 行數：{len(gsod):,} → 輸出 {len(df):,} 行 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
