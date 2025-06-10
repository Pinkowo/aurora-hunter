#!/usr/bin/env python3
"""
build_dataset.py
----------------
1. 讀 data/interim/labels_isd.csv
2. 合併 daily Kp/index → 增加 kp_max / kp_mean / Ap / F10.7obs
3. 依 station_id + 日期 併入 GSOD(TEMP/VISIB/PRCP/WDSP)
4. 輸出 data/processed/dataset.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- 路徑 ----------
LABEL_ISD = Path("data/interim/labels_isd.csv")
KPIN_TXT  = Path("data/interim/kp_index.csv")
GSOD_DIR  = Path("data/raw/gsod")
OUT_CSV   = Path("data/processed/dataset.csv")
# -------------------------

def load_kp(df_path: Path) -> pd.DataFrame:
    """讀取每日 Kp, Ap, F10.7 → 建日期索引"""
    kp = pd.read_csv(df_path)
    kp["date"] = pd.to_datetime(
        kp[["YYYY", "MM", "DD"]].astype(str).agg("-".join, axis=1)
    )
    kp["kp_max"]  = kp[[f"Kp{i}" for i in range(1, 9)]].max(axis=1)
    kp["kp_mean"] = kp[[f"Kp{i}" for i in range(1, 9)]].mean(axis=1)
    kp["date"] = kp["date"].dt.strftime("%Y-%m-%d")
    return kp[["date", "kp_max", "kp_mean", "Ap", "F10.7obs"]]

def load_gsod_row(station: str, date: str) -> dict:
    """讀取單站 GSOD csv 並取特定日期行，若缺值回傳 NaNs."""
    year = date[:4]
    f = GSOD_DIR / year / f"{station}.csv"
    if not f.exists():
        return dict(temp=np.nan, visib=np.nan, prcp=np.nan, wdsp=np.nan)

    g = pd.read_csv(f, low_memory=False)
    row = g[g["DATE"] == date]
    if row.empty:
        return dict(temp=np.nan, visib=np.nan, prcp=np.nan, wdsp=np.nan)

    r = row.iloc[0]
    return dict(
        temp=r["TEMP"].strip() if isinstance(r["TEMP"], str) else r["TEMP"],
        visib=r["VISIB"].strip() if isinstance(r["VISIB"], str) else r["VISIB"],
        prcp=r["PRCP"].strip() if isinstance(r["PRCP"], str) else r["PRCP"],
        wdsp=r["WDSP"].strip() if isinstance(r["WDSP"], str) else r["WDSP"],
    )

def main():
    labels = pd.read_csv(LABEL_ISD, parse_dates=["time"])
    labels["date"] = labels["time"].dt.strftime("%Y-%m-%d")
    labels["station_id"] = labels["isd_usaf"].astype(str).str.zfill(6) + \
                           labels["isd_wban"].astype(str).str.zfill(5)

    # --- 合併 Kp ---
    kp_daily = load_kp(KPIN_TXT)
    merged = labels.merge(kp_daily, how="left", on="date")

    # --- 逐行併入 GSOD ---
    gsod_feats = {"temp": [], "visib": [], "prcp": [], "wdsp": []}
    for _, r in tqdm(merged.iterrows(), total=len(merged), desc="Merging GSOD"):
        feat = load_gsod_row(r["station_id"], r["date"])
        for k, v in feat.items():
            gsod_feats[k].append(v)

    for k, v in gsod_feats.items():
        merged[k] = v

    # -- 精選欄位 & 輸出 --
    out_cols = [
        "time", "station_id", "st_y", "st_x", "country",
        "kp_max", "kp_mean", "Ap", "F10.7obs",
        "temp", "visib", "prcp", "wdsp",
        "see_aurora"
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged[out_cols].to_csv(OUT_CSV, index=False)
    print(f"✅ dataset.csv 產生完成，共 {len(merged):,} 列 → {OUT_CSV}")

if __name__ == "__main__":
    main()
