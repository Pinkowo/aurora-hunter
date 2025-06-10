#!/usr/bin/env python3
"""
kp_index_converter.py  (fix)
--------------------------
讀取 data/raw/kp_index.txt（Kp_ap_Ap_SN_F107_since_1932.txt 類）
→ 只保留 2015、2016 → 逗號分隔輸出 CSV
"""

from pathlib import Path
import pandas as pd

RAW_TXT = Path("data/raw/kp_index.txt")
OUT_CSV = Path("data/interim/kp_index.csv")

def main():
    # 1) 讀 txt：skip 40 行註解；以「連續空白」為分隔符
    df = pd.read_csv(
        RAW_TXT,
        delim_whitespace=True,
        skiprows=40,
        header=None,
        names=[
            "YYYY","MM","DD","days","days_m","Bsr","dB",
            "Kp1","Kp2","Kp3","Kp4","Kp5","Kp6","Kp7","Kp8",
            "ap1","ap2","ap3","ap4","ap5","ap6","ap7","ap8",
            "Ap","SN","F10.7obs","F10.7adj","D"
        ],
    )

    # 2) 篩選年份
    df = df[df["YYYY"].isin([2015, 2016])]

    # 3) 輸出
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ 已輸出 {len(df):,} 行 → {OUT_CSV}")

if __name__ == "__main__":
    main()
