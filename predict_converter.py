import pandas as pd
from pathlib import Path

# 設定檔案路徑
file_path = Path("output/predictions_20250615_20260614.csv")

if not file_path.exists():
    print("⚠️ 找不到檔案：", file_path)
else:
    # 讀取 CSV
    df = pd.read_csv(file_path)

    # 將 predict 欄位 0 → False, 1 → True
    df["predict"] = df["predict"].map({0: False, 1: True})

    # 覆寫存檔
    df.to_csv(file_path, index=False)
    print("✅ 已更新 predict 欄位為布林值（False/True）")
