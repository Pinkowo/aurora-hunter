#!/usr/bin/env python3
# aurora_train.py  ― 2024-06-15 ➜ 2025-06-14 & country lookup
# ----------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib, matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------- 路徑 ----------
DATA = Path("data/processed/dataset.csv")
ISD  = Path("data/raw/isd-history.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True, parents=True)
# -------------------------

TARGET_P = .90         # Precision 下限
SEED     = 42

# ----------------- 讀原始資料 -----------------
df = pd.read_csv(DATA, parse_dates=["date"])
# ----------- 站點對應 country ---------------
isd = pd.read_csv(ISD, dtype=str, low_memory=False)
isd["station_id"] = isd["USAF"].str.zfill(6) + isd["WBAN"].str.zfill(5)
station2ctry = isd.set_index("station_id")["CTRY"].to_dict()
df["station_id"] = df["station_id"].astype(str).str.zfill(11)
df["country"] = df["station_id"].map(station2ctry).fillna("NA")

# ------------- 基本特徵（示例，同原有） -------------
feat_cols = [
    "kp_max","kp_mean","Ap","F107",
    "temp","visib","prcp","wdsp",
    "LAT","LON","ELEV(M)"
]
# ----------- train / valid 切分 ---------------
train = df[df["date"].dt.year==2015]
valid = df[df["date"].dt.year==2016]

X_tr, y_tr = train[feat_cols], train["see_aurora"]
X_va, y_va = valid[feat_cols], valid["see_aurora"]

# ----------- LightGBM 訓練 --------------------
scale_pos = (len(y_tr)-y_tr.sum()) / y_tr.sum()
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    subsample=.8,
    colsample_bytree=.8,
    class_weight={0:1, 1:scale_pos},
    random_state=SEED,
    verbose=-1,
)
model.fit(X_tr, y_tr)

# ----------- PR curve & threshold ------------
prob_va = model.predict_proba(X_va)[:,1]
prec, rec, thr = precision_recall_curve(y_va, prob_va)
thr = np.append(thr, 1.0)
# 找第一個 precision ≥ TARGET_P 的 threshold，若無則取 0.5
mask = prec >= TARGET_P
best_thr = thr[np.where(mask)[0][0]] if mask.any() else 0.5

print(f"Threshold = {best_thr:.3f}  (Precision target {TARGET_P})")
print(f"Valid  Precision={prec[mask][0] if mask.any() else prec.max():.3f}")

# ----------- 未來一年資料框 -------------------
start = datetime(2024,6,15)
end   = datetime(2025,6,14)
days  = (end - start).days + 1
future_dates = [start + timedelta(d) for d in range(days)]

# 用 2024-06-14 最近一次的站點靜態資訊複製
latest = df.groupby("station_id").last().reset_index()
rows=[]
for d in future_dates:
    tmp = latest.copy()
    tmp["date"] = d
    rows.append(tmp)
future = pd.concat(rows, ignore_index=True)

# 填缺值：數值 -> 訓練中位數；類別 -> "NA"
future[feat_cols] = future[feat_cols].fillna(X_tr.median())
future["country"] = future["country"].fillna("NA")

# ---------- 推論 + predict -------------------
future["prob_model"] = model.predict_proba(future[feat_cols])[:,1]
future["predict"] = (future["prob_model"] >= best_thr).astype(int)

# ------------ 輸出 CSV -----------------------
out_csv = OUT_DIR / "predictions_20240615_20250614.csv"
cols_out = ["date","station_id","country","LAT","LON",
            "prob_model","predict"]
future[cols_out].to_csv(out_csv, index=False)
print("✅  saved to", out_csv)

# ------------ 保存模型 & PR 圖 ---------------
joblib.dump({"model":model,"thr":best_thr,"features":feat_cols}, OUT_DIR/"lgbm_aurora.pkl")

plt.figure()
plt.plot(rec, prec); plt.axhline(TARGET_P, ls="--", c="r")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve v2")
plt.savefig(OUT_DIR/"pr_curve_v2.png", dpi=250, bbox_inches="tight"); plt.close()
