# aurora_train.py  (All‑in‑one, patched)
# -------------------------------------------------------------
# 小資料極光預測流程
# 1. 讀 dataset_station_daily.csv（你已整理好的每日站點特徵）
# 2. 特徵工程 + 物理先驗 (prob_phy)
# 3. Positive‑Unlabeled：
#    • Positive  = see_aurora == 1
#    • Negative* = 夜間 & kp_max<=2 & visib<=2 (高可信 0)
# 4. LightGBM 訓練 (2015 train / 2016 valid)
#    • choose_threshold：取驗證機率 95‑th percentile，且下限 0.5
# 5. 產生 2025‑06‑15 ➜ 2026‑06‑14 每日預測
#    • 缺值以低值 + 訓練中位數補
#    • Hard gate：夜間 & kp_max≥4 & LAT≥55 才進模型，其餘 prob=0
# 6. 輸出
#    • 模型  → output/lgbm_aurora_v2.pkl
#    • 圖    → output/pr_curve_v2.png
#    • 預測  → output/predictions_20250615_20260614.csv
# -------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt

# ---------------- 路徑 ----------------
DATA = Path("data/processed/dataset.csv")
OUT  = Path("output"); OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "lgbm_aurora_v2.pkl"
PRED_PATH  = OUT / "predictions_20250615_20260614.csv"
FIG_PATH   = OUT / "pr_curve_v2.png"

TARGET_PREC = 0.90

BASE_FEATURES = [
    "kp_max", "kp_mean", "Ap", "F107",   # 太陽/地磁
    "temp", "visib", "prcp", "wdsp",      # 天氣
    "LAT", "LON", "ELEV(M)"                # 地理
]

# -------------- Feature 工具 --------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    utc_hour = df["date"].dt.hour
    local_hour = (utc_hour + (df["LON"] / 15.0)).mod(24)
    df["local_hour"] = local_hour
    df["is_night"]   = ((local_hour >= 18) | (local_hour <= 6)).astype(int)

    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    df["cos_doy"] = np.cos(2*np.pi*doy/365.25)

    df["kp_ratio"] = df["kp_max"] / (df["kp_mean"] + 1e-6)
    return df

# choose threshold by percentile

def choose_threshold_by_pct(y_prob: np.ndarray, pct: float = 95, floor: float = 0.5):
    thr = np.percentile(y_prob, pct)
    return max(thr, floor)

# -------------- 主流程 --------------

def main():
    # 1. 讀資料
    df = pd.read_csv(DATA, parse_dates=["date"], low_memory=False)
    df = add_features(df)

    # 物理先驗：sigmoid(kp_max)
    df["prob_phy"] = 1/(1+np.exp(-1.2*(df["kp_max"]-4)))

    # 2. PU 樣本
    pos_df = df[df["see_aurora"] == 1]
    neg_df = df[(df["is_night"]==1) & (df["kp_max"]<=2) & (df["visib"]<=2)]
    train_df = pd.concat([pos_df, neg_df], ignore_index=True).sort_values("date")

    # Train / valid split
    train = train_df[train_df["date"].dt.year == 2015]
    valid = train_df[train_df["date"].dt.year == 2016]

    feat_cols = BASE_FEATURES + ["local_hour","is_night","sin_doy","cos_doy","kp_ratio","prob_phy"]
    X_train, y_train = train[feat_cols], train["see_aurora"].fillna(0).astype(int)
    X_valid, y_valid = valid[feat_cols], valid["see_aurora"].fillna(0).astype(int)

    # 3. LightGBM
    scale_pos_weight = (len(y_train)-y_train.sum())/y_train.sum()
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=8,
        min_gain_to_split=1e-3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight={0:1,1:scale_pos_weight},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.log_evaluation(period=0)])

    y_prob_val = model.predict_proba(X_valid)[:,1]
    thr = choose_threshold_by_pct(y_prob_val, pct=95, floor=0.5)
    P = ((y_prob_val>=thr) & (y_valid==1)).sum() / max((y_prob_val>=thr).sum(),1)
    R = ((y_prob_val>=thr) & (y_valid==1)).sum() / y_valid.sum()
    pr_auc = average_precision_score(y_valid, y_prob_val)

    print(f"Validation  P={P:.3f} R={R:.3f} PR_AUC={pr_auc:.3f} thr={thr:.3f}")

    # PR curve
    prec, rec, _ = precision_recall_curve(y_valid, y_prob_val)
    plt.figure(); plt.plot(rec, prec); plt.axhline(TARGET_PREC, ls="--", c="r")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve v2")
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight"); plt.close()

    # 保存模型
    joblib.dump({"model":model, "threshold":thr, "features":feat_cols}, MODEL_PATH)

    # 4. 未來一年推論
    last_rows = df.sort_values("date").groupby("station_id").tail(1).copy()
    future_dates = pd.date_range("2025-06-15", "2026-06-14", freq="D")
    fut = pd.concat([last_rows.assign(date=d) for d in future_dates], ignore_index=True)
    fut = add_features(fut)

    # 缺值補：物理低值 + 中位
    low_fill = {"kp_max":0, "kp_mean":0, "Ap":0, "F107":X_train["F107"].min()}
    fut[feat_cols] = fut[feat_cols].fillna(low_fill).fillna(X_train.median())

    # Hard gate
    gate = ( (fut["is_night"]==1) & (fut["kp_max"]>=4) & (fut["LAT"]>=55) )
    fut["prob_model"] = 0.0
    fut.loc[gate, "prob_model"] = model.predict_proba(fut.loc[gate, feat_cols])[:,1]
    fut["predict"] = (fut["prob_model"] >= thr).astype(int)

    fut[["date","station_id","LAT","LON","prob_model","predict"]].to_csv(PRED_PATH, index=False)
    print("輸出:\n  模型 →", MODEL_PATH, "\n  圖   →", FIG_PATH, "\n  預測 →", PRED_PATH)

if __name__ == "__main__":
    main()
