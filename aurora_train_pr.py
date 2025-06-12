#!/usr/bin/env python3
"""
aurora_train_pr.py
------------------
• 讀 processed/dataset_station_daily.csv
• 2015 -> train, 2016 -> valid
• XGBoost  + PR curve
• 找 Precision ≥ TARGET_P 的最小 threshold
• 儲存模型 & 圖
"""

from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# ----- 參數 -----
DATA_PATH  = Path("data/processed/dataset.csv")
MODEL_DIR  = Path("models")
FIG_DIR    = Path("figures")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_P   = 0.90            # 目標 Precision 下限
RANDOM_SEED = 42

FEATURES = [
    "kp_max","kp_mean","Ap","F107",
    "temp","visib","prcp","wdsp",
    "LAT","LON","ELEV(M)"
]
# ----------------

def load_split():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    train = df[df["date"].dt.year == 2015]
    valid = df[df["date"].dt.year == 2016]

    X_tr, y_tr = train[FEATURES], train["see_aurora"]
    X_va, y_va = valid[FEATURES], valid["see_aurora"]
    return X_tr, y_tr, X_va, y_va

def train_xgb(X_tr, y_tr):
    pos_w = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_w,
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return clf

def choose_threshold(y_true, y_prob, target_p):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve 不返回最後一個 threshold，因此加一個 1.0
    thr = np.append(thr, 1.0)
    mask = prec >= target_p
    if not mask.any():
        # 若達不到目標 precision，取 precision 最大點
        idx = np.argmax(prec)
    else:
        idx = np.where(mask)[0][0]     # 第一個達標的最小 threshold
    return thr[idx], prec[idx], rec[idx], prec, rec, thr

def plot_pr(prec, rec, target_p, out_path):
    plt.figure()
    plt.plot(rec, prec)
    plt.axhline(target_p, ls="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (target P≥{target_p})")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("📥 Loading & splitting …")
    X_tr, y_tr, X_va, y_va = load_split()

    print("🚂 Training XGBoost …")
    model = train_xgb(X_tr, y_tr)

    print("🔎 Evaluating on 2016 …")
    y_prob = model.predict_proba(X_va)[:,1]
    thr, p_at_thr, r_at_thr, prec, rec, thr_all = choose_threshold(y_va, y_prob, TARGET_P)

    pr_auc = average_precision_score(y_va, y_prob)

    print(f"\nSelected threshold = {thr:.3f}")
    print(f"Precision = {p_at_thr:.3f} | Recall = {r_at_thr:.3f} | PR-AUC = {pr_auc:.3f}")

    # 圖 & 模型輸出
    fig_path   = FIG_DIR / "pr_curve.png"
    model_path = MODEL_DIR / "xgb_aurora.pkl"
    plot_pr(prec, rec, TARGET_P, fig_path)
    joblib.dump({"model": model, "threshold": thr, "features": FEATURES}, model_path)

    print(f"\n✅ Saved PR curve  → {fig_path}")
    print(f"✅ Saved model     → {model_path}")

if __name__ == "__main__":
    main()
