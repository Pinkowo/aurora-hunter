#!/usr/bin/env python3
"""
aurora_model_v2.py
------------------
• 讀 data/processed/dataset_station_daily.csv
• rule-based filter：夜間 & geomag_lat≥57 & kp_max≥4
• 新增時間週期特徵
• LightGBM 訓練 (2015→train, 2016→valid)
• 畫 PR curve & 自動找 Precision≥TARGET_P threshold
• 保存模型 + threshold + PR PNG
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
from datetime import timezone, timedelta

# ── 參數 ─────────────────────────────────────────
DATA   = Path("data/processed/dataset.csv")
MODEL  = Path("models/lgbm_aurora.pkl")
PR_PNG = Path("figures/pr_curve_v2.png")
THR_TXT= Path("models/best_threshold.txt")
TARGET_P = 0.90       # 目標 Precision
# ────────────────────────────────────────────────

FEATURES_BASE = [
    "kp_max","kp_mean","Ap","F107",
    "temp","visib","prcp","wdsp",
    "LAT","LON","ELEV(M)"
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # local hour (粗略：經度/15 + UTC hour)
    utc_hour = df["date"].dt.hour
    local_hour = (utc_hour + (df["LON"] / 15.0)).mod(24)
    df["local_hour"] = local_hour

    # night flag（18–6）
    df["is_night"] = ((local_hour >= 18) | (local_hour <= 6)).astype(int)

    # day of year 週期
    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2*np.pi*doy/365.25)
    df["cos_doy"] = np.cos(2*np.pi*doy/365.25)

    # kp ratio
    df["kp_ratio"] = df["kp_max"] / (df["kp_mean"] + 1e-6)
    return df

def rule_filter(df: pd.DataFrame) -> pd.DataFrame:
    cond = (
        (df["is_night"] == 1) &
        (df["LAT"] >= 57) &              # 用幾何緯度近似磁緯
        (df["kp_max"] >= 4)
    )
    return df[cond].reset_index(drop=True)

def choose_threshold(y_true, y_prob, target_p):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    mask = p >= target_p
    idx = np.where(mask)[0][0] if mask.any() else np.argmax(p)
    return t[idx], p[idx], r[idx], p, r

def plot_pr(p, r):
    plt.figure()
    plt.plot(r, p)
    plt.axhline(TARGET_P, ls="--", c="red")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (target ≥{TARGET_P})")
    PR_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PR_PNG, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    # ── 讀取 & 基本處理 ──────────────────
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = add_features(df)
    df = rule_filter(df)                 # ⬅ rule-based 提升 Precision

    # ── 切 train / valid ────────────────
    train = df[df["date"].dt.year == 2015]
    valid = df[df["date"].dt.year == 2016]

    X_train = train[FEATURES_BASE + ["local_hour","is_night","sin_doy","cos_doy","kp_ratio"]]
    y_train = train["see_aurora"]
    X_valid = valid[X_train.columns]
    y_valid = valid["see_aurora"]

    # ── LightGBM 訓練 ────────────────────
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight={0:1, 1:scale_pos_weight},
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── PR curve & threshold ─────────────
    y_prob = model.predict_proba(X_valid)[:,1]
    thr, P, R, prec, rec = choose_threshold(y_valid, y_prob, TARGET_P)

    pr_auc = average_precision_score(y_valid, y_prob)
    print(f"\nChosen threshold = {thr:.3f}")
    print(f"Precision = {P:.3f} | Recall = {R:.3f} | PR-AUC = {pr_auc:.3f}")

    plot_pr(prec, rec)

    # ── 保存模型 & threshold ─────────────
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model, "threshold":thr, "features":X_train.columns.tolist()}, MODEL)

    with open(THR_TXT, "w") as f:
        f.write(f"threshold={thr:.5f}\nprecision={P:.5f}\nrecall={R:.5f}\npr_auc={pr_auc:.5f}\n")

    print(f"\n✅ model → {MODEL}")
    print(f"✅ PR curve → {PR_PNG}")
    print(f"✅ threshold txt → {THR_TXT}")

if __name__ == "__main__":
    main()
