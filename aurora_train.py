# aurora_predictor_v2.py (patched)
# -------------------------------------------------------------
# 更新: 移除 LightGBM fit 的 `verbose` 參數，改用 callback 關閉日誌，
# 以解決舊版 lightgbm 不接受 `verbose` 的 TypeError。
# -------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt

DATA = Path("data/processed/dataset.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "lgbm_aurora_v2.pkl"
PRED_PATH = OUT_DIR / "predictions_20250615_20260614.csv"
FIG_PATH = OUT_DIR / "pr_curve_v2.png"
TARGET_PRECISION = 0.90

BASE_FEATURES = [
    "kp_max", "kp_mean", "Ap", "F107",
    "temp", "visib", "prcp", "wdsp",
    "LAT", "LON", "ELEV(M)"
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    utc_hour = df["date"].dt.hour
    local_hour = (utc_hour + (df["LON"] / 15.0)).mod(24)
    df["local_hour"] = local_hour
    df["is_night"] = ((local_hour >= 18) | (local_hour <= 6)).astype(int)
    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["kp_ratio"] = df["kp_max"] / (df["kp_mean"] + 1e-6)
    return df


def choose_threshold(y_true, y_prob, target_p):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.append(thr, 1.0)
    idx = np.where(prec >= target_p)[0][0] if (prec >= target_p).any() else np.argmax(prec)
    return thr[idx], prec[idx], rec[idx], prec, rec


def main():
    df = pd.read_csv(DATA, parse_dates=["date"], low_memory=False)
    df = add_features(df)
    df["prob_phy"] = 1 / (1 + np.exp(-1.2 * (df["kp_max"] - 4)))

    pos = df[df["see_aurora"] == 1]
    neg = df[(df["is_night"] == 1) & (df["kp_max"] <= 2) & (df["visib"] <= 2)]
    train_df = pd.concat([pos, neg])

    train_df.sort_values("date", inplace=True)
    train = train_df[train_df["date"].dt.year == 2015]
    valid = train_df[train_df["date"].dt.year == 2016]

    feat_cols = BASE_FEATURES + ["local_hour", "is_night", "sin_doy", "cos_doy", "kp_ratio", "prob_phy"]
    X_train, y_train = train[feat_cols], train["see_aurora"].fillna(0).astype(int)
    X_valid, y_valid = valid[feat_cols], valid["see_aurora"].fillna(0).astype(int)

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        class_weight={0: 1, 1: scale_pos_weight},
        random_state=42,
        n_jobs=-1,
    )

    # 使用 callback 取消評估輸出，兼容舊版 LightGBM
    cb = lgb.log_evaluation(period=0)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[cb])

    y_prob = model.predict_proba(X_valid)[:, 1]
    thr, P, R, prec, rec = choose_threshold(y_valid, y_prob, TARGET_PRECISION)
    pr_auc = average_precision_score(y_valid, y_prob)
    print(f"Precision={P:.3f} Recall={R:.3f} AUC={pr_auc:.3f} thr={thr:.3f}")

    plt.figure(); plt.plot(rec, prec); plt.axhline(TARGET_PRECISION, ls="--", color="r")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve v2")
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight"); plt.close()

    joblib.dump({"model": model, "threshold": thr, "features": feat_cols}, MODEL_PATH)
    print("模型與圖已輸出")

    # ---- 產生未來一年推論 (簡化：複製最後一筆) ----
    last_rows = df.sort_values("date").groupby("station_id").tail(1).copy()
    future_dates = pd.date_range("2025-06-15", "2026-06-14", freq="D")
    fut_frames = [last_rows.assign(date=d) for d in future_dates]
    fut = pd.concat(fut_frames, ignore_index=True)
    fut = add_features(fut)
    fut["prob_model"] = model.predict_proba(fut[feat_cols])[:, 1]
    fut["predict"] = (fut["prob_model"] >= thr).astype(int)
    fut[["date", "station_id", "LAT", "LON", "prob_model", "predict"]].to_csv(PRED_PATH, index=False)
    print(f"未來一年預測輸出 -> {PRED_PATH}")

if __name__ == "__main__":
    main()
