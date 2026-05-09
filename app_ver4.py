import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# ═══════════════════════════════════════════════════════
# 1. LOAD & CLEAN HDFC DATA
# ═══════════════════════════════════════════════════════
file_path = "rawdata/HDFCBANK_2015_2026_merged.csv"
df = pd.read_csv(file_path, parse_dates=["DATE"])
df = df.sort_values("DATE").set_index("DATE")

cols_to_convert = ["OPEN", "HIGH", "LOW", "PREV. CLOSE",
                   "LTP", "CLOSE", "VWAP", "52W H", "52W L", "VALUE"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", ""), errors="coerce"
    )
df.drop(columns=["SERIES"], inplace=True, errors="ignore")

# ═══════════════════════════════════════════════════════
# 2. LOAD & PREPARE INDEX DATA
# ═══════════════════════════════════════════════════════
nifty     = pd.read_csv("rawdata/NIFTY50_2015_2026_merged.csv",   parse_dates=["Date"])
banknifty = pd.read_csv("rawdata/NIFTYBANK_2015_2026_merged.csv", parse_dates=["Date"])

for idx_df in [nifty, banknifty]:
    idx_df.sort_values("Date", inplace=True)
    idx_df.set_index("Date", inplace=True)
    idx_df["Close"] = pd.to_numeric(
        idx_df["Close"].astype(str).str.replace(",", ""), errors="coerce"
    )

nifty["nifty_return"]          = np.log(nifty["Close"] / nifty["Close"].shift(1))
nifty["nifty_vol_10"]          = nifty["nifty_return"].rolling(10).std()
nifty["nifty_ma_ratio"]        = nifty["Close"] / nifty["Close"].rolling(20).mean()

banknifty["banknifty_return"]  = np.log(banknifty["Close"] / banknifty["Close"].shift(1))
banknifty["banknifty_vol_10"]  = banknifty["banknifty_return"].rolling(10).std()
banknifty["banknifty_ma_ratio"]= banknifty["Close"] / banknifty["Close"].rolling(20).mean()

df = df.merge(
    nifty[["nifty_return", "nifty_vol_10", "nifty_ma_ratio"]],
    left_index=True, right_index=True, how="inner"
)
df = df.merge(
    banknifty[["banknifty_return", "banknifty_vol_10", "banknifty_ma_ratio"]],
    left_index=True, right_index=True, how="inner"
)

# ═══════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

# Returns
df["log_return"]     = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
df["return_3d"]      = df["CLOSE"].pct_change(3)
df["return_5d"]      = df["CLOSE"].pct_change(5)
df["return_10d"]     = df["CLOSE"].pct_change(10)
df["return_21d"]     = df["CLOSE"].pct_change(21)

# MA ratios
df["ma5_ratio"]      = df["CLOSE"] / df["CLOSE"].rolling(5).mean()
df["ma20_ratio"]     = df["CLOSE"] / df["CLOSE"].rolling(20).mean()
df["ma_cross"]       = df["CLOSE"].rolling(5).mean() / df["CLOSE"].rolling(20).mean()

# Volatility
df["vol_5"]          = df["log_return"].rolling(5).std()
df["vol_20"]         = df["log_return"].rolling(20).std()
df["vol_ratio"]      = df["vol_5"] / (df["vol_20"] + 1e-9)

# RSI
delta = df["CLOSE"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"]            = 100 - (100 / (1 + gain / loss))
df["rsi_signal"]     = df["rsi"] - 50

# MACD histogram
exp1 = df["CLOSE"].ewm(span=12, adjust=False).mean()
exp2 = df["CLOSE"].ewm(span=26, adjust=False).mean()
macd_line            = exp1 - exp2
df["macd_hist"]      = macd_line - macd_line.ewm(span=9, adjust=False).mean()

# Bollinger Band position
bb_mid               = df["CLOSE"].rolling(20).mean()
bb_std               = df["CLOSE"].rolling(20).std()
df["bb_position"]    = (df["CLOSE"] - bb_mid) / (2 * bb_std + 1e-9)

# Intraday features
df["price_position"] = (df["CLOSE"] - df["LOW"])   / (df["HIGH"] - df["LOW"] + 1e-9)
df["hl_range_ratio"] = (df["HIGH"]  - df["LOW"])   /  df["CLOSE"]
df["oc_range_ratio"] = (df["OPEN"]  - df["CLOSE"]) /  df["CLOSE"]

# VWAP & 52W
df["vwap_ratio"]     = df["CLOSE"] / df["VWAP"]
df["position_52w"]   = (df["CLOSE"] - df["52W L"]) / (df["52W H"] - df["52W L"] + 1e-9)

# Volume
df["volume_ratio"]   = df["VOLUME"] / (df["VOLUME"].rolling(20).mean() + 1e-9)
df["volume_trend"]   = df["VOLUME"].pct_change(5)

# Relative strength vs indices
df["hdfc_vs_nifty"]     = df["log_return"] - df["nifty_return"]
df["hdfc_vs_banknifty"] = df["log_return"] - df["banknifty_return"]

# Calendar
df["day_of_week"]    = df.index.dayofweek
df["month"]          = df.index.month

# Interaction features
df["rsi_x_volume"]   = df["rsi_signal"] * df["volume_ratio"]
df["trend_x_vol"]    = df["ma_cross"]   * df["vol_ratio"]
df["momentum_x_pos"] = df["return_5d"]  * df["price_position"]
df["vwap_x_bb"]      = df["vwap_ratio"] * df["bb_position"]

# ═══════════════════════════════════════════════════════
# 4. TARGET — 3-day forward return, noise filtered
# ═══════════════════════════════════════════════════════
fwd_return_3d = df["CLOSE"].shift(-3) / df["CLOSE"] - 1
THRESHOLD     = 0.005  # 0.5% — skip ambiguous days

df["target"] = np.where(
    fwd_return_3d >  THRESHOLD,  1,
    np.where(fwd_return_3d < -THRESHOLD, 0, np.nan)
)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

print(f"\nClass balance after noise filter:")
print(df["target"].value_counts())
print(f"Trading days kept: {len(df)}")

# ═══════════════════════════════════════════════════════
# 5. FEATURE LIST
# ═══════════════════════════════════════════════════════
features = [
    "log_return", "return_5d", "return_21d",
    "ma_cross", "macd_hist",
    "rsi_signal", "bb_position",
    "vol_ratio",
    "price_position", "vwap_ratio",
    "hdfc_vs_nifty", "hdfc_vs_banknifty",
    "volume_ratio",
    "rsi_x_volume", "trend_x_vol", "momentum_x_pos",
]

df = df.dropna(subset=features)

X = df[features]
y = df["target"]

neg, pos    = (y == 0).sum(), (y == 1).sum()
scale_pos   = neg / pos
print(f"Class imbalance ratio (neg/pos): {scale_pos:.3f}\n")

# ═══════════════════════════════════════════════════════
# 6. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════
TRAIN_WINDOW = 504
TEST_WINDOW  = 63
n_windows    = (len(df) - TRAIN_WINDOW) // TEST_WINDOW

print(f"{'═'*60}")
print(f"Walk-Forward Validation  ({n_windows} windows × {TEST_WINDOW} days)")
print(f"{'═'*60}")

wf_results = {"lr": [], "rf": [], "xgb": []}

for i in range(n_windows):
    start     = i * TEST_WINDOW
    end_train = start + TRAIN_WINDOW
    end_test  = end_train + TEST_WINDOW
    if end_test > len(df):
        break

    X_tr, y_tr = X.iloc[start:end_train], y.iloc[start:end_train]
    X_te, y_te = X.iloc[end_train:end_test], y.iloc[end_train:end_test]

    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        continue

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
    lr.fit(X_tr_sc, y_tr)
    wf_results["lr"].append(accuracy_score(y_te, lr.predict(X_te_sc)))

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    wf_results["rf"].append(accuracy_score(y_te, rf.predict(X_te)))

    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=scale_pos,
        eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    wf_results["xgb"].append(accuracy_score(y_te, xgb.predict(X_te)))

for name, scores in wf_results.items():
    if scores:
        print(f"{name.upper():5s} → Mean: {np.mean(scores):.4f}  "
              f"Std: {np.std(scores):.4f}  "
              f"Min: {np.min(scores):.4f}  "
              f"Max: {np.max(scores):.4f}")

# ═══════════════════════════════════════════════════════
# 7. FINAL HOLDOUT — train on 80%, evaluate on last 20%
# ═══════════════════════════════════════════════════════
split       = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler_f    = StandardScaler()
X_train_sc  = scaler_f.fit_transform(X_train)
X_test_sc   = scaler_f.transform(X_test)

print(f"\n{'═'*60}")
print(f"Final Holdout Results  (last 20% = {len(X_test)} filtered days)")
print(f"{'═'*60}")

# Logistic Regression
lr_f = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
lr_f.fit(X_train_sc, y_train)
y_pred_lr = lr_f.predict(X_test_sc)
print(f"\nLogistic Regression: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf_f = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_f.fit(X_train, y_train)
y_pred_rf = rf_f.predict(X_test)
print(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# XGBoost
xgb_f = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=scale_pos,
    eval_metric="logloss", verbosity=0
)
xgb_f.fit(X_train, y_train)
y_pred_xgb = xgb_f.predict(X_test)
print(f"XGBoost: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# ═══════════════════════════════════════════════════════
# 8. CONFIDENCE THRESHOLD — XGBoost
# ═══════════════════════════════════════════════════════
def threshold_table(model_name, proba, preds, y_true, thresholds):
    print(f"\n{'═'*60}")
    print(f"Confidence Threshold Analysis  ({model_name})")
    print(f"─ Only trade when model confidence ≥ threshold")
    print(f"{'─'*60}")
    print(f"{'Threshold':>10} │ {'Accuracy':>10} │ {'Days':>8} │ {'Coverage':>10}")
    print(f"{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}")
    confidence = proba.max(axis=1)
    for t in thresholds:
        mask = confidence >= t
        n    = mask.sum()
        if n < 10:
            print(f"{t:>10.2f} │ {'—':>10} │ {n:>8} │  too few")
            continue
        acc      = accuracy_score(y_true.values[mask], preds[mask])
        coverage = mask.mean() * 100
        print(f"{t:>10.2f} │ {acc:>10.4f} │ {n:>8} │ {coverage:>8.1f}%")
    return confidence

thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]

xgb_proba = xgb_f.predict_proba(X_test)
xgb_conf  = threshold_table("XGBoost", xgb_proba, y_pred_xgb, y_test, thresholds)

rf_proba  = rf_f.predict_proba(X_test)
rf_conf   = threshold_table("Random Forest", rf_proba, y_pred_rf, y_test, thresholds)

# ═══════════════════════════════════════════════════════
# 9. AGREEMENT ENSEMBLE
#    Trade only when RF + XGBoost agree AND both confident
#    This is where the real alpha lives
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("Agreement Ensemble  (RF + XGBoost must agree on direction)")
print("─ Highest accuracy subset — use this for actual signals")
print(f"{'─'*60}")
print(f"{'Conf ≥':>8} │ {'Accuracy':>10} │ {'Days':>8} │ {'Coverage':>10} │ {'Up signals':>12} │ {'Down signals':>13}")
print(f"{'─'*8}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*13}")

agreement_mask = (y_pred_xgb == y_pred_rf)

for t in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]:
    conf_mask  = (xgb_conf >= t) & (rf_conf >= t)
    final_mask = agreement_mask & conf_mask
    n          = final_mask.sum()
    if n < 10:
        print(f"{t:>8.2f} │ {'—':>10} │ {n:>8} │  too few")
        continue
    preds_agreed = y_pred_xgb[final_mask]
    acc          = accuracy_score(y_test.values[final_mask], preds_agreed)
    coverage     = final_mask.mean() * 100
    up_signals   = (preds_agreed == 1).sum()
    down_signals = (preds_agreed == 0).sum()
    print(f"{t:>8.2f} │ {acc:>10.4f} │ {n:>8} │ {coverage:>8.1f}%"
          f" │ {up_signals:>12} │ {down_signals:>13}")

# ═══════════════════════════════════════════════════════
# 10. ENSEMBLE SIGNAL BREAKDOWN AT BEST THRESHOLD
#     Shows what the ensemble would have traded
# ═══════════════════════════════════════════════════════
BEST_THRESHOLD = 0.58  # adjust based on your threshold table output

conf_mask  = (xgb_conf >= BEST_THRESHOLD) & (rf_conf >= BEST_THRESHOLD)
final_mask = agreement_mask & conf_mask

print(f"\n{'═'*60}")
print(f"Signal Breakdown at Confidence ≥ {BEST_THRESHOLD}")
print(f"{'═'*60}")

if final_mask.sum() >= 10:
    signal_df = pd.DataFrame({
        "date":       X_test.index[final_mask],
        "signal":     np.where(y_pred_xgb[final_mask] == 1, "UP", "DOWN"),
        "xgb_conf":   (xgb_proba.max(axis=1))[final_mask].round(4),
        "rf_conf":    (rf_proba.max(axis=1))[final_mask].round(4),
        "actual":     np.where(y_test.values[final_mask] == 1, "UP", "DOWN"),
        "correct":    (y_pred_xgb[final_mask] == y_test.values[final_mask]),
    })
    print(signal_df.to_string(index=False))

    correct_up   = signal_df[(signal_df["signal"] == "UP")   & signal_df["correct"]].shape[0]
    total_up     = signal_df[signal_df["signal"] == "UP"].shape[0]
    correct_down = signal_df[(signal_df["signal"] == "DOWN") & signal_df["correct"]].shape[0]
    total_down   = signal_df[signal_df["signal"] == "DOWN"].shape[0]

    print(f"\nUP   signal accuracy: {correct_up}/{total_up} "
          f"= {correct_up/total_up:.1%}" if total_up else "\nNo UP signals")
    print(f"DOWN signal accuracy: {correct_down}/{total_down} "
          f"= {correct_down/total_down:.1%}" if total_down else "No DOWN signals")
else:
    print("Too few signals at this threshold — lower BEST_THRESHOLD.")

# ═══════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE  (XGBoost)
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("Feature Importance  (XGBoost)")
print(f"{'═'*60}")
importance = pd.Series(xgb_f.feature_importances_, index=features)
print(importance.sort_values(ascending=False).to_string())