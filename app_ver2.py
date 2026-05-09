import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# ─────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────
file_path = "rawdata/HDFCBANK_2015_2026_merged.csv"
df = pd.read_csv(file_path, parse_dates=["DATE"])
df = df.sort_values("DATE").set_index("DATE")

cols_to_convert = ["OPEN", "HIGH", "LOW", "PREV. CLOSE",
                   "LTP", "CLOSE", "VWAP", "52W H", "52W L", "VALUE"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

df.drop(columns=["SERIES"], inplace=True, errors="ignore")

# ─────────────────────────────────────────
# 2. LOAD INDEX DATA
# ─────────────────────────────────────────
nifty = pd.read_csv("rawdata/NIFTY50_2015_2026_merged.csv", parse_dates=["Date"])
banknifty = pd.read_csv("rawdata/NIFTYBANK_2015_2026_merged.csv", parse_dates=["Date"])

for idx_df in [nifty, banknifty]:
    idx_df.sort_values("Date", inplace=True)
    idx_df.set_index("Date", inplace=True)
    idx_df["Close"] = pd.to_numeric(idx_df["Close"].astype(str).str.replace(",", ""), errors="coerce")

nifty["nifty_return"]         = np.log(nifty["Close"] / nifty["Close"].shift(1))
nifty["nifty_vol_10"]         = nifty["nifty_return"].rolling(10).std()
nifty["nifty_ma_ratio"]       = nifty["Close"] / nifty["Close"].rolling(20).mean()

banknifty["banknifty_return"]  = np.log(banknifty["Close"] / banknifty["Close"].shift(1))
banknifty["banknifty_vol_10"]  = banknifty["banknifty_return"].rolling(10).std()
banknifty["banknifty_ma_ratio"]= banknifty["Close"] / banknifty["Close"].rolling(20).mean()

df = df.merge(nifty[["nifty_return", "nifty_vol_10", "nifty_ma_ratio"]],
              left_index=True, right_index=True, how="inner")
df = df.merge(banknifty[["banknifty_return", "banknifty_vol_10", "banknifty_ma_ratio"]],
              left_index=True, right_index=True, how="inner")

# ─────────────────────────────────────────
# 3. FEATURE ENGINEERING (stationary features only)
# ─────────────────────────────────────────

# Returns & momentum (% based, not raw price)
df["log_return"]     = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
df["return_2d"]      = df["CLOSE"].pct_change(2)
df["return_5d"]      = df["CLOSE"].pct_change(5)
df["return_10d"]     = df["CLOSE"].pct_change(10)
df["return_21d"]     = df["CLOSE"].pct_change(21)

# MA ratios (price relative to MA — stationary & scale-invariant)
df["ma5_ratio"]      = df["CLOSE"] / df["CLOSE"].rolling(5).mean()
df["ma20_ratio"]     = df["CLOSE"] / df["CLOSE"].rolling(20).mean()
df["ma_cross"]       = df["CLOSE"].rolling(5).mean() / df["CLOSE"].rolling(20).mean()

# Volatility
df["vol_5"]          = df["log_return"].rolling(5).std()
df["vol_20"]         = df["log_return"].rolling(20).std()
df["vol_ratio"]      = df["vol_5"] / df["vol_20"]   # vol contraction signal

# RSI
delta = df["CLOSE"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss))
df["rsi_signal"] = df["rsi"] - 50   # centered, easier for LR

# MACD
exp1 = df["CLOSE"].ewm(span=12, adjust=False).mean()
exp2 = df["CLOSE"].ewm(span=26, adjust=False).mean()
df["macd_hist"] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9, adjust=False).mean()

# Bollinger Band position (-1 to +1)
bb_mid    = df["CLOSE"].rolling(20).mean()
bb_std    = df["CLOSE"].rolling(20).std()
df["bb_position"] = (df["CLOSE"] - bb_mid) / (2 * bb_std + 1e-9)

# Intraday features (available at EOD — no leakage)
df["price_position"]  = (df["CLOSE"] - df["LOW"])  / (df["HIGH"] - df["LOW"] + 1e-9)
df["hl_range_ratio"]  = (df["HIGH"] - df["LOW"])   / df["CLOSE"]   # normalized range
df["oc_range_ratio"]  = (df["OPEN"] - df["CLOSE"]) / df["CLOSE"]   # gap signal

# VWAP deviation
df["vwap_ratio"]      = df["CLOSE"] / df["VWAP"]

# 52W position (where are we in the yearly range)
df["position_52w"]    = (df["CLOSE"] - df["52W L"]) / (df["52W H"] - df["52W L"] + 1e-9)

# Volume features (ratios, not raw levels)
df["volume_ratio"]    = df["VOLUME"] / (df["VOLUME"].rolling(20).mean() + 1e-9)
df["volume_trend"]    = df["VOLUME"].pct_change(5)

# Relative strength vs indices
df["hdfc_vs_nifty"]      = df["log_return"] - df["nifty_return"]
df["hdfc_vs_banknifty"]  = df["log_return"] - df["banknifty_return"]

# Calendar effects
df["day_of_week"] = df.index.dayofweek
df["month"]       = df.index.month

# ─────────────────────────────────────────
# 4. TARGET (predict next day direction)
# ─────────────────────────────────────────
df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)

df = df.dropna()

# ─────────────────────────────────────────
# 5. FEATURE LIST
# ─────────────────────────────────────────
features = [
    # Returns
    "log_return", "return_2d", "return_5d", "return_10d", "return_21d",
    # MA
    "ma5_ratio", "ma20_ratio", "ma_cross",
    # Volatility
    "vol_5", "vol_20", "vol_ratio",
    # Momentum indicators
    "rsi_signal", "macd_hist", "bb_position",
    # Intraday
    "price_position", "hl_range_ratio", "oc_range_ratio",
    # VWAP & 52W
    "vwap_ratio", "position_52w",
    # Volume
    "volume_ratio", "volume_trend",
    # Index context
    "nifty_return", "nifty_vol_10", "nifty_ma_ratio",
    "banknifty_return", "banknifty_vol_10", "banknifty_ma_ratio",
    # Relative strength
    "hdfc_vs_nifty", "hdfc_vs_banknifty",
    # Calendar
    "day_of_week", "month",
]

X = df[features]
y = df["target"]

# ─────────────────────────────────────────
# 6. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────
# Simulates real trading: train on past, test on future, roll forward
TRAIN_WINDOW = 504   # ~2 years
TEST_WINDOW  = 63    # ~3 months

wf_results = {"lr": [], "rf": [], "xgb": []}

n_windows = (len(df) - TRAIN_WINDOW) // TEST_WINDOW

print(f"\n{'='*55}")
print(f"Walk-Forward Validation ({n_windows} windows)")
print(f"{'='*55}")

for i in range(n_windows):
    start = i * TEST_WINDOW
    end_train = start + TRAIN_WINDOW
    end_test  = end_train + TEST_WINDOW

    if end_test > len(df):
        break

    X_tr, y_tr   = X.iloc[start:end_train], y.iloc[start:end_train]
    X_te, y_te   = X.iloc[end_train:end_test], y.iloc[end_train:end_test]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # LR
    lr = LogisticRegression(max_iter=2000, C=0.1)
    lr.fit(X_tr_sc, y_tr)
    wf_results["lr"].append(accuracy_score(y_te, lr.predict(X_te_sc)))

    # RF
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=20, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    wf_results["rf"].append(accuracy_score(y_te, rf.predict(X_te)))

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4,
        learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    wf_results["xgb"].append(accuracy_score(y_te, xgb.predict(X_te)))

for name, scores in wf_results.items():
    print(f"{name.upper():5s} → Mean: {np.mean(scores):.4f}  Std: {np.std(scores):.4f}  "
          f"Min: {np.min(scores):.4f}  Max: {np.max(scores):.4f}")

# ─────────────────────────────────────────
# 7. FINAL HOLDOUT EVALUATION (last 20%)
# ─────────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler_final = StandardScaler()
X_train_sc = scaler_final.fit_transform(X_train)
X_test_sc  = scaler_final.transform(X_test)

print(f"\n{'='*55}")
print("Final Holdout Results (last 20%)")
print(f"{'='*55}")

# LR
lr_f = LogisticRegression(max_iter=2000, C=0.1)
lr_f.fit(X_train_sc, y_train)
print(f"\nLogistic Regression: {accuracy_score(y_test, lr_f.predict(X_test_sc)):.4f}")
print(classification_report(y_test, lr_f.predict(X_test_sc)))

# RF
rf_f = RandomForestClassifier(
    n_estimators=300, max_depth=6,
    min_samples_leaf=20, random_state=42, n_jobs=-1
)
rf_f.fit(X_train, y_train)
print(f"Random Forest: {accuracy_score(y_test, rf_f.predict(X_test)):.4f}")
print(classification_report(y_test, rf_f.predict(X_test)))

# XGBoost
xgb_f = XGBClassifier(
    n_estimators=500, max_depth=4,
    learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", verbosity=0
)
xgb_f.fit(X_train, y_train)
print(f"XGBoost: {accuracy_score(y_test, xgb_f.predict(X_test)):.4f}")
print(classification_report(y_test, xgb_f.predict(X_test)))

# ─────────────────────────────────────────
# 8. FEATURE IMPORTANCE (XGBoost)
# ─────────────────────────────────────────
importance = pd.Series(xgb_f.feature_importances_, index=features)
print("\nTop 10 Features:")
print(importance.sort_values(ascending=False).head(10).to_string())