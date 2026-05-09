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

nifty["nifty_return"]           = np.log(nifty["Close"] / nifty["Close"].shift(1))
nifty["nifty_vol_10"]           = nifty["nifty_return"].rolling(10).std()
nifty["nifty_ma_ratio"]         = nifty["Close"] / nifty["Close"].rolling(20).mean()

banknifty["banknifty_return"]   = np.log(banknifty["Close"] / banknifty["Close"].shift(1))
banknifty["banknifty_vol_10"]   = banknifty["banknifty_return"].rolling(10).std()
banknifty["banknifty_ma_ratio"] = banknifty["Close"] / banknifty["Close"].rolling(20).mean()

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
df["log_return"]      = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
df["return_3d"]       = df["CLOSE"].pct_change(3)
df["return_5d"]       = df["CLOSE"].pct_change(5)
df["return_10d"]      = df["CLOSE"].pct_change(10)
df["return_21d"]      = df["CLOSE"].pct_change(21)

# MA ratios
df["ma5_ratio"]       = df["CLOSE"] / df["CLOSE"].rolling(5).mean()
df["ma20_ratio"]      = df["CLOSE"] / df["CLOSE"].rolling(20).mean()
df["ma_cross"]        = df["CLOSE"].rolling(5).mean() / df["CLOSE"].rolling(20).mean()

# Volatility
df["vol_5"]           = df["log_return"].rolling(5).std()
df["vol_20"]          = df["log_return"].rolling(20).std()
df["vol_ratio"]       = df["vol_5"] / (df["vol_20"] + 1e-9)

# RSI
delta = df["CLOSE"].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi"]             = 100 - (100 / (1 + gain / loss))
df["rsi_signal"]      = df["rsi"] - 50

# MACD histogram
exp1 = df["CLOSE"].ewm(span=12, adjust=False).mean()
exp2 = df["CLOSE"].ewm(span=26, adjust=False).mean()
macd_line             = exp1 - exp2
df["macd_hist"]       = macd_line - macd_line.ewm(span=9, adjust=False).mean()

# Bollinger Band position
bb_mid                = df["CLOSE"].rolling(20).mean()
bb_std                = df["CLOSE"].rolling(20).std()
df["bb_position"]     = (df["CLOSE"] - bb_mid) / (2 * bb_std + 1e-9)

# Intraday features
df["price_position"]  = (df["CLOSE"] - df["LOW"])   / (df["HIGH"] - df["LOW"] + 1e-9)
df["hl_range_ratio"]  = (df["HIGH"]  - df["LOW"])   /  df["CLOSE"]
df["oc_range_ratio"]  = (df["OPEN"]  - df["CLOSE"]) /  df["CLOSE"]

# VWAP & 52W
df["vwap_ratio"]      = df["CLOSE"] / df["VWAP"]
df["position_52w"]    = (df["CLOSE"] - df["52W L"]) / (df["52W H"] - df["52W L"] + 1e-9)

# Volume
df["volume_ratio"]    = df["VOLUME"] / (df["VOLUME"].rolling(20).mean() + 1e-9)
df["volume_trend"]    = df["VOLUME"].pct_change(5)

# Relative strength vs indices
df["hdfc_vs_nifty"]     = df["log_return"] - df["nifty_return"]
df["hdfc_vs_banknifty"] = df["log_return"] - df["banknifty_return"]

# Calendar
df["day_of_week"]     = df.index.dayofweek
df["month"]           = df.index.month

# Interaction features
df["rsi_x_volume"]    = df["rsi_signal"] * df["volume_ratio"]
df["trend_x_vol"]     = df["ma_cross"]   * df["vol_ratio"]
df["momentum_x_pos"]  = df["return_5d"]  * df["price_position"]
df["vwap_x_bb"]       = df["vwap_ratio"] * df["bb_position"]

# ── NEW: Regime features ──────────────────────────────
# Is price above its 20-day MA? 1=uptrend, 0=downtrend
df["trend_direction"] = np.where(df["CLOSE"] > df["CLOSE"].rolling(20).mean(), 1, 0)

# Slope of 20MA (positive = rising trend, negative = falling)
df["ma20_slope"]      = df["CLOSE"].rolling(20).mean().pct_change(5)

# ADX proxy — directional movement strength
df["dm_plus"]         = (df["HIGH"] - df["HIGH"].shift(1)).clip(lower=0)
df["dm_minus"]        = (df["LOW"].shift(1) - df["LOW"]).clip(lower=0)
df["dm_diff"]         = (df["dm_plus"] - df["dm_minus"]).rolling(14).mean()
df["dm_sum"]          = (df["dm_plus"] + df["dm_minus"]).rolling(14).mean()
df["adx_proxy"]       = (df["dm_diff"] / (df["dm_sum"] + 1e-9)).abs()
# adx_proxy > 0.2 → trending market; < 0.2 → ranging market

# ═══════════════════════════════════════════════════════
# 4. TARGET — 3-day forward return, noise filtered
# ═══════════════════════════════════════════════════════
fwd_return_3d = df["CLOSE"].shift(-3) / df["CLOSE"] - 1
THRESHOLD     = 0.005

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
    # new regime features
    "ma20_slope", "adx_proxy",
]

df = df.dropna(subset=features)

X = df[features]
y = df["target"]

neg, pos  = (y == 0).sum(), (y == 1).sum()
scale_pos = neg / pos
print(f"Class imbalance ratio: {scale_pos:.3f}\n")

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
# 7. FINAL HOLDOUT
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

lr_f = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
lr_f.fit(X_train_sc, y_train)
y_pred_lr = lr_f.predict(X_test_sc)
print(f"\nLogistic Regression: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

rf_f = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_f.fit(X_train, y_train)
y_pred_rf = rf_f.predict(X_test)
print(f"Random Forest: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

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
# 8. CONFIDENCE THRESHOLD TABLES
# ═══════════════════════════════════════════════════════
def threshold_table(model_name, proba, preds, y_true, thresholds):
    print(f"\n{'═'*60}")
    print(f"Confidence Threshold Analysis  ({model_name})")
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
rf_proba  = rf_f.predict_proba(X_test)

xgb_conf = threshold_table("XGBoost",       xgb_proba, y_pred_xgb, y_test, thresholds)
rf_conf  = threshold_table("Random Forest", rf_proba,  y_pred_rf,  y_test, thresholds)

# ═══════════════════════════════════════════════════════
# 9. AGREEMENT ENSEMBLE
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("Agreement Ensemble  (RF + XGBoost must agree on direction)")
print(f"{'─'*60}")
print(f"{'Conf ≥':>8} │ {'Accuracy':>10} │ {'Days':>8} │ {'Coverage':>10} │ {'Up':>6} │ {'Down':>6}")
print(f"{'─'*8}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*6}─┼─{'─'*6}")

agreement_mask = (y_pred_xgb == y_pred_rf)

for t in thresholds:
    conf_mask  = (xgb_conf >= t) & (rf_conf >= t)
    final_mask = agreement_mask & conf_mask
    n          = final_mask.sum()
    if n < 10:
        print(f"{t:>8.2f} │ {'—':>10} │ {n:>8} │  too few")
        continue
    preds_agreed = y_pred_xgb[final_mask]
    acc          = accuracy_score(y_test.values[final_mask], preds_agreed)
    coverage     = final_mask.mean() * 100
    up           = (preds_agreed == 1).sum()
    down         = (preds_agreed == 0).sum()
    print(f"{t:>8.2f} │ {acc:>10.4f} │ {n:>8} │ {coverage:>8.1f}% │ {up:>6} │ {down:>6}")

# ═══════════════════════════════════════════════════════
# 10. REGIME FILTER
#     Only take signals that align with the current
#     20MA trend direction — suppresses counter-trend traps
# ═══════════════════════════════════════════════════════
BEST_THRESHOLD = 0.54   # tune this based on ensemble table above

conf_mask    = (xgb_conf >= BEST_THRESHOLD) & (rf_conf >= BEST_THRESHOLD)
final_mask   = agreement_mask & conf_mask

test_dates   = X_test.index[final_mask]
signals_arr  = y_pred_xgb[final_mask]
actual_arr   = y_test.values[final_mask]

# Regime at each signal date (from the full df)
regime_arr   = df.loc[test_dates, "trend_direction"].values   # 1=uptrend, 0=downtrend
adx_arr      = df.loc[test_dates, "adx_proxy"].values

# Signal aligns with regime: UP signal in uptrend OR DOWN signal in downtrend
regime_aligned = (
    ((signals_arr == 1) & (regime_arr == 1)) |
    ((signals_arr == 0) & (regime_arr == 0))
)

# ADX filter: only trade when market is trending (not ranging)
ADX_THRESHOLD  = 0.15   # tune: higher = stricter trend requirement
trending_days  = adx_arr >= ADX_THRESHOLD

signal_df = pd.DataFrame({
    "date":           test_dates,
    "signal":         np.where(signals_arr == 1, "UP", "DOWN"),
    "xgb_conf":       xgb_proba.max(axis=1)[final_mask].round(4),
    "rf_conf":        rf_proba.max(axis=1)[final_mask].round(4),
    "regime":         np.where(regime_arr == 1, "UP", "DOWN"),
    "adx_proxy":      adx_arr.round(4),
    "actual":         np.where(actual_arr == 1, "UP", "DOWN"),
    "correct":        (signals_arr == actual_arr),
    "regime_aligned": regime_aligned,
    "trending":       trending_days,
})

print(f"\n{'═'*60}")
print(f"Signal Breakdown at Confidence ≥ {BEST_THRESHOLD}")
print(f"{'═'*60}")
print(signal_df.to_string(index=False))

# ── Accuracy cuts ─────────────────────────────────────
def accuracy_cut(label, mask_series):
    subset = signal_df[mask_series]
    if len(subset) == 0:
        print(f"{label}: no signals")
        return
    acc = subset["correct"].mean()
    up_s   = subset[subset["signal"] == "UP"]
    down_s = subset[subset["signal"] == "DOWN"]
    up_acc   = up_s["correct"].mean()   if len(up_s)   else float("nan")
    down_acc = down_s["correct"].mean() if len(down_s) else float("nan")
    print(f"\n{label}")
    print(f"  Total  : {len(subset)} signals  |  Accuracy: {acc:.1%}")
    print(f"  UP     : {len(up_s):3d} signals  |  Accuracy: {up_acc:.1%}"
          if len(up_s) else f"  UP     : 0 signals")
    print(f"  DOWN   : {len(down_s):3d} signals  |  Accuracy: {down_acc:.1%}"
          if len(down_s) else f"  DOWN   : 0 signals")

print(f"\n{'─'*60}")
accuracy_cut("All ensemble signals (no filter)",
             pd.Series([True] * len(signal_df)))
accuracy_cut(f"Regime-aligned only (signal matches 20MA trend)",
             signal_df["regime_aligned"])
accuracy_cut(f"Trending market only (ADX proxy ≥ {ADX_THRESHOLD})",
             signal_df["trending"])
accuracy_cut(f"Regime-aligned AND trending (strictest filter)",
             signal_df["regime_aligned"] & signal_df["trending"])

# ═══════════════════════════════════════════════════════
# 11. SHARPE RATIO BACKTEST
#     Tests three strategies:
#       A. Raw ensemble (all signals)
#       B. Regime-aligned signals only
#       C. Regime-aligned + trending market (ADX filtered)
# ═══════════════════════════════════════════════════════

def backtest(label, mask_series, signal_df, df_full):
    subset = signal_df[mask_series].copy()
    if len(subset) < 5:
        print(f"\n{label}: too few signals to backtest")
        return

    dates      = pd.DatetimeIndex(subset["date"])
    signals    = np.where(subset["signal"].values == "UP", 1, -1)

    # Use next-day log return as the trade P&L proxy
    # (buy/short at close, exit next day at close)
    next_dates = [df_full.index[df_full.index.get_loc(d) + 1]
                  if df_full.index.get_loc(d) + 1 < len(df_full) else None
                  for d in dates]
    returns    = []
    for d, sig in zip(dates, signals):
        loc = df_full.index.get_loc(d)
        if loc + 1 < len(df_full):
            ret = df_full["log_return"].iloc[loc + 1]
            returns.append(sig * ret)

    returns    = np.array(returns)
    cum_ret    = np.exp(np.cumsum(returns))
    total_ret  = np.exp(returns.sum()) - 1
    ann_ret    = np.exp(returns.mean() * 252) - 1
    ann_vol    = returns.std() * np.sqrt(252)
    sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0
    win_rate   = (returns > 0).mean()
    max_dd     = ((cum_ret / np.maximum.accumulate(cum_ret)) - 1).min()

    print(f"\n{'═'*60}")
    print(f"Backtest: {label}")
    print(f"{'─'*60}")
    print(f"  Signals traded : {len(returns)}")
    print(f"  Win rate       : {win_rate:.1%}")
    print(f"  Total return   : {total_ret:.2%}")
    print(f"  Ann. return    : {ann_ret:.2%}")
    print(f"  Ann. volatility: {ann_vol:.2%}")
    print(f"  Sharpe ratio   : {sharpe:.3f}  "
          f"{'★ Strong' if sharpe > 1.5 else '✓ Good' if sharpe > 1.0 else '~ Marginal' if sharpe > 0.5 else '✗ Weak'}")
    print(f"  Max drawdown   : {max_dd:.2%}")

backtest(
    "A — All ensemble signals",
    pd.Series([True] * len(signal_df)),
    signal_df, df
)
backtest(
    "B — Regime-aligned signals",
    signal_df["regime_aligned"],
    signal_df, df
)
backtest(
    "C — Regime-aligned + Trending (strictest)",
    signal_df["regime_aligned"] & signal_df["trending"],
    signal_df, df
)

# ═══════════════════════════════════════════════════════
# 12. EQUITY CURVE  (Strategy C vs Buy-and-Hold)
# ═══════════════════════════════════════════════════════
strict_mask = signal_df["regime_aligned"] & signal_df["trending"]
subset      = signal_df[strict_mask].copy()

if len(subset) >= 5:
    dates   = pd.DatetimeIndex(subset["date"])
    signals = np.where(subset["signal"].values == "UP", 1, -1)
    rets    = []
    for d, sig in zip(dates, signals):
        loc = df.index.get_loc(d)
        if loc + 1 < len(df):
            rets.append(sig * df["log_return"].iloc[loc + 1])

    rets       = np.array(rets)
    equity     = pd.Series(np.exp(np.cumsum(rets)), index=dates[:len(rets)])

    # Buy-and-hold over the same period
    bh_start   = df["CLOSE"].loc[dates[0]]
    bh_end     = df["CLOSE"].loc[dates[-1]]
    bh_ret     = bh_end / bh_start - 1

    print(f"\n{'═'*60}")
    print(f"Equity Curve  (Strategy C — Regime + Trend filtered)")
    print(f"{'─'*60}")
    print(f"  Period       : {dates[0].date()} → {dates[-1].date()}")
    print(f"  Strategy     : {equity.iloc[-1] - 1:.2%} total return")
    print(f"  Buy-and-hold : {bh_ret:.2%} total return  (HDFC same period)")
    print(f"\n  Monthly equity snapshots:")
    monthly = equity.resample("ME").last().dropna()
    for date, val in monthly.items():
        bar = "█" * int((val - 1) * 40)
        print(f"    {date.strftime('%b %Y')}  {val:.4f}  {bar}")

# ═══════════════════════════════════════════════════════
# 13. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("Feature Importance  (XGBoost)")
print(f"{'═'*60}")
importance = pd.Series(xgb_f.feature_importances_, index=features)
print(importance.sort_values(ascending=False).to_string())