#using lstm for a change
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")

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

# Regime features
df["trend_direction"] = np.where(df["CLOSE"] > df["CLOSE"].rolling(20).mean(), 1, 0)
df["ma20_slope"]      = df["CLOSE"].rolling(20).mean().pct_change(5)
df["dm_plus"]         = (df["HIGH"] - df["HIGH"].shift(1)).clip(lower=0)
df["dm_minus"]        = (df["LOW"].shift(1) - df["LOW"]).clip(lower=0)
df["dm_diff"]         = (df["dm_plus"] - df["dm_minus"]).rolling(14).mean()
df["dm_sum"]          = (df["dm_plus"] + df["dm_minus"]).rolling(14).mean()
df["adx_proxy"]       = (df["dm_diff"] / (df["dm_sum"] + 1e-9)).abs()

# NEW: Overnight gap
df["overnight_gap"]   = (df["OPEN"] - df["PREV. CLOSE"]) / df["PREV. CLOSE"]

# NEW: Distance from 52W high
df["dist_52w_high"]   = (df["52W H"] - df["CLOSE"]) / df["52W H"]

# NEW: Avg trade size ratio (large block trade signal)
df["avg_trade_size"]  = df["VALUE"] / (df["VOLUME"] + 1e-9)
df["trade_size_ratio"]= df["avg_trade_size"] / (
    df["avg_trade_size"].rolling(20).mean() + 1e-9
)

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
print(f"Total filtered trading days: {len(df)}")

# ═══════════════════════════════════════════════════════
# 5. FEATURE LIST
# ═══════════════════════════════════════════════════════
features = [
    "log_return", "return_5d", "return_21d",
    "ma_cross", "macd_hist", "ma20_slope",
    "rsi_signal", "bb_position",
    "vol_ratio",
    "price_position", "vwap_ratio", "oc_range_ratio",
    "position_52w", "dist_52w_high",
    "volume_ratio", "trade_size_ratio",
    "hdfc_vs_nifty", "hdfc_vs_banknifty",
    "overnight_gap",
    "adx_proxy",
    "rsi_x_volume", "trend_x_vol", "momentum_x_pos",
]

df = df.dropna(subset=features)
print(f"After dropping NaN in features: {len(df)} days\n")

# ═══════════════════════════════════════════════════════
# 6. BUILD LSTM SEQUENCES
#    LSTM needs 3D input: (samples, timesteps, features)
#    Each sample = a LOOKBACK-day window of features
#    predicting the target on day t
# ═══════════════════════════════════════════════════════
LOOKBACK = 20   # 20 trading days = ~1 month of context per prediction

def make_sequences(X_arr, y_arr, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X_arr)):
        Xs.append(X_arr[i - lookback:i])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)

# ═══════════════════════════════════════════════════════
# 7. CHRONOLOGICAL SPLIT  70% train / 10% val / 20% test
# ═══════════════════════════════════════════════════════
X_raw = df[features].values
y_raw = df["target"].values
dates = df.index

n         = len(X_raw)
train_end = int(n * 0.70)

# Fit scaler strictly on train portion only
scaler = StandardScaler()
scaler.fit(X_raw[:train_end])
X_scaled = scaler.transform(X_raw)

X_seq, y_seq = make_sequences(X_scaled, y_raw, LOOKBACK)
dates_seq    = dates[LOOKBACK:]

n_seq         = len(X_seq)
train_end_seq = int(n_seq * 0.70)
val_end_seq   = int(n_seq * 0.80)

X_train = X_seq[:train_end_seq]
y_train = y_seq[:train_end_seq]
X_val   = X_seq[train_end_seq:val_end_seq]
y_val   = y_seq[train_end_seq:val_end_seq]
X_test  = X_seq[val_end_seq:]
y_test  = y_seq[val_end_seq:]
dates_test = dates_seq[val_end_seq:]

print(f"{'═'*60}")
print(f"Data Split")
print(f"{'─'*60}")
print(f"  Train      : {len(X_train):4d} sequences  "
      f"({dates_seq[0].date()} → {dates_seq[train_end_seq-1].date()})")
print(f"  Validation : {len(X_val):4d} sequences  "
      f"({dates_seq[train_end_seq].date()} → {dates_seq[val_end_seq-1].date()})")
print(f"  Test       : {len(X_test):4d} sequences  "
      f"({dates_seq[val_end_seq].date()} → {dates_seq[-1].date()})")
print(f"  Input shape: {X_train.shape}  (samples, lookback_days, features)")

# ═══════════════════════════════════════════════════════
# 8. LSTM ARCHITECTURE
#    Layer 1 LSTM(128): captures short-term patterns
#                       (RSI crossovers, momentum shifts)
#    Layer 2 LSTM(64):  captures medium-term structure
#                       (trend regimes, vol cycles)
#    Layer 3 LSTM(32):  compresses into a decision signal
#    Dense head:        maps to binary probability
# ═══════════════════════════════════════════════════════
n_features = X_train.shape[2]

model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(LOOKBACK, n_features),
         kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(64, return_sequences=True,
         kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(32, return_sequences=False,
         kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ═══════════════════════════════════════════════════════
# 9. TRAIN
# ═══════════════════════════════════════════════════════
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
class_weight = {0: 1.0, 1: neg_count / pos_count}
print(f"\nClass weights → 0: 1.000  1: {class_weight[1]:.3f}")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=7, min_lr=1e-6, verbose=1)
]

print(f"\n{'═'*60}")
print("Training LSTM  (early stopping enabled, max 100 epochs)")
print(f"{'═'*60}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

best_epoch   = np.argmin(history.history["val_loss"]) + 1
best_val_acc = max(history.history["val_accuracy"])
print(f"\nStopped at epoch : {len(history.history['val_loss'])}")
print(f"Best epoch       : {best_epoch}")
print(f"Best val accuracy: {best_val_acc:.4f}")

# ═══════════════════════════════════════════════════════
# 10. HOLDOUT EVALUATION
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"Final Holdout Results  ({len(X_test)} test sequences)")
print(f"{'═'*60}")

lstm_proba = model.predict(X_test, verbose=0).flatten()
lstm_preds = (lstm_proba >= 0.5).astype(int)

print(f"\nLSTM Raw Accuracy: {accuracy_score(y_test, lstm_preds):.4f}")
print(classification_report(y_test, lstm_preds,
                             target_names=["DOWN (0)", "UP (1)"]))

# ═══════════════════════════════════════════════════════
# 11. CONFIDENCE THRESHOLD ANALYSIS
#     P(up) > 0.5 = UP signal, confidence = P(up)
#     P(up) < 0.5 = DOWN signal, confidence = 1 - P(up)
# ═══════════════════════════════════════════════════════
confidence = np.where(lstm_proba >= 0.5, lstm_proba, 1 - lstm_proba)

print(f"\n{'═'*60}")
print("Confidence Threshold Analysis  (LSTM)")
print(f"{'─'*60}")
print(f"{'Threshold':>10} │ {'Accuracy':>10} │ {'Days':>8} │ {'Coverage':>10}")
print(f"{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}")

thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]
for t in thresholds:
    mask = confidence >= t
    n    = mask.sum()
    if n < 10:
        print(f"{t:>10.2f} │ {'—':>10} │ {n:>8} │  too few")
        continue
    acc      = accuracy_score(y_test[mask], lstm_preds[mask])
    coverage = mask.mean() * 100
    print(f"{t:>10.2f} │ {acc:>10.4f} │ {n:>8} │ {coverage:>8.1f}%")

# ═══════════════════════════════════════════════════════
# 12. REGIME FILTER
# ═══════════════════════════════════════════════════════
BEST_THRESHOLD = 0.54
ADX_THRESHOLD  = 0.15

regime_arr = df.loc[dates_test, "trend_direction"].values
adx_arr    = df.loc[dates_test, "adx_proxy"].values

conf_mask      = confidence >= BEST_THRESHOLD
regime_aligned = (
    ((lstm_preds == 1) & (regime_arr == 1)) |
    ((lstm_preds == 0) & (regime_arr == 0))
)
trending_days  = adx_arr >= ADX_THRESHOLD

signal_df = pd.DataFrame({
    "date":           dates_test,
    "signal":         np.where(lstm_preds == 1, "UP", "DOWN"),
    "confidence":     confidence.round(4),
    "p_up":           lstm_proba.round(4),
    "regime":         np.where(regime_arr == 1, "UP", "DOWN"),
    "adx_proxy":      adx_arr.round(4),
    "actual":         np.where(y_test == 1, "UP", "DOWN"),
    "correct":        (lstm_preds == y_test),
    "conf_filter":    conf_mask,
    "regime_aligned": regime_aligned,
    "trending":       trending_days,
})

print(f"\n{'═'*60}")
print(f"Signal Breakdown at Confidence ≥ {BEST_THRESHOLD}")
print(f"{'═'*60}")
display_cols = ["date", "signal", "confidence", "regime",
                "adx_proxy", "actual", "correct", "regime_aligned", "trending"]
print(signal_df[signal_df["conf_filter"]][display_cols].to_string(index=False))

# ── Accuracy cuts ─────────────────────────────────────
def accuracy_cut(label, mask_series, df_signals):
    subset = df_signals[mask_series]
    if len(subset) == 0:
        print(f"\n{label}: no signals"); return
    acc    = subset["correct"].mean()
    up_s   = subset[subset["signal"] == "UP"]
    down_s = subset[subset["signal"] == "DOWN"]
    print(f"\n{label}")
    print(f"  Total  : {len(subset)} signals  |  Accuracy: {acc:.1%}")
    if len(up_s)   > 0:
        print(f"  UP     : {len(up_s):3d} signals  |  Accuracy: {up_s['correct'].mean():.1%}")
    if len(down_s) > 0:
        print(f"  DOWN   : {len(down_s):3d} signals  |  Accuracy: {down_s['correct'].mean():.1%}")

base = signal_df["conf_filter"]
print(f"\n{'─'*60}")
accuracy_cut("All LSTM signals (confidence filter only)",
             base, signal_df)
accuracy_cut("Regime-aligned only",
             base & signal_df["regime_aligned"], signal_df)
accuracy_cut(f"Trending market only (ADX ≥ {ADX_THRESHOLD})",
             base & signal_df["trending"], signal_df)
accuracy_cut("Regime-aligned AND trending (strictest)",
             base & signal_df["regime_aligned"] & signal_df["trending"], signal_df)

# ═══════════════════════════════════════════════════════
# 13. BACKTEST
# ═══════════════════════════════════════════════════════
def backtest(label, mask_series, signal_df, df_full):
    subset = signal_df[mask_series].copy()
    if len(subset) < 5:
        print(f"\n{label}: too few signals"); return None

    dates_bt = pd.DatetimeIndex(subset["date"])
    sigs     = np.where(subset["signal"].values == "UP", 1, -1)
    returns  = []
    for d, sig in zip(dates_bt, sigs):
        loc = df_full.index.get_loc(d)
        if loc + 1 < len(df_full):
            returns.append(sig * df_full["log_return"].iloc[loc + 1])

    if len(returns) < 5:
        return None

    returns  = np.array(returns)
    cum_ret  = np.exp(np.cumsum(returns))
    tot_ret  = np.exp(returns.sum()) - 1
    ann_ret  = np.exp(returns.mean() * 252) - 1
    ann_vol  = returns.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    win_rate = (returns > 0).mean()
    max_dd   = ((cum_ret / np.maximum.accumulate(cum_ret)) - 1).min()
    rating   = ("★ Strong"   if sharpe > 1.5 else
                "✓ Good"     if sharpe > 1.0 else
                "~ Marginal" if sharpe > 0.5 else "✗ Weak")

    print(f"\n{'═'*60}")
    print(f"Backtest: {label}")
    print(f"{'─'*60}")
    print(f"  Signals traded : {len(returns)}")
    print(f"  Win rate       : {win_rate:.1%}")
    print(f"  Total return   : {tot_ret:.2%}")
    print(f"  Ann. return    : {ann_ret:.2%}")
    print(f"  Ann. volatility: {ann_vol:.2%}")
    print(f"  Sharpe ratio   : {sharpe:.3f}  {rating}")
    print(f"  Max drawdown   : {max_dd:.2%}")
    return pd.Series(cum_ret, index=dates_bt[:len(returns)])

backtest("A — All LSTM signals",          base, signal_df, df)
backtest("B — Regime-aligned",            base & signal_df["regime_aligned"], signal_df, df)
backtest("C — Regime-aligned + Trending", base & signal_df["regime_aligned"] & signal_df["trending"], signal_df, df)

# ═══════════════════════════════════════════════════════
# 14. EQUITY CURVE  (Strategy C)
# ═══════════════════════════════════════════════════════
strict_mask = base & signal_df["regime_aligned"] & signal_df["trending"]
subset      = signal_df[strict_mask].copy()

if len(subset) >= 5:
    dates_bt = pd.DatetimeIndex(subset["date"])
    sigs     = np.where(subset["signal"].values == "UP", 1, -1)
    rets     = []
    for d, sig in zip(dates_bt, sigs):
        loc = df.index.get_loc(d)
        if loc + 1 < len(df):
            rets.append(sig * df["log_return"].iloc[loc + 1])

    rets   = np.array(rets)
    equity = pd.Series(np.exp(np.cumsum(rets)), index=dates_bt[:len(rets)])

    bh_ret = df["CLOSE"].loc[dates_bt[-1]] / df["CLOSE"].loc[dates_bt[0]] - 1

    print(f"\n{'═'*60}")
    print(f"Equity Curve  (LSTM Strategy C)")
    print(f"{'─'*60}")
    print(f"  Period        : {dates_bt[0].date()} → {dates_bt[-1].date()}")
    print(f"  LSTM strategy : {equity.iloc[-1] - 1:.2%} total return")
    print(f"  Buy-and-hold  : {bh_ret:.2%} total return")
    print(f"\n  Monthly snapshots:")
    for date, val in equity.resample("ME").last().dropna().items():
        bar = "█" * int((val - 1) * 40)
        print(f"    {date.strftime('%b %Y')}  {val:.4f}  {bar}")

# ═══════════════════════════════════════════════════════
# 15. LSTM vs v5 ENSEMBLE — SIDE-BY-SIDE
# ═══════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("v6 LSTM vs v5 Ensemble — Direct Comparison")
print(f"{'─'*60}")
print(f"  v5 reference: Sharpe=5.624 | MaxDD=-5.66% | TotalRet=+15.27%")
print(f"                Accuracy(C)=68.2% | Signals(C)=44 | WinRate=68.2%")
print(f"{'─'*60}")

strict_subset = signal_df[strict_mask]
lstm_c_acc    = strict_subset["correct"].mean() if len(strict_subset) > 0 else 0
print(f"\n  LSTM raw accuracy       : {accuracy_score(y_test, lstm_preds):.4f}")
print(f"  LSTM filtered acc (C)   : {lstm_c_acc:.4f}")
print(f"  LSTM signals (C)        : {len(strict_subset)}")
print(f"\n  Full Sharpe + return for LSTM printed in Section 13 above.")
print(f"\n  Key difference — LSTM advantage:")
print(f"    Tree models see features independently per day.")
print(f"    LSTM sees the SEQUENCE of the last {LOOKBACK} days,")
print(f"    learning how patterns evolve over time — not just")
print(f"    what the indicators are, but how they're changing.")