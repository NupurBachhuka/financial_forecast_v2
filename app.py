import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
# from sklearn.ensemble import VotingClassifier

file_path = "rawdata/HDFCBANK_2015_2026_merged.csv"

df = pd.read_csv(file_path, parse_dates=["DATE"])

# sort chronologically (important for time series)
df = df.sort_values("DATE")

# set date as index
df.set_index("DATE", inplace=True)

# print(df.head())
# print(df.info())
# print(df.isnull().sum())

#change data types of numeric columns (remove commas and convert to float)
cols_to_convert = [
    "OPEN", "HIGH", "LOW", "PREV. CLOSE", 
    "LTP", "CLOSE", "VWAP", "52W H", "52W L", "VALUE"
]

for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(",", "")
    df[col] = pd.to_numeric(df[col], errors="coerce")

#print(df.dtypes)
df.drop(columns=["SERIES"], inplace=True)
#print(df.isna().sum())

#feature engineering:
df["log_return"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

# Moving averages
df["ma_5"] = df["CLOSE"].rolling(window=5).mean()
df["ma_20"] = df["CLOSE"].rolling(window=20).mean()

# Rolling volatility
df["volatility_10"] = df["log_return"].rolling(window=10).std()
df["volatility_20"] = df["log_return"].rolling(window=20).std()

# Momentum
df["momentum_5"] = df["CLOSE"] - df["CLOSE"].shift(5)
df["momentum_10"] = df["CLOSE"] - df["CLOSE"].shift(10)

#RSI
delta = df["CLOSE"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["rsi"] = 100 - (100 / (1 + rs))

#MACD
exp1 = df["CLOSE"].ewm(span=12, adjust=False).mean()
exp2 = df["CLOSE"].ewm(span=26, adjust=False).mean()
df["macd"] = exp1 - exp2
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

#Volume features
df["volume_change"] = df["VOLUME"].pct_change()
df["volume_ma_10"] = df["VOLUME"].rolling(10).mean()

#Price range features
df["high_low_range"] = df["HIGH"] - df["LOW"]
df["open_close_range"] = df["OPEN"] - df["CLOSE"]

df["target_class"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)

# print(df.isnull().sum())
df = df.dropna()
# print(df.isnull().sum())

#feature engineering

features = ["log_return", "ma_5", "ma_20", "volatility_10", "volatility_20",
            "momentum_5", "momentum_10", "rsi", "macd", "macd_signal", "volume_change", "volume_ma_10",
            "high_low_range", "open_close_range"]

for col in features:
    df[col] = df[col].shift(1)

df = df.dropna()

X = df[features]
y = df["target_class"]

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

#XGBoost
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


#phase two: add index data and retrain models
nifty = pd.read_csv("rawdata/NIFTY50_2015_2026_merged.csv", parse_dates=["Date"])
banknifty = pd.read_csv("rawdata/NIFTYBANK_2015_2026_merged.csv", parse_dates=["Date"])

# sort + set index
nifty = nifty.sort_values("Date").set_index("Date")
banknifty = banknifty.sort_values("Date").set_index("Date")

for df_idx in [nifty, banknifty]:
    df_idx["Close"] = df_idx["Close"].astype(str).str.replace(",", "")
    df_idx["Close"] = pd.to_numeric(df_idx["Close"], errors="coerce")

# Nifty features
nifty["nifty_return"] = np.log(nifty["Close"] / nifty["Close"].shift(1))
nifty["nifty_volatility_10"] = nifty["nifty_return"].rolling(10).std()

# Bank Nifty features
banknifty["banknifty_return"] = np.log(banknifty["Close"] / banknifty["Close"].shift(1))
banknifty["banknifty_volatility_10"] = banknifty["banknifty_return"].rolling(10).std()

df = df.merge(nifty[["nifty_return", "nifty_volatility_10"]],
              left_index=True, right_index=True, how="inner")

df = df.merge(banknifty[["banknifty_return", "banknifty_volatility_10"]],
              left_index=True, right_index=True, how="inner")

index_features = [
    "nifty_return", "nifty_volatility_10",
    "banknifty_return", "banknifty_volatility_10"
]

for col in index_features:
    df[col] = df[col].shift(1)

features = features + index_features
df = df.dropna()

X_index = df[features]
y_index = df["target_class"]

split = int(len(df) * 0.8)
X_train_index, X_test_index = X_index[:split], X_index[split:]
y_train_index, y_test_index = y_index[:split], y_index[split:]

scaler_index = StandardScaler()

X_train_scaled_index = scaler_index.fit_transform(X_train_index)
X_test_scaled_index = scaler_index.transform(X_test_index)

# Logistic Regression(with index features)
lr_index = LogisticRegression(max_iter=2000)
lr_index.fit(X_train_scaled_index, y_train_index)
y_pred_lr_index = lr_index.predict(X_test_scaled_index)
print("Logistic Regression Accuracy(with index features):", accuracy_score(y_test_index, y_pred_lr_index))
print("Logistic Regression Classification Report(with index features):")
print(classification_report(y_test_index, y_pred_lr_index))
# Random Forest(with index features)
rf_index = RandomForestClassifier(n_estimators=100, random_state=42)
rf_index.fit(X_train_index, y_train_index)
y_pred_rf_index = rf_index.predict(X_test_index)
print("Random Forest Accuracy(with index features):", accuracy_score(y_test_index, y_pred_rf_index))
print("Random Forest Classification Report(with index features):")
print(classification_report(y_test_index, y_pred_rf_index))
#XGBoost(with index features)
xgb_index = XGBClassifier(eval_metric='logloss')
xgb_index.fit(X_train_index, y_train_index)
y_pred_xgb_index = xgb_index.predict(X_test_index)
print("XGBoost Accuracy(with index features):", accuracy_score(y_test_index, y_pred_xgb_index))
print("XGBoost Classification Report(with index features):")
print(classification_report(y_test_index, y_pred_xgb_index))