
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime

st.title("📈 日本株予測アプリ (XGBoost)")

symbol = st.selectbox("銘柄コードを選択してください", ["7203.T", "6758.T", "9984.T", "9432.T", "8306.T"])
days = st.slider("予測日数", 7, 90, 30)

@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="2y")
    df = df[["Close"]]
    df = df.dropna()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["Std20"] = df["Close"].rolling(window=20).std()
    df["Target"] = df["Close"].shift(-days)
    df = df.dropna()
    return df

df = get_data(symbol)

def train_and_predict(df, days):
    features = df[["Close", "Return", "MA5", "MA20", "Std20"]]
    target = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)

    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    future_input = features.iloc[-days:]
    preds = model.predict(future_input)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')

    return df.index, df["Close"], future_dates, preds

dates_past, close_past, future_dates, preds = train_and_predict(df, days)

# グラフ表示
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates_past, close_past, label="過去実績", color="blue")
ax.plot(future_dates, preds, label="未来予測", color="orange", linestyle="--")
ax.axvline(future_dates[0], color="gray", linestyle="dotted", label="予測開始点")
ax.set_title(f"{symbol} 株価予測（{days}営業日）")
ax.set_xlabel("日付")
ax.set_ylabel("終値")
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)
