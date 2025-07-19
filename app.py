import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("📈 日本株 LightGBM 株価予測アプリ")

@st.cache_data
def download_stock_data(symbol):
    df = yf.download(symbol, start="2015-01-01", end="2025-07-01")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def create_features(df):
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    features = df[['Close', 'MA20', 'RSI', 'MACD']]
    return features, df['Target']

def train_and_predict(features, target, days):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test[-days:])
    dates = X_test.index[-days:]
    return dates, preds, model

symbols = st.multiselect("銘柄コードを選んでください（複数可）",
                         ["7203.T", "6758.T", "9984.T", "9434.T", "9983.T"],
                         default=["7203.T", "6758.T"])
days = st.slider("予測日数", 5, 30, 10)

if st.button("予測開始"):
    for symbol in symbols:
        st.header(f"📊 {symbol} の予測")
        df = download_stock_data(symbol)
        features, target = create_features(df)
        dates, preds, model = train_and_predict(features, target, days)

        # プロット
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-100:], df['Close'].iloc[-100:], label="実株価")
        ax.plot(dates, preds, label="予測", color="red")
        ax.set_title(f"{symbol} 株価予測（LightGBM）")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        result_df = pd.DataFrame({"予測日": dates, "予測株価": preds})
        result_df.set_index("予測日", inplace=True)
        st.dataframe(result_df)
