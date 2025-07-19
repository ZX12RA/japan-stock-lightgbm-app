import streamlit as st
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt

st.title("🇯🇵 日本株予測アプリ（XGBoost）")

@st.cache_data
def load_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["RSI"] = 100 - 100 / (1 + df["Close"].pct_change().rolling(window=14).mean())
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df.dropna(inplace=True)
    return df

def create_features(df):
    features = df[["MA20", "RSI", "MACD"]]
    target = df["Close"].shift(-1)
    features = features[:-1]
    target = target[:-1]
    return features, target

def train_and_predict(features, target, days):
    X_train = features[:-days]
    y_train = target[:-days]
    X_test = features[-days:]

    # XGBoost用にDMatrixへ変換
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dates = features.index[-days:]
    return dates, y_pred, model

ticker = st.selectbox("銘柄を選択", ["7203.T", "6758.T", "9984.T", "9432.T", "8306.T"])
df = load_stock_data(ticker)
df = add_technical_indicators(df)
features, target = create_features(df)

days = st.slider("予測日数", 5, 60, 30)
dates, preds, model = train_and_predict(features, target, days)

# グラフ描画
fig, ax = plt.subplots()
ax.plot(df.index[-days:], df["Close"].iloc[-days:], label="Actual")
ax.plot(dates, preds, label="Predicted")
ax.set_title(f"{ticker} 株価予測")
ax.legend()
st.pyplot(fig)

st.subheader("📈 予測結果")
st.dataframe(pd.DataFrame({"Date": dates, "Predicted": preds}).set_index("Date"))
