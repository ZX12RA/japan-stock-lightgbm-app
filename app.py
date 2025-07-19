import streamlit as st
import pandas as pd
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np

st.title("日本株 予測アプリ（XGBoost版）")
symbol = st.selectbox("銘柄コードを選択", ["7203.T", "6758.T", "9984.T", "9432.T", "8306.T"])
days = st.slider("予測日数（営業日）", 5, 30, 15)

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="1y")
    df.dropna(inplace=True)
    return df

df = load_data(symbol)
df["Target"] = df["Close"].shift(-days)
df.dropna(inplace=True)

features = df[["Open", "High", "Low", "Close", "Volume"]]
target = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)

model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# プロット用データ
future_dates = pd.bdate_range(end=df.index[-1], periods=days + 1, closed="right")
historical_close = df["Close"].iloc[-(days * 2):-days]
historical_dates = df.index[-(days * 2):-days]

fig = go.Figure()
if len(historical_dates) == len(historical_close):
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_close, mode="lines", name="過去実績", line=dict(color="blue")))

if len(future_dates) == len(preds[:days]):
    fig.add_trace(go.Scatter(x=future_dates, y=preds[:days], mode="lines", name="未来予測", line=dict(color="orange", dash="dash")))

fig.update_layout(
    title=f"{symbol} 株価予測（{days}営業日）",
    xaxis_title="日付",
    yaxis_title="終値",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)