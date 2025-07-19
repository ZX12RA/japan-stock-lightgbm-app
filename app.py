
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split

st.title("日本株 XGBoost 予測（スクロール可能グラフ）")

symbol = st.selectbox("銘柄を選択", ["7203.T", "6758.T", "9984.T", "9432.T", "8306.T"])
days = st.slider("予測営業日数", 1, 30, 7)

@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="3y")
    df = df.reset_index()
    return df

def prepare_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df = df.dropna()
    df['Target'] = df['Close'].shift(-days)
    df = df.dropna()
    return df

def train_and_predict(df):
    features = df[['Close', 'Return', 'Volatility']]
    target = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    future = features.iloc[-days:]
    preds = model.predict(future)
    return df['Date'].iloc[:-days], df['Close'].iloc[:-days], df['Date'].iloc[-days:], preds

df = load_data(symbol)
df = prepare_features(df)

dates_past, close_past, future_dates, preds = train_and_predict(df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates_past, y=close_past, mode='lines', name='過去実績', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines', name='未来予測', line=dict(color='orange', dash='dash')))
fig.add_vline(x=future_dates.iloc[0], line=dict(color='gray', dash='dot'))

fig.update_layout(
    title=f"{symbol} 株価予測（{days}営業日）",
    xaxis_title="日付",
    yaxis_title="終値",
    hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True), type="date"),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
