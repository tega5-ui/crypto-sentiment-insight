import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ta
import requests
import datetime

st.set_page_config(page_title="📈 التحليل الفني مع إشارات الدخول", layout="wide")
st.title("💹 نظام التحليل الفني - بيانات مباشرة + إشارات دخول")

# خريطة الرموز لـ CoinGecko
symbol_map = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "ADA-USD": "cardano"
}

tickers = list(symbol_map.keys())
ticker = st.selectbox("🪙 اختر العملة:", tickers, index=0)
symbol_id = symbol_map[ticker]

# التاريخ
start_date = st.date_input("📅 تاريخ البداية", datetime.date(2023, 1, 1))
end_date = st.date_input("📅 تاريخ النهاية", datetime.date.today())

# السعر اللحظي
def get_price(symbol="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        return requests.get(url).json()[symbol]["usd"]
    except:
        return None

# التحميل والتحليل
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df[["Close"]].dropna().rename(columns={"Close": "price"}).reset_index()

if st.button("🚀 تحليل الآن"):
    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning("🚫 لا توجد بيانات!")
        st.stop()

    df["EMA_7"] = df["price"].ewm(span=7).mean()
    df["EMA_14"] = df["price"].ewm(span=14).mean()
    rsi_series = pd.Series(df["price"].values, index=df.index)
    df["RSI"] = ta.momentum.RSIIndicator(close=rsi_series).rsi()
    bb = ta.volatility.BollingerBands(close=rsi_series)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    latest = df.iloc[-1]

    # إشارات الدخول
    entry_points = df[(df["RSI"] < 30) & (df["EMA_7"] > df["EMA_14"])]
    df["Entry"] = 0
    df.loc[entry_points.index, "Entry"] = 1

    # السعر اللحظي
    live_price = get_price(symbol_id)
    st.subheader("💲 السعر اللحظي")
    if live_price:
        st.metric("السعر من CoinGecko", f"${live_price:,.2f}")
    else:
        st.warning("⚠️ تعذر جلب السعر اللحظي.")

    # عرض المؤشرات
    st.subheader("📊 المؤشرات الفنية")
    st.markdown(f"""
    - السعر الأخير: **${latest['price']:.2f}**
    - EMA 7: **${latest['EMA_7']:.2f}**
    - EMA 14: **${latest['EMA_14']:.2f}**
    - RSI: **{latest['RSI']:.2f}**
    - Bollinger Band: **{latest['BB_lower']:.2f} ~ {latest['BB_upper']:.2f}**
    """)

    # مناطق الدخول
    st.subheader("📍 إشارات دخول مكتشفة")
    if not entry_points.empty:
        for idx, row in entry_points.iterrows():
            st.success(f"✅ دخول محتمل يوم {row['Date'].date()} عند ${row['price']:.2f}")
    else:
        st.info("⏸ لا توجد إشارات دخول قوية حاليًا.")

    # رسم تفاعلي مع إشارات دخول
    st.subheader("📈 الرسم البياني التفاعلي")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["price"], name="السعر", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_7"], name="EMA 7", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_14"], name="EMA 14", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="Bollinger Upper", line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="Bollinger Lower", line=dict(dash="dot", color="gray")))

    # سهم دخول
    entries = df[df["Entry"] == 1]
    fig.add_trace(go.Scatter(
        x=entries["Date"], y=entries["price"],
        mode="markers",
        marker=dict(symbol="arrow-up", size=12, color="lime"),
        name="📍 دخول"))

    fig.update_layout(height=600, xaxis_title="التاريخ", yaxis_title="السعر (USD)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
