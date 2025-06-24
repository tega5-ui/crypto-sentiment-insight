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

# اختيار التاريخ
start_date = st.date_input("📅 تاريخ البداية", datetime.date(2023, 1, 1))
end_date = st.date_input("📅 تاريخ النهاية", datetime.date.today())

# دالة السعر اللحظي
def get_price(symbol="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        return requests.get(url).json()[symbol]["usd"]
    except:
        return None

# تحميل البيانات
@st.cache_data
def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    df = df[["Close"]].dropna().rename(columns={"Close": "price"}).reset_index()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.dropna(subset=["price"], inplace=True)
    return df

# تنفيذ التحليل
if st.button("🚀 تحليل الآن"):
    df = get_data(ticker, start_date, end_date)
    if df.empty:
        st.warning("🚫 لا توجد بيانات متاحة.")
        st.stop()

    # المؤشرات الفنية
    df["EMA_7"] = df["price"].ewm(span=7).mean()
    df["EMA_14"] = df["price"].ewm(span=14).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close=df["price"]).rsi()
    bb = ta.volatility.BollingerBands(close=df["price"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df.dropna(inplace=True)
    if df.empty:
        st.warning("🚫 لا توجد بيانات كافية بعد الحسابات الفنية.")
        st.stop()

    latest = df.iloc[-1]

    # إشارات الدخول
    df["Entry"] = ((df["RSI"] < 30) & (df["EMA_7"] > df["EMA_14"])).astype(int)
    entry_points = df[df["Entry"] == 1]

    # السعر اللحظي
    st.subheader("💲 السعر اللحظي")
    live_price = get_price(symbol_id)
    if live_price:
        st.metric("السعر من CoinGecko", f"${live_price:,.2f}")
    else:
        st.warning("⚠️ تعذر جلب السعر اللحظي.")

    # المؤشرات
    st.subheader("📊 المؤشرات الفنية")
    st.markdown(f"""
    - السعر الأخير: **${latest['price']:.2f}**
    - EMA 7: **${latest['EMA_7']:.2f}**
    - EMA 14: **${latest['EMA_14']:.2f}**
    - RSI: **{latest['RSI']:.2f}**
    - Bollinger Band: **{latest['BB_lower']:.2f} ~ {latest['BB_upper']:.2f}**
    """)

    # توصيات الدخول
    st.subheader("📍 إشارات دخول")
    if not entry_points.empty:
        for idx, row in entry_points.iterrows():
            st.success(f"✅ دخول محتمل عند ${row['price']:.2f} بتاريخ {row['Date'].date()}")
    else:
        st.info("⏸ لا توجد إشارات دخول حالياً.")

    # رسم تفاعلي
    st.subheader("📈 الرسم البياني التفاعلي")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["price"], name="السعر", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_7"], name="EMA 7", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_14"], name="EMA 14", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="Bollinger Upper", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="Bollinger Lower", line=dict(color="gray", dash="dot")))

    # سهم دخول
    entries = df[df["Entry"] == 1]
    fig.add_trace(go.Scatter(
        x=entries["Date"],
        y=entries["price"],
        mode="markers",
        name="📍 دخول",
        marker=dict(symbol="arrow-up", size=12, color="lime")
    ))

    fig.update_layout(height=600, xaxis_title="التاريخ", yaxis_title="السعر (USD)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
