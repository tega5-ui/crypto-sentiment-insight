import streamlit as st
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="📈 تحليل لحظي من Binance", layout="wide")
st.title("💹 تحليل فني لحظي + مناطق دخول من Binance")

# 🧠 إعدادات المستخدم
symbols = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT",
    "SOL/USDT": "SOLUSDT"
}
symbol_name = st.selectbox("🪙 اختر العملة:", list(symbols.keys()))
binance_symbol = symbols[symbol_name]
interval = st.selectbox("⏱️ الإطار الزمني:", ["1m", "5m", "15m", "1h"], index=0)
limit = st.slider("📊 عدد الشموع:", 50, 1000, 200)

# 🛰️ جلب بيانات Binance
def get_binance_ohlcv(symbol, interval="1m", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["close"].astype(float)
    return df[["ds", "price"]].copy()

# 🚀 تنفيذ التحليل
if st.button("🚀 تحليل الآن"):
    try:
        df = get_binance_ohlcv(binance_symbol, interval, limit)
        df["EMA_7"] = df["price"].ewm(span=7).mean()
        df["EMA_14"] = df["price"].ewm(span=14).mean()
        df["RSI"] = ta.momentum.RSIIndicator(close=df["price"]).rsi()
        bb = ta.volatility.BollingerBands(close=df["price"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df = df.dropna()
        latest = df.iloc[-1]

        # 💡 مناطق الدخول
        entry_signals = []
        if latest["RSI"] < 30 and latest["EMA_7"] > latest["EMA_14"]:
            entry_signals.append(f"✅ دخول محتمل عند السعر: ${latest['price']:.2f} (RSI منخفض و EMA صاعد)")
        if latest["price"] < latest["BB_lower"]:
            entry_signals.append(f"📉 السعر دون نطاق بولينجر → دخول محتمل عند: ${latest['price']:.2f}")
        if not entry_signals:
            entry_signals.append("⏸ لا توجد إشارات دخول قوية حاليًا.")

        # 📊 عرض المؤشرات
        st.subheader("📊 المؤشرات الفنية")
        st.markdown(f"""
        - السعر الحالي: **${latest['price']:.2f}**
        - EMA 7: **${latest['EMA_7']:.2f}**
        - EMA 14: **${latest['EMA_14']:.2f}**
        - RSI: **{latest['RSI']:.2f}**
        - Bollinger Band: **{latest['BB_lower']:.2f} ~ {latest['BB_upper']:.2f}**
        """)

        # 📍 مناطق الدخول
        st.subheader("📍 مناطق دخول محتملة")
        for signal in entry_signals:
            st.success(signal)

        # 📈 الرسم البياني
        st.subheader("📈 الرسم البياني")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["ds"], df["price"], label="السعر", color="blue")
        ax.plot(df["ds"], df["EMA_7"], label="EMA 7", linestyle="--", color="orange")
        ax.plot(df["ds"], df["EMA_14"], label="EMA 14", linestyle="--", color="green")
        ax.plot(df["ds"], df["BB_upper"], label="Bollinger Upper", linestyle=":", color="gray")
        ax.plot(df["ds"], df["BB_lower"], label="Bollinger Lower", linestyle=":", color="gray")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
