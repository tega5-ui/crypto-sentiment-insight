import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="📈 تحليل السعر الفني", layout="wide")
st.title("📊 تحليل فني للعملات الرقمية + توصيات تداول")

# قائمة العملات
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))

if st.button("🚀 تنفيذ التحليل"):
    try:
        # تحميل البيانات وتنظيفها
        df = yf.download(ticker, start=start, end=end)[["Close"]].dropna().reset_index()
        df.rename(columns={"Date": "ds", "Close": "price"}, inplace=True)

        # التأكد من أن 'price' سلسلة 1D صافية
        df["price"] = pd.Series(df["price"].values.reshape(-1))

        # حساب المؤشرات الفنية
        df["EMA_7"] = df["price"].ewm(span=7).mean()
        df["SMA_14"] = df["price"].rolling(window=14).mean()
        df["RSI"] = ta.momentum.RSIIndicator(close=df["price"]).rsi()
        bb = ta.volatility.BollingerBands(close=df["price"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        # إزالة القيم الناقصة
        df = df.dropna()
        latest = df.iloc[-1]

        # إشارات فنية
        ema_val = float(latest["EMA_7"])
        sma_val = float(latest["SMA_14"])
        rsi_val = float(latest["RSI"])
        trend = "📈 صاعد" if ema_val > sma_val else "📉 هابط"
        rsi_signal = "🔴 تشبع شراء" if rsi_val > 70 else "🟢 تشبع بيع" if rsi_val < 30 else "⚪ حيادي"

        # توصية تداول تلقائية
        if rsi_val < 30 and ema_val > sma_val:
            signal = "🔼 توصية: شراء"
        elif rsi_val > 70 and ema_val < sma_val:
            signal = "🔽 توصية: بيع"
        else:
            signal = "⏸ توصية: انتظر / حيادي"

        # عرض النتائج
        st.subheader("📊 المؤشرات الفنية الأخيرة")
        st.markdown(f"""
        - السعر الحالي: **${float(latest['price']):.2f}**
        - EMA 7: **${ema_val:.2f}**
        - SMA 14: **${sma_val:.2f}**
        - RSI: **{rsi_val:.2f} → {rsi_signal}**
        - Bollinger Band: **{float(latest['bb_lower']):.2f} ~ {float(latest['bb_upper']):.2f}**
        - الاتجاه العام: **{trend}**
        - 🚦 {signal}
        """)

        # الرسم البياني
        st.subheader("📈 الرسم البياني")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["ds"], df["price"], label="السعر", color="blue")
        ax.plot(df["ds"], df["EMA_7"], label="EMA 7", linestyle="--", color="orange")
        ax.plot(df["ds"], df["SMA_14"], label="SMA 14", linestyle="--", color="green")
        ax.plot(df["ds"], df["bb_upper"], label="Bollinger Upper", linestyle=":", color="gray")
        ax.plot(df["ds"], df["bb_lower"], label="Bollinger Lower", linestyle=":", color="gray")
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
