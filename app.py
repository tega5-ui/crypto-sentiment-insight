import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta

st.set_page_config(page_title="📈 تحليل السعر الفني", layout="wide")
st.title("📊 تحليل فني للعملات الرقمية")

# واجهة المستخدم
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))

if st.button("🚀 تنفيذ التحليل"):
    try:
        # تحميل البيانات
        df = yf.download(ticker, start=start, end=end)[["Close"]].dropna().reset_index()
        df.rename(columns={"Date": "ds", "Close": "price"}, inplace=True)
        
        # تحويل السعر إلى Series 1D
        price_series = df["price"].squeeze()
        
        # حساب المؤشرات الفنية
        df["EMA_7"] = price_series.ewm(span=7).mean()
        df["SMA_14"] = price_series.rolling(window=14).mean()
        df["RSI"] = ta.momentum.RSIIndicator(close=price_series).rsi()
        bb = ta.volatility.BollingerBands(close=price_series)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()

        df = df.dropna()
        latest = df.iloc[-1]

        # إشارات فنية (معالجة المشكلة باستخدام .item())
        ema_value = latest["EMA_7"].item() if pd.notna(latest["EMA_7"]) else 0
        sma_value = latest["SMA_14"].item() if pd.notna(latest["SMA_14"]) else 0
        trend = "📈 صاعد" if ema_value > sma_value else "📉 هابط"
        
        rsi_value = latest["RSI"].item() if pd.notna(latest["RSI"]) else 50
        rsi_signal = "🔴 تشبع شراء" if rsi_value > 70 else "🟢 تشبع بيع" if rsi_value < 30 else "⚪ حيادي"

        # عرض النتائج
        st.subheader("📊 المؤشرات الفنية الأخيرة")
        st.markdown(f"""
        - السعر الحالي: **${latest['price'].item():.2f}**
        - EMA 7: **${ema_value:.2f}**
        - SMA 14: **${sma_value:.2f}**
        - RSI: **{rsi_value:.2f} → {rsi_signal}**
        - Bollinger Band: **{latest['bb_lower'].item():.2f} ~ {latest['bb_upper'].item():.2f}**
        - الاتجاه العام: **{trend}**
        """)

        # رسم المؤشرات
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
