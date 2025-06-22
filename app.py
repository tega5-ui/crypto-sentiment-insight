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
        
        # التحويل الصحيح إلى سلسلة أحادية البعد
        price_series = df["price"].values.flatten()  # أو .squeeze()
        
        # حساب المؤشرات الفنية باستخدام السلسلة 1D
        df["EMA_7"] = pd.Series(price_series).ewm(span=7).mean().values
        df["SMA_14"] = pd.Series(price_series).rolling(window=14).mean().values
        
        # حساب RSI مع التحقق من الأبعاد
        rsi_calculator = ta.momentum.RSIIndicator(close=pd.Series(price_series))
        df["RSI"] = rsi_calculator.rsi().values
        
        # حساب Bollinger Bands
        bb = ta.volatility.BollingerBands(close=pd.Series(price_series))
        df["bb_upper"] = bb.bollinger_hband().values
        df["bb_lower"] = bb.bollinger_lband().values

        df = df.dropna()
        latest = df.iloc[-1]

        # إشارات فنية (باستخدام القيم الفردية)
        ema_val = float(latest["EMA_7"])
        sma_val = float(latest["SMA_14"])
        trend = "📈 صاعد" if ema_val > sma_val else "📉 هابط"
        
        rsi_val = float(latest["RSI"])
        rsi_signal = "🔴 تشبع شراء" if rsi_val > 70 else "🟢 تشبع بيع" if rsi_val < 30 else "⚪ حيادي"

        # عرض النتائج
        st.subheader("📊 المؤشرات الفنية الأخيرة")
        st.markdown(f"""
        - السعر الحالي: **${float(latest['price']):.2f}**
        - EMA 7: **${ema_val:.2f}**
        - SMA 14: **${sma_val:.2f}**
        - RSI: **{rsi_val:.2f} → {rsi_signal}**
        - Bollinger Band: **{float(latest['bb_lower']):.2f} ~ {float(latest['bb_upper']):.2f}**
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
