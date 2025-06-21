import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import ta

st.set_page_config(page_title="📈 توقع السعر الفني", layout="wide")
st.title("📊 توقع السعر المستقبلي بالتحليل الفني فقط (بدون مشاعر)")

# اختيار العملة والفترة
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("🗓️ بداية الفترة", pd.to_datetime("2023-01-01"))
end = st.date_input("🗓️ نهاية الفترة", pd.to_datetime("2025-07-01"))
forecast_days = st.radio("🔮 عدد الأيام المستقبلية:", [5, 14, 30], horizontal=True)

if st.button("🚀 تشغيل التحليل الفني"):
    try:
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        df = df[["Date", "Close"]].rename(columns={"Close": "price"})

        # الحسابات الفنية
        df['SMA_7'] = df['price'].rolling(window=7).mean()
        df['EMA_7'] = df['price'].ewm(span=7).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['price']).rsi()
        bb = ta.volatility.BollingerBands(close=df['price'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # توقع ARIMA
        model = ARIMA(df['price'], order=(3,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'التاريخ': forecast_dates, 'السعر المتوقع': forecast.values})

        # الرسم البياني
        st.subheader("📈 السعر والتحليل الفني")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Date'], df['price'], label="السعر الفعلي", color='blue')
        ax.plot(df['Date'], df['EMA_7'], label="EMA 7", linestyle="--", color='orange')
        ax.plot(df['Date'], df['SMA_7'], label="SMA 7", linestyle="--", color='green')
        ax.plot(df['Date'], df['bb_upper'], label="Bollinger Upper", linestyle=":", color='gray')
        ax.plot(df['Date'], df['bb_lower'], label="Bollinger Lower", linestyle=":", color='gray')
        ax.legend()
        st.pyplot(fig)

        # جدول التوقع المستقبلي
        st.subheader(f"📅 جدول توقع السعر لـ {forecast_days} يومًا قادمة")
        forecast_df['السعر المتوقع'] = forecast_df['السعر المتوقع'].round(2)
        st.dataframe(forecast_df)

        # جدول المؤشرات الفنية لليوم الأخير
        latest = df.dropna().iloc[-1]
        st.subheader("📊 آخر قراءة للمؤشرات الفنية:")
        st.markdown(f"""
        - السعر الحالي: **${latest['price']:.2f}**
        - EMA 7: **${latest['EMA_7']:.2f}**
        - RSI: **{latest['RSI']:.2f}** → {"📈 تشبع شراء" if latest['RSI'] > 70 else "📉 تشبع بيع" if latest['RSI'] < 30 else "⚖️ حيادي"}
        - Bollinger Band: **{latest['bb_lower']:.2f} ~ {latest['bb_upper']:.2f}**
        """)

    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
