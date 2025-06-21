import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import ta

st.set_page_config(page_title="📈 توقع السعر الفني", layout="wide")
st.title("📊 توقع السعر المستقبلي باستخدام ARIMA والتحليل الفني")

tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))
forecast_days = st.radio("🔮 عدد الأيام المستقبلية:", [5, 14, 30], horizontal=True)

if st.button("🚀 شغّل التحليل"):
    try:
        df = yf.download(ticker, start=start, end=end)[['Close']]
        df.reset_index(inplace=True)
        df.rename(columns={'Close': 'price'}, inplace=True)

        # التحليل الفني
        df['EMA_7'] = df['price'].ewm(span=7).mean()
        df['SMA_7'] = df['price'].rolling(window=7).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close=df['price']).rsi()
        bb = ta.volatility.BollingerBands(close=df['price'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # تدريب نموذج ARIMA
        model = ARIMA(df['price'], order=(3,1,1))
        fitted = model.fit()
        raw_forecast = fitted.forecast(steps=forecast_days)

        # ضبط الحدود المنطقية للتوقع
        last_price = df['price'].iloc[-1]
        lower_bound = last_price * 0.85
        upper_bound = last_price * 1.15
        clipped_forecast = raw_forecast.clip(lower=lower_bound, upper=upper_bound)

        # تحويل إلى مصفوفة 1D
        clipped_array = np.ravel(np.array(clipped_forecast))

        forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            'التاريخ': forecast_dates,
            'السعر المتوقع': clipped_array.round(2),
            'المقارنة الحالية': ['📈 أعلى' if x > last_price else '📉 أقل' for x in clipped_array]
        })

        # مرجعية السعر الحالية
        ema_now = df['EMA_7'].iloc[-1]
        st.info(f"🎯 السعر الحالي: ${last_price:,.2f} | المتوسط EMA 7: ${ema_now:,.2f}")

        # عرض الجدول
        st.subheader(f"📅 توقع السعر لـ {forecast_days} يومًا قادمة")
        st.dataframe(forecast_df)

        # الرسم البياني الفني
        st.subheader("📈 السعر والتحليل الفني")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'], df['price'], label="السعر الفعلي", color='blue')
        ax.plot(df['Date'], df['EMA_7'], label="EMA 7", linestyle="--", color='orange')
        ax.plot(df['Date'], df['SMA_7'], label="SMA 7", linestyle="--", color='green')
        ax.plot(df['Date'], df['bb_upper'], linestyle=":", label="Bollinger Upper", color='gray')
        ax.plot(df['Date'], df['bb_lower'], linestyle=":", label="Bollinger Lower", color='gray')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        # تقييم المؤشرات الفنية الأخيرة
        st.subheader("📊 تقييم آخر المؤشرات")
        latest = df.dropna().iloc[-1]
        st.markdown(f"""
        - السعر الحالي: **${latest['price']:.2f}**
        - RSI: **{latest['RSI']:.2f}** → {"📈 تشبع شراء" if latest['RSI'] > 70 else "📉 تشبع بيع" if latest['RSI'] < 30 else "⚖️ حيادي"}
        - نطاق Bollinger: **{latest['bb_lower']:.2f} ~ {latest['bb_upper']:.2f}**
        """)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
