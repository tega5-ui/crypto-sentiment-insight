import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import ta

st.set_page_config(page_title="📈 توقع السعر باستخدام ARIMA", layout="wide")
st.title("🔮 توقع السعر الفني باستخدام ARIMA والتحليل الفني")

tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))
forecast_days = st.radio("🔮 عدد الأيام المستقبلية:", [5, 14, 30], horizontal=True)

if st.button("🚀 شغّل التحليل"):
    try:
        # تحميل البيانات
        df = yf.download(ticker, start=start, end=end)[['Close']]
        df.reset_index(inplace=True)
        df.rename(columns={'Close': 'price'}, inplace=True)

        # المؤشرات الفنية
        df['EMA_7'] = df['price'].ewm(span=7).mean()
        df['SMA_7'] = df['price'].rolling(window=7).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close=df['price']).rsi()
        bb = ta.volatility.BollingerBands(close=df['price'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # تحويل السعر إلى Series 1D آمن
        price_series = df['price']

        # تدريب نموذج ARIMA
        model = ARIMA(price_series, order=(3, 1, 1))
        fitted = model.fit()
        raw_forecast = fitted.forecast(steps=forecast_days)

        # تقليم التوقعات ضمن حدود منطقية
        last_price = price_series.iloc[-1]
        lower = last_price * 0.85
        upper = last_price * 1.15
        forecast_array = np.clip(raw_forecast.to_numpy().reshape(-1), lower, upper)

        forecast_dates = pd.date_range(start=price_series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            'التاريخ': forecast_dates,
            'السعر المتوقع': forecast_array.round(2),
            'المقارنة الحالية': ['📈 أعلى' if val > last_price else '📉 أقل' for val in forecast_array]
        })

        # العرض النهائي
        st.info(f"🎯 السعر الحالي: ${last_price:,.2f}")
        st.subheader(f"📅 توقع السعر لـ {forecast_days} يومًا قادمة")
        st.dataframe(forecast_df)

        st.subheader("📈 الرسم البياني والتحليل الفني")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'], df['price'], label="السعر", color='blue')
        ax.plot(df['Date'], df['EMA_7'], label="EMA 7", linestyle='--', color='orange')
        ax.plot(df['Date'], df['SMA_7'], label="SMA 7", linestyle='--', color='green')
        ax.plot(df['Date'], df['bb_upper'], label="Bollinger Upper", linestyle=':', color='gray')
        ax.plot(df['Date'], df['bb_lower'], label="Bollinger Lower", linestyle=':', color='gray')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 المؤشرات الفنية الأخيرة")
        latest = df.dropna().iloc[-1]
        rsi = latest['RSI']
        rsi_status = "📈 تشبع شراء" if rsi > 70 else "📉 تشبع بيع" if rsi < 30 else "⚖️ حيادي"
        st.markdown(f"""
        - RSI: **{rsi:.2f}** → {rsi_status}
        - Bollinger Band: **{latest['bb_lower']:.2f} ~ {latest['bb_upper']:.2f}**
        """)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
