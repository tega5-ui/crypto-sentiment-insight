import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import ta

st.set_page_config(page_title="🔮 توقع السعر باستخدام Prophet", layout="wide")
st.title("📈 توقع السعر باستخدام Prophet والتحليل الفني")

tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))
forecast_days = st.radio("🔮 عدد الأيام المستقبلية:", [5, 14, 30], horizontal=True)

if st.button("🚀 شغّل التنبؤ"):
    try:
        # 1. تحميل البيانات
        df = yf.download(ticker, start=start, end=end)[['Close']].dropna()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        # 2. تدريب Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        # 3. إنشاء التواريخ المستقبلية والتوقع
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # 4. تحضير الجدول النهائي للتوقعات القادمة
        latest_price = df['y'].iloc[-1]
        forecast_tail = forecast[['ds', 'yhat']].tail(forecast_days).copy()
        forecast_tail['yhat'] = forecast_tail['yhat'].squeeze()

        def categorize(x, ref):
            try:
                return "📈 أعلى" if float(x) > ref else "📉 أقل"
            except:
                return "⚠️"

        forecast_tail['المقارنة الحالية'] = forecast_tail['yhat'].apply(lambda x: categorize(x, latest_price))
        forecast_tail.rename(columns={'ds': 'التاريخ', 'yhat': 'السعر المتوقع'}, inplace=True)

        # 5. التحليل الفني
        ta_df = df.copy()
        ta_df['EMA_7'] = ta_df['y'].ewm(span=7).mean()
        ta_df['SMA_7'] = ta_df['y'].rolling(window=7).mean()
        ta_df['RSI'] = ta.momentum.RSIIndicator(close=ta_df['y']).rsi()
        bb = ta.volatility.BollingerBands(close=ta_df['y'])
        ta_df['bb_upper'] = bb.bollinger_hband()
        ta_df['bb_lower'] = bb.bollinger_lband()

        # 6. عرض الرسوم البيانية
        st.subheader("📊 توقعات Prophet")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("🧩 مكونات النموذج")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        st.subheader(f"📅 التوقع لـ {forecast_days} يومًا قادمة")
        st.dataframe(forecast_tail)

        st.subheader("📈 التحليل الفني")
        fig3, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ta_df['ds'], ta_df['y'], label='السعر الفعلي', color='blue')
        ax.plot(ta_df['ds'], ta_df['EMA_7'], label='EMA 7', linestyle="--", color='orange')
        ax.plot(ta_df['ds'], ta_df['SMA_7'], label='SMA 7', linestyle="--", color='green')
        ax.plot(ta_df['ds'], ta_df['bb_upper'], label='Bollinger Upper', linestyle=":", color='gray')
        ax.plot(ta_df['ds'], ta_df['bb_lower'], label='Bollinger Lower', linestyle=":", color='gray')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig3)

        st.subheader("📊 المؤشرات الفنية الأخيرة")
        latest = ta_df.dropna().iloc[-1]
        rsi_value = latest['RSI']
        rsi_status = "📈 تشبع شراء" if rsi_value > 70 else "📉 تشبع بيع" if rsi_value < 30 else "⚖️ حيادي"

        st.markdown(f"""
        - السعر الحالي: **${latest['y']:.2f}**
        - RSI: **{rsi_value:.2f}** → {rsi_status}
        - Bollinger Band: **{latest['bb_lower']:.2f} ~ {latest['bb_upper']:.2f}**
        """)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
