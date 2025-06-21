import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import ta
from datetime import datetime, timedelta

# إعداد الصفحة
st.set_page_config(page_title="توقع السعر الفني", layout="wide", page_icon="📈")
st.title("📊 توقع السعر باستخدام ARIMA والتحليل الفني")

# تحميل البيانات (تُخزن مؤقتًا)
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# العملات المتاحة
tickers = {
    "BTC-USD": "بتكوين",
    "ETH-USD": "إيثريوم",
    "ADA-USD": "كاردانو"
}

# واجهة المستخدم
with st.sidebar:
    st.header("الإعدادات")
    ticker = st.selectbox("العملة", list(tickers.keys()), format_func=lambda x: f"{x} ({tickers[x]})")
    start = st.date_input("تاريخ البداية", pd.to_datetime("2023-01-01"))
    end = st.date_input("تاريخ النهاية", datetime.now())
    forecast_days = st.slider("عدد أيام التنبؤ", 1, 30, 14)
    st.markdown("---")

# عند الضغط على زر التحليل
if st.button("ابدأ التحليل", use_container_width=True):
    with st.spinner("جاري التحليل..."):
        try:
            df = load_data(ticker, start, end)
            if df.empty:
                st.error("⚠️ لا توجد بيانات!")
                st.stop()

            df = df[['Close']].copy()
            df.reset_index(inplace=True)
            df.rename(columns={'Close': 'price'}, inplace=True)

            # حساب المؤشرات الفنية
            df['EMA_14'] = df['price'].ewm(span=14, adjust=False).mean()
            df['RSI_14'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
            bb = ta.volatility.BollingerBands(close=df['price'], window=14)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            macd = ta.trend.MACD(close=df['price'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()

            # تدريب نموذج ARIMA بمعاملات ثابتة (3,1,1)
            price_series = df['price'].dropna()
            model = ARIMA(price_series, order=(3, 1, 1))
            fitted = model.fit()

            # التنبؤ
            forecast = fitted.forecast(steps=forecast_days)
            forecast_values = forecast.values.flatten()
            last_price = price_series.iloc[-1]
            volatility = price_series.pct_change().std()
            forecast_values = np.clip(
                forecast_values,
                last_price * (1 - 2 * volatility),
                last_price * (1 + 2 * volatility)
            )

            forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({
                'التاريخ': forecast_dates,
                'السعر المتوقع': forecast_values.round(2),
                'النسبة المئوية': ((forecast_values / last_price - 1) * 100).round(2),
                'الإشارة': np.where(forecast_values > last_price, '📈 صعود', '📉 هبوط')
            })

            # عرض النتائج
            st.success("✅ التحليل مكتمل")
            st.dataframe(forecast_df, use_container_width=True)

            # رسم بياني
            st.subheader("الرسم البياني")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Date'], df['price'], label="السعر", color="blue")
            ax.plot(df['Date'], df['EMA_14'], label="EMA 14", linestyle="--")
            ax.fill_between(df['Date'], df['bb_lower'], df['bb_upper'], color='gray', alpha=0.1)
            ax.plot(forecast_dates, forecast_values, 'ro--', label="التنبؤ")
            ax.legend()
            st.pyplot(fig)

            # ملخص
            st.subheader("ملخص التحليل")
            last = df.iloc[-1]
            rsi = last['RSI_14']
            summary = f"""
            - **اتجاه السعر**: {'⬆️ صاعد' if last['price'] > last['EMA_14'] else '⬇️ هابط'}
            - **RSI:** {rsi:.1f} → {'تشبع شراء 🔴' if rsi > 70 else 'تشبع بيع 🟢' if rsi < 30 else 'محايد 🟡'}
            - **MACD:** {'🟢 إيجابي' if last['MACD'] > last['MACD_signal'] else '🔴 سلبي'}
            """
            st.markdown(summary)

            # التحميل
            st.download_button("📥 تحميل التوقعات", forecast_df.to_csv(index=False).encode('utf-8'), file_name="forecast.csv", mime="text/csv")

        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
