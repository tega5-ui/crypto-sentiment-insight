import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import ta

st.set_page_config(page_title="📈 تحليل السعر الفني", layout="wide")
st.title("📊 تحليل فني للعملات الرقمية")

# واجهة الإدخال
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start = st.date_input("📆 تاريخ البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📆 تاريخ النهاية", pd.to_datetime("2025-07-01"))

if st.button("🚀 تنفيذ التحليل"):
    try:
        df = yf.download(ticker, start=start, end=end)[['Close']].dropna()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'ds', 'Close': 'price'}, inplace=True)

        # المؤشرات الفنية
        df['EMA_7'] = df['price'].ewm(span=7).mean()
        df['SMA_14'] = df['price'].rolling(window=14).mean()
        df['RSI'] = ta.momentum.RSIIndicator(close=df['price']).rsi()
        bb = ta.volatility.BollingerBands(close=df['price'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # إشارات مبسطة
        last = df.dropna().iloc[-1]
        trend = "📈 صاعد" if last['EMA_7'] > last['SMA_14'] else "📉 هابط"
        rsi = last['RSI']
        rsi_signal = "🔴 تشبع شراء" if rsi > 70 else "🟢 تشبع بيع" if rsi < 30 else "⚪ حيادي"

        st.subheader("📊 المؤشرات الفنية الأخيرة")
        st.markdown(f"""
        - السعر الحالي: **${last['price']:.2f}**
        - EMA 7: **${last['EMA_7']:.2f}**
        - SMA 14: **${last['SMA_14']:.2f}**
        - الاتجاه العام: **{trend}**
        - RSI: **{rsi:.2f} → {rsi_signal}**
        - Bollinger Band: **{last['bb_lower']:.2f} ~ {last['bb_upper']:.2f}**
        """)

        st.subheader("📈 الرسم البياني")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ds'], df['price'], label='السعر', color='blue')
        ax.plot(df['ds'], df['EMA_7'], label='EMA 7', linestyle='--', color='orange')
        ax.plot(df['ds'], df['SMA_14'], label='SMA 14', linestyle='--', color='green')
        ax.plot(df['ds'], df['bb_upper'], label='Bollinger Upper', linestyle=':', color='gray')
        ax.plot(df['ds'], df['bb_lower'], label='Bollinger Lower', linestyle=':', color='gray')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
