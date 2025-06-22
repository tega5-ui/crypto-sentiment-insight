import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import datetime

st.set_page_config(page_title="📈 التحليل الفني للعملات الرقمية", layout="wide", page_icon="💹")
st.title("💹 نظام التحليل الفني للعملات الرقمية")

@st.cache_data
def load_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        return None

tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
start_date = st.date_input("📆 تاريخ البداية", datetime.date(2023, 1, 1))
end_date = st.date_input("📆 تاريخ النهاية", datetime.date.today())

if st.button("🚀 تنفيذ التحليل"):
    try:
        df = load_data(ticker, start_date, end_date)
        if df is None or df.empty:
            st.error("⚠️ لا توجد بيانات متاحة")
            st.stop()

        df = df[["Close"]].dropna().reset_index()
        df.rename(columns={"Date": "ds", "Close": "price"}, inplace=True)
        price_series = df["price"].values.flatten()
        index = df.index

        df["EMA_7"] = pd.Series(price_series, index=index).ewm(span=7).mean()
        df["EMA_14"] = pd.Series(price_series, index=index).ewm(span=14).mean()
        df["SMA_20"] = pd.Series(price_series, index=index).rolling(20).mean()

        rsi = ta.momentum.RSIIndicator(close=pd.Series(price_series, index=index), window=14)
        df["RSI"] = rsi.rsi()

        bb = ta.volatility.BollingerBands(close=pd.Series(price_series, index=index), window=20)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()

        macd = ta.trend.MACD(close=pd.Series(price_series, index=index))
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        df = df.dropna()
        latest = df.iloc[-1]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("السعر الحالي", f"${float(latest['price']):.2f}")
            st.metric("EMA 7", f"${float(latest['EMA_7']):.2f}")
        with col2:
            st.metric("EMA 14", f"${float(latest['EMA_14']):.2f}")
            st.metric("SMA 20", f"${float(latest['SMA_20']):.2f}")
        with col3:
            st.metric("RSI", f"{float(latest['RSI']):.2f}")
            st.metric("نطاق بولينجر", f"{float(latest['BB_lower']):.2f} ~ {float(latest['BB_upper']):.2f}")
        # 📈 رسم المؤشرات
        st.subheader("📈 الرسم البياني")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["ds"], df["price"], label="السعر", color="blue")
        ax.plot(df["ds"], df["EMA_7"], label="EMA 7", linestyle="--", color="orange")
        ax.plot(df["ds"], df["EMA_14"], label="EMA 14", linestyle="--", color="green")
        ax.plot(df["ds"], df["SMA_20"], label="SMA 20", linestyle="-.", color="purple")
        ax.fill_between(df["ds"], df["BB_lower"], df["BB_upper"], alpha=0.1, color="gray", label="نطاق بولينجر")
        ax.legend()
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        st.pyplot(fig)

        # 📉 MACD
        st.subheader("📉 مؤشر MACD")
        fig_macd, ax_macd = plt.subplots(figsize=(14, 3))
        ax_macd.plot(df["ds"], df["MACD"], label="MACD", color="blue")
        ax_macd.plot(df["ds"], df["MACD_signal"], label="خط الإشارة", color="red")
        ax_macd.axhline(0, color="gray", linestyle="--")
        ax_macd.legend()
        st.pyplot(fig_macd)

        # 🔔 إشارات تداول تلقائية
        st.subheader("🚦 توصيات تداول")
        ema7 = float(latest["EMA_7"])
        ema14 = float(latest["EMA_14"])
        rsi_val = float(latest["RSI"])
        price = float(latest["price"])

        if rsi_val < 30 and ema7 > ema14:
            signal = "🔼 توصية: شراء"
        elif rsi_val > 70 and ema7 < ema14:
            signal = "🔽 توصية: بيع"
        else:
            signal = "⏸ توصية: انتظر / حيادي"

        st.markdown(f"### {signal}")

        # 🔔 تنبيهات فنية إضافية
        st.subheader("📌 تنبيهات فنية")
        if rsi_val > 70:
            st.warning("⚠️ RSI في منطقة تشبع شرائي")
        elif rsi_val < 30:
            st.info("ℹ️ RSI في منطقة تشبع بيعي")

        if price > float(latest["BB_upper"]):
            st.warning("⚠️ السعر عند الحد الأعلى لنطاق بولينجر")
        elif price < float(latest["BB_lower"]):
            st.success("💡 السعر عند الحد الأدنى لنطاق بولينجر — فرصة شراء محتملة")

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
