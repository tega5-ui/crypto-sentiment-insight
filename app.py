import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
import datetime
import requests

# إعداد الصفحة
st.set_page_config(page_title="📈 التحليل الفني للعملات الرقمية", layout="wide", page_icon="💹")
st.title("💹 نظام التحليل الفني اللحظي للعملات الرقمية")

# ترجمة رموز العملات من Yahoo إلى CoinGecko
symbol_map = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "ADA": "cardano",
    "BNB": "binancecoin",
    "SOL": "solana"
}

# دالة لجلب السعر اللحظي
def get_realtime_price(symbol="bitcoin", vs_currency="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies={vs_currency}"
    try:
        response = requests.get(url)
        data = response.json()
        return data[symbol][vs_currency]
    except:
        return None

# تحميل البيانات
@st.cache_data
def load_data(ticker, start, end):
    try:
        return yf.download(ticker, start=start, end=end)
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل البيانات: {e}")
        return None

# واجهة المستخدم
tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
ticker = st.selectbox("🪙 اختر العملة:", tickers)
symbol_name = symbol_map.get(ticker.split("-")[0], "bitcoin")

start = st.date_input("📆 تاريخ البداية", datetime.date(2023, 1, 1))
end = st.date_input("📆 تاريخ النهاية", datetime.date.today())

if st.button("🚀 تنفيذ التحليل"):
    try:
        df = load_data(ticker, start, end)
        if df is None or df.empty:
            st.error("⚠️ لا توجد بيانات متاحة")
            st.stop()

        df = df[["Close"]].dropna().reset_index()
        df.rename(columns={"Date": "ds", "Close": "price"}, inplace=True)
        idx = df.index
        price_series = df["price"].values.flatten()

        # المؤشرات الفنية
        df["EMA_7"] = pd.Series(price_series, index=idx).ewm(span=7).mean()
        df["EMA_14"] = pd.Series(price_series, index=idx).ewm(span=14).mean()
        df["SMA_20"] = pd.Series(price_series, index=idx).rolling(20).mean()

        rsi = ta.momentum.RSIIndicator(close=pd.Series(price_series, index=idx))
        df["RSI"] = rsi.rsi()

        bb = ta.volatility.BollingerBands(close=pd.Series(price_series, index=idx))
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()

        macd = ta.trend.MACD(close=pd.Series(price_series, index=idx))
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        df = df.dropna()
        latest = df.iloc[-1]
        price = float(latest["price"])

        # 💲 السعر اللحظي
        st.subheader("💲 السعر اللحظي")
        realtime_price = get_realtime_price(symbol_name)
        if realtime_price:
            st.metric("السعر اللحظي", f"${realtime_price:,.2f}")
        else:
            st.warning("⚠️ تعذر جلب السعر اللحظي من CoinGecko")

        # 📊 المؤشرات الفنية
        st.subheader("📊 المؤشرات الفنية")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("EMA 7", f"${float(latest['EMA_7']):.2f}")
            if float(latest["EMA_7"]) > float(latest["EMA_14"]):
                st.info("🔎 EMA 7 أعلى من EMA 14 → زخم صاعد.")
            else:
                st.info("🔎 EMA 7 أقل من EMA 14 → زخم ضعيف أو هبوط محتمل.")
        with col2:
            st.metric("RSI", f"{float(latest['RSI']):.2f}")
            if float(latest["RSI"]) > 70:
                st.warning("⚠️ RSI يشير إلى تشبع شرائي.")
            elif float(latest["RSI"]) < 30:
                st.success("✅ RSI في منطقة تشبع بيعي.")
            else:
                st.info("ℹ️ RSI في المنطقة المحايدة.")
        with col3:
            st.metric("نطاق بولينجر", f"{float(latest['BB_lower']):.2f} ~ {float(latest['BB_upper']):.2f}")
            if price > float(latest["BB_upper"]):
                st.warning("📈 السعر خارج النطاق الأعلى — احتمال تصحيح.")
            elif price < float(latest["BB_lower"]):
                st.success("📉 السعر تحت النطاق — احتمال ارتداد.")
            else:
                st.info("📊 السعر داخل نطاق بولينجر — تقلب طبيعي.")

        # 🚦 توصية تداول تلقائية
        st.subheader("🚦 توصية تداول")
        if float(latest["RSI"]) < 30 and float(latest["EMA_7"]) > float(latest["EMA_14"]):
            st.success("🔼 توصية: شراء")
        elif float(latest["RSI"]) > 70 and float(latest["EMA_7"]) < float(latest["EMA_14"]):
            st.error("🔽 توصية: بيع")
        else:
            st.info("⏸ توصية: حيادية")

        # 📈 الرسم البياني
        st.subheader("📈 الرسم البياني")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["ds"], df["price"], label="السعر", color="blue")
        ax.plot(df["ds"], df["EMA_7"], label="EMA 7", linestyle="--", color="orange")
        ax.plot(df["ds"], df["EMA_14"], label="EMA 14", linestyle="--", color="green")
        ax.plot(df["ds"], df["SMA_20"], label="SMA 20", linestyle="--", color="purple")
        ax.fill_between(df["ds"], df["BB_lower"], df["BB_upper"], alpha=0.1, label="نطاق بولينجر", color="gray")
        ax.legend()
        st.pyplot(fig)

        # 📉 MACD
        st.subheader("📉 MACD")
        fig_macd, ax_macd = plt.subplots(figsize=(14, 3))
        ax_macd.plot(df["ds"], df["MACD"], label="MACD", color="blue")
        ax_macd.plot(df["ds"], df["MACD_signal"], label="إشارة", color="red")
        ax_macd.axhline(0, color="gray", linestyle="--")
        ax_macd.legend()
        st.pyplot(fig_macd)

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
