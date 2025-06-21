import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="تحليل العملات والتوقع", layout="wide")
st.title("📊 تحليل العملات المشفرة وتوقع الأسعار بناءً على الأخبار")

# واجهة اختيار العملة
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "GALA-USD", "أخرى..."]
selected = st.selectbox("🪙 اختر العملة:", tickers)
ticker = st.text_input("✍️ أدخل الرمز يدويًا:", "") if selected == "أخرى..." else selected

# عرض السعر الحالي من CoinGecko
symbol_map = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "ADA-USD": "cardano",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
    "DOGE-USD": "dogecoin",
    "AVAX-USD": "avalanche-2",
    "MATIC-USD": "matic-network",
    "GALA-USD": "gala"
}
if ticker:
    coin_id = symbol_map.get(ticker.upper())
    if coin_id:
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            response = requests.get(url)
            if response.status_code == 200:
                price = response.json()[coin_id]['usd']
                st.metric(label="💰 السعر الحالي", value=f"${price:,.4f}")
            else:
                st.warning("⚠️ تعذر جلب السعر الحالي من CoinGecko.")
        except:
            st.warning("⚠️ حدث خطأ أثناء الاتصال بـ CoinGecko.")
    else:
        st.info("🔎 لا يمكن عرض السعر المباشر لهذه العملة حالياً.")

# تحديد الفترة
start = st.date_input("📅 البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📅 النهاية", pd.to_datetime("2025-07-01"))

# اختيار عدد الأيام المستقبلية
forecast_days = st.radio("📆 عدد الأيام للتوقع:", [5, 14, 30], horizontal=True)

if ticker and st.button("🚀 شغّل التحليل"):
    try:
        df = yf.download(ticker, start=start, end=end)[['Close']]
        df.reset_index(inplace=True)
        df.columns = ['Date', 'price']
        df['SMA_7'] = df['price'].rolling(7).mean()
        df['EMA_7'] = df['price'].ewm(span=7, adjust=False).mean()

        def fetch_news(q):
            url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}+crypto&hl=en-US&gl=US&ceid=US:en"
            entries = feedparser.parse(url).entries
            data = [{'title': e.title, 'description': e.description, 'published': e.published} for e in entries[:5]]
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['published']).dt.date
            return df

        def analyze(text):
            return TextBlob(text).sentiment.polarity if text else 0

        base = ticker.split("-")[0]
        news = fetch_news(base)
        news['sentiment_score'] = news['description'].apply(analyze)
        news['Date'] = pd.to_datetime(news['date'])
        daily_sent = news.groupby('Date')['sentiment_score'].mean().reset_index()

        merged = pd.merge(df, daily_sent, on='Date', how='left')
        merged['sentiment_score'].fillna(0, inplace=True)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        last_sentiment = merged['sentiment_score'].tail(3).mean()
        future_dates = pd.date_range(start=merged['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        future = pd.DataFrame({'Date': future_dates})
        future['lagged_sentiment'] = last_sentiment
        future['predicted_price'] = model.predict(future[['lagged_sentiment']])
        future['trend'] = future['predicted_price'].diff().apply(lambda x: "📈 صعود" if x > 0 else ("📉 نزول" if x < 0 else "— ثبات"))

        # الرسم البياني
        st.subheader("📈 تطور السعر الفعلي والتوقعات المستقبلية")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged['Date'], merged['price'], label="السعر الفعلي", color='blue')
        ax.plot(merged['Date'], merged['SMA_7'], label="SMA 7", linestyle="--", color='gray')
        ax.plot(merged['Date'], merged['EMA_7'], label="EMA 7", linestyle="--", color='purple')
        ax.plot(merged['Date'], merged['predicted_price'], label="توقع على البيانات", color='green')
        ax.plot(future['Date'], future['predicted_price'], label="توقع مستقبلي", linestyle='--', color='orange')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        # معامل الارتباط
        corr = merged['price'].corr(merged['sentiment_score'])
