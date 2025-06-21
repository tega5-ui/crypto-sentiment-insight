import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل العملات والمشاعر", layout="wide")
st.title("📊 تحليل سعر العملة مقابل مشاعر الأخبار")

# العملات الشائعة
popular_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD",
    "SOL-USD", "XRP-USD", "DOGE-USD", "AVAX-USD",
    "MATIC-USD", "GALA-USD", "أخرى..."
]
st.markdown("### 🪙 اختر عملة أو أدخل رمزًا يدويًا")
selected = st.selectbox("🔽 العملات المتاحة:", popular_tickers)
ticker = st.text_input("✍️ أدخل الرمز يدويًا:", "") if selected == "أخرى..." else selected

# التاريخ
start = st.date_input("📅 من:", pd.to_datetime("2024-01-01"))
end = st.date_input("📅 إلى:", pd.to_datetime("2025-07-01"))

if ticker and st.button("🚀 شغّل التحليل"):
    try:
        # تحميل البيانات
        price_data = yf.download(ticker, start=start, end=end)[['Close']]
        price_data.reset_index(inplace=True)
        price_data.columns = ['Date', 'price']  # تصحيح الأسماء بشكل مسطّح
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        # التأكد من البنية المسطّحة للأعمدة
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = [col[0] if isinstance(col, tuple) else col for col in price_data.columns]

        # تحليل الأخبار
        def fetch_google_news_rss(query):
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+crypto&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            entries = [{'title': e.title, 'description': e.description, 'date': e.published} for e in feed.entries]
            df = pd.DataFrame(entries)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df

        def analyze_sentiment(text):
            return TextBlob(text).sentiment.polarity if text else 0

        news_df = fetch_google_news_rss(ticker.split('-')[0])
        news_df['sentiment_score'] = news_df['description'].apply(analyze_sentiment)
        news_df['Date'] = pd.to_datetime(news_df['date'])
        daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean().reset_index()

        # الدمج
        merged = pd.merge(price_data, daily_sentiment, on='Date', how='left')
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        # التنبؤ
        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        # الرسم
        st.subheader("📈 السعر الفعلي مقابل السعر المتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged['Date'], merged['price'], label="السعر الفعلي", color='blue')
        ax.plot(merged['Date'], merged['predicted_price'], label="السعر المتوقع", color='green')
        ax.set_xlabel("📅 التاريخ")
        ax.set_ylabel("💰 السعر")
        ax.legend()
        st.pyplot(fig)

        # معامل الارتباط
        corr = merged['price'].corr(merged['sentiment_score'])
        st.success(f"💡 معامل الارتباط بين السعر والمشاعر: {corr:.3f}")

    except Exception as e:
        st.error(f"🚨 حدث خطأ أثناء التحليل:\n\n{str(e)}")
