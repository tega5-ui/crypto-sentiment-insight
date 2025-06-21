import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل العملات والمشاعر", layout="wide")
st.title("📊 تحليل سعر العملة مقابل مشاعر الأخبار")

# واجهة اختيار العملة
st.markdown("### 🪙 اختر عملة أو اكتب رمزًا يدويًا")
popular_tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD",
    "SOL-USD", "XRP-USD", "DOGE-USD", "AVAX-USD",
    "MATIC-USD", "GALA-USD", "أخرى..."
]
selected = st.selectbox("🔽 العملات الشائعة:", popular_tickers)
if selected == "أخرى...":
    ticker = st.text_input("✍️ أدخل الرمز يدويًا:", "")
else:
    ticker = selected

start = st.date_input("📅 البداية", pd.to_datetime("2024-01-01"))
end = st.date_input("📅 النهاية", pd.to_datetime("2025-07-01"))

if ticker and st.button("🚀 شغّل التحليل"):
    try:
        # تحميل البيانات المالية
        price_data = yf.download(ticker, start=start, end=end)[['Close']].copy()
        price_data.reset_index(inplace=True)
        price_data.columns.name = None  # إزالة أي أسماء للأعمدة من الفهرس
        price_data.rename(columns={'Close': 'price'}, inplace=True)
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        # معالجة الفهرسة: إزالة MultiIndex إن وجد
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = ['_'.join(col).strip() for col in price_data.columns.values]
            price_data.reset_index(inplace=True)

        # جلب الأخبار
        def fetch_google_news_rss(query):
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            articles = [{'title': entry.title, 'description': entry.description, 'date': entry.published} for entry in feed.entries]
            df = pd.DataFrame(articles)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df

        def analyze_sentiment(text):
            if not text:
                return 0
            return TextBlob(text).sentiment.polarity

        news_query = ticker.split('-')[0] + " crypto"
        news_df = fetch_google_news_rss(news_query)
        news_df['sentiment_score'] = news_df['description'].apply(analyze_sentiment)
        news_df['Date'] = pd.to_datetime(news_df['date'])

        # مشاعر يومية
        daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean().reset_index()
        daily_sentiment.columns.name = None
        if isinstance(daily_sentiment.columns, pd.MultiIndex):
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
            daily_sentiment.reset_index(inplace=True)

        # الدمج والتحليل
        merged = pd.merge(price_data, daily_sentiment, on='Date', how='left')
        merged['sentiment_score'].fillna(0, inplace=True)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        # التنبؤ
        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        # الرسم البياني
        st.subheader("📈 السعر الفعلي مقابل المتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged['Date'], merged['price'], label="السعر الفعلي", color='blue')
        ax.plot(merged['Date'], merged['predicted_price'], label="السعر المتوقع", color='green')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        # الارتباط
        corr = merged['price'].corr(merged['sentiment_score'])
        st.success(f"💡 معامل الارتباط بين السعر والمشاعر: {corr:.3f}")

    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {str(e)}", icon="🚨")
