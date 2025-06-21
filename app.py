import streamlit as st
import pandas as pd
import yfinance as yf
from textblob import TextBlob
import feedparser
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل العملات والمشاعر", layout="wide")
st.title("📊 تحليل سعر العملة مقابل مشاعر الأخبار")

# إدخال المستخدم للعملة
ticker = st.text_input("🔎 أدخل رمز العملة (مثلاً GALA-USD)", "GALA-USD")

# تحديد الفترة الزمنية
start = st.date_input("📅 البداية", pd.to_datetime("2024-01-01"))
end = st.date_input("📅 النهاية", pd.to_datetime("2025-07-01"))

if st.button("🚀 شغّل التحليل"):
    try:
        # تحميل بيانات الأسعار
        data = yf.download(ticker, start=start, end=end)
        data = data[['Close']].rename(columns={'Close': 'price'})
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)

        # جلب الأخبار
        def fetch_google_news_rss(query):
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            articles = [{'title': entry.title,
                         'description': entry.description,
                         'date': entry.published}
                        for entry in feed.entries]
            df = pd.DataFrame(articles)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df

        def analyze_sentiment(text):
            if not text:
                return 0
            return TextBlob(text).sentiment.polarity

        news_df = fetch_google_news_rss(ticker.split('-')[0] + " crypto")
        news_df['sentiment_score'] = news_df['description'].apply(analyze_sentiment)
        news_df['Date'] = pd.to_datetime(news_df['date'])

        daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean().reset_index()

        # دمج البيانات
        merged = pd.merge(data, daily_sentiment, on='Date', how='left')
        merged['sentiment_score'].fillna(0, inplace=True)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        # النموذج التنبؤي
        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        # الرسم البياني
        st.subheader("📈 السعر الفعلي مقابل السعر المتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged['Date'], merged['price'], label="السعر الفعلي", color='blue')
        ax.plot(merged['Date'], merged['predicted_price'], label="السعر المتوقع", color='green')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        # معامل الارتباط
        corr = merged['price'].corr(merged['sentiment_score'])
        st.success(f"💡 معامل الارتباط بين السعر ومشاعر الأخبار: {corr:.3f}")

    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {e}")
