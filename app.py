import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل اصــــــــــــالة العملات والمشاعر", layout="wide")
st.title("📊 تحليل العملات اصــــــــــــالة  مقابل المشاعر وتوقع الأسعار")

# قائمة العملات
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "GALA-USD", "أخرى..."]
selected = st.selectbox("🪙 اختر عملة:", tickers)
ticker = st.text_input("✍️ أدخل رمز العملة (مثلاً SHIB-USD):") if selected == "أخرى..." else selected

start = st.date_input("📅 البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📅 النهاية", pd.to_datetime("2025-07-01"))

if ticker and st.button("🚀 شغّل التحليل"):
    try:
        # تحميل البيانات
        df = yf.download(ticker, start=start, end=end)[['Close']]
        df.reset_index(inplace=True)
        df.columns = ['Date', 'price']
        df['SMA_7'] = df['price'].rolling(window=7).mean()
        df['EMA_7'] = df['price'].ewm(span=7, adjust=False).mean()

        # تحليل الأخبار
        def fetch_news(query):
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+crypto&hl=en-US&gl=US&ceid=US:en"
            entries = feedparser.parse(url).entries
            news = [{'title': e.title, 'description': e.description, 'published': e.published} for e in entries[:5]]
            df = pd.DataFrame(news)
            df['date'] = pd.to_datetime(df['published']).dt.date
            return df

        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity if text else 0

        base = ticker.split("-")[0]
        news_df = fetch_news(base)
        news_df['sentiment_score'] = news_df['description'].apply(get_sentiment)
        news_df['Date'] = pd.to_datetime(news_df['date'])
        daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean().reset_index()

        # الدمج
        merged = pd.merge(df, daily_sentiment, on='Date', how='left')
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        # نموذج التنبؤ
        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        # توقع للأيام القادمة
        recent_sentiment = merged['sentiment_score'].tail(3).mean()
        future_dates = pd.date_range(start=merged['Date'].max() + pd.Timedelta(days=1), periods=7)
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['lagged_sentiment'] = recent_sentiment
        future_df['predicted_price'] = model.predict(future_df[['lagged_sentiment']])

        # الرسم
        st.subheader("📈 السعر التاريخي + المتوسطات + السعر المتوقع")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(merged['Date'], merged['price'], label="سعر فعلي", color='blue')
        ax.plot(merged['Date'], merged['SMA_7'], label="SMA 7", linestyle="--", color='gray')
        ax.plot(merged['Date'], merged['EMA_7'], label="EMA 7", linestyle="--", color='purple')
        ax.plot(merged['Date'], merged['predicted_price'], label="توقعات تاريخية", color='green')
        ax.plot(future_df['Date'], future_df['predicted_price'], label="توقع مستقبلي", linestyle='--', color='orange')
        ax.set_xlabel("التاريخ")
        ax.set_ylabel("السعر")
        ax.legend()
        st.pyplot(fig)

        # معامل الارتباط
        correlation = merged['price'].corr(merged['sentiment_score'])
        st.markdown(f"### 💡 معامل الارتباط: `{correlation:.3f}`")
        if abs(correlation) < 0.1:
            st.info("لا يوجد ارتباط قوي بين السعر والمشاعر — ربما السوق لا يتأثر بالأخبار حاليًا.")
        elif correlation > 0:
            st.success("المشاعر الإيجابية مرتبطة بارتفاع السعر.")
        else:
            st.warning("المشاعر السلبية مرتبطة بانخفاض السعر.")

        # عرض أهم الأخبار
        st.subheader("📰 أهم الأخبار المتعلقة بالعملة")
        for i, row in news_df.head(5).iterrows():
            st.markdown(f"**{row['title']}**  \n_{row['published']}_  \n> {row['description'][:200]}...")

    except Exception as e:
        st.error(f"🚨 حدث خطأ أثناء التحليل:\n\n{str(e)}")
