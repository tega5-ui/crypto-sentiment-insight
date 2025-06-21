import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل العملات والتوقع", layout="wide")
st.title("📊 تحليل العملات المشفرة وتوقع الأسعار بناءً على الأخبار")

# واجهة اختيار العملة
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "GALA-USD", "أخرى..."]
selected = st.selectbox("🪙 اختر العملة:", tickers)
ticker = st.text_input("✍️ أدخل الرمز يدويًا:", "") if selected == "أخرى..." else selected

# تحديد الفترة
start = st.date_input("📅 البداية", pd.to_datetime("2023-01-01"))
end = st.date_input("📅 النهاية", pd.to_datetime("2025-07-01"))

# اختيار عدد الأيام المستقبلية
forecast_days = st.radio("📆 عدد الأيام للتوقع:", [5, 14, 30], horizontal=True)

if ticker and st.button("🚀 شغّل التحليل"):
    try:
        # تحميل الأسعار
        df = yf.download(ticker, start=start, end=end)[['Close']]
        df.reset_index(inplace=True)
        df.columns = ['Date', 'price']
        df['SMA_7'] = df['price'].rolling(7).mean()
        df['EMA_7'] = df['price'].ewm(span=7, adjust=False).mean()

        # تحليل الأخبار
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

        # الدمج والتنبؤ
        merged = pd.merge(df, daily_sent, on='Date', how='left')
        merged['sentiment_score'].fillna(0, inplace=True)
        merged['lagged_sentiment'] = merged['sentiment_score'].shift(1)
        merged.dropna(inplace=True)

        model = LinearRegression()
        model.fit(merged[['lagged_sentiment']], merged['price'])
        merged['predicted_price'] = model.predict(merged[['lagged_sentiment']])

        # توقع المستقبل
        last_sentiment = merged['sentiment_score'].tail(3).mean()
        future_dates = pd.date_range(start=merged['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
        future = pd.DataFrame({'Date': future_dates})
        future['lagged_sentiment'] = last_sentiment
        future['predicted_price'] = model.predict(future[['lagged_sentiment']])
        future['trend'] = future['predicted_price'].diff().apply(lambda x: "📈 صعود" if x > 0 else ("📉 نزول" if x < 0 else "— ثبات"))

        # الرسم
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

        # معامل الارتباط + تفسير
        corr = merged['price'].corr(merged['sentiment_score'])
        st.markdown(f"### 💡 معامل الارتباط: `{corr:.3f}`")
        if abs(corr) < 0.1:
            st.info("↔️ لا يوجد ارتباط قوي بين السعر والمشاعر.")
        elif corr > 0:
            st.success("🔺 المشاعر الإيجابية مرتبطة بارتفاع السعر.")
        else:
            st.warning("🔻 المشاعر السلبية مرتبطة بانخفاض السعر.")

        # عرض جدول التوقعات
        st.subheader(f"📅 توقع الأسعار لمدة {forecast_days} يومًا")
        future_display = future[['Date', 'predicted_price', 'trend']]
        future_display.columns = ['التاريخ', 'السعر المتوقع', 'الاتجاه']
        st.dataframe(future_display.style.format({'السعر المتوقع': '{:.4f}'}))

        # عرض الأخبار
        st.subheader("📰 أهم 5 أخبار مؤثرة")
        for i, row in news.iterrows():
            st.markdown(f"**{row['title']}**  \n_{row['published']}_  \n> {row['description'][:200]}...")

    except Exception as e:
        st.error(f"🚨 حدث خطأ أثناء التحليل:\n\n{str(e)}")
