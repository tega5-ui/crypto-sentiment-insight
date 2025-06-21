import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import numpy as np
import ta
from datetime import datetime, timedelta

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="\U0001F4C8 توقع السعر الفني المتقدم",
    layout="wide",
    page_icon="\U0001F4CA"
)
st.title("\U0001F4CA توقع السعر المستقبلي باستخدام ARIMA والتحليل الفني المتقدم")

# تحسين الأداء باستخدام cache
@st.cache_data
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# قائمة العملات مع أسماء واضحة
tickers = {
    "BTC-USD": "بتكوين",
    "ETH-USD": "إيثريوم",
    "ADA-USD": "كاردانو",
    "BNB-USD": "بينانس كوين",
    "SOL-USD": "سولانا"
}

# واجهة المستخدم
with st.sidebar:
    st.header("⚙️ الإعدادات")
    ticker = st.selectbox(
        "🪙 اختر العملة:",
        options=list(tickers.keys()),
        format_func=lambda x: f"{x} ({tickers[x]})"
    )
    start = st.date_input(
        "📆 تاريخ البداية",
        value=pd.to_datetime("2023-01-01"),
        max_value=datetime.now() - timedelta(days=7)
    )
    end = st.date_input(
        "📆 تاريخ النهاية",
        value=datetime.now(),
        min_value=start + timedelta(days=7),
        max_value=datetime.now())
    forecast_days = st.slider(
        "🔮 عدد الأيام المستقبلية:",
        min_value=1, max_value=60, value=14)
    arima_order = st.selectbox(
        "🧠 معاملات ARIMA (p,d,q):",
        options=[(3,1,1), (5,1,2), (7,2,3), "تلقائي"],
        index=0
    )
    st.markdown("---")
    st.info("""
    **ملاحظات:**
    - استخدام 'تلقائي' قد يستغرق وقتاً أطول.
    - البيانات تُحمّل من Yahoo Finance.
    """)

if st.button("\U0001F680 بدء التحليل", use_container_width=True):
    with st.spinner("جاري تحليل البيانات..."):
        try:
            df = load_data(ticker, start, end)
            if df.empty:
                st.error("⚠️ لم يتم العثور على بيانات للفترة المحددة!")
                st.stop()

            df = df[['Close']].copy()
            df.reset_index(inplace=True)
            df.rename(columns={'Close': 'price'}, inplace=True)

            # المؤشرات الفنية
            df['EMA_7'] = df['price'].ewm(span=7, adjust=False).mean()
            df['EMA_14'] = df['price'].ewm(span=14, adjust=False).mean()
            df['SMA_7'] = df['price'].rolling(window=7).mean()
            df['SMA_14'] = df['price'].rolling(window=14).mean()
            df['RSI_14'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
            df['RSI_7'] = ta.momentum.RSIIndicator(close=df['price'], window=7).rsi()
            bb = ta.volatility.BollingerBands(close=df['price'], window=14, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            macd = ta.trend.MACD(close=df['price'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()

            price_series = df['price'].dropna()

            # اختيار معاملات ARIMA
            if arima_order == "تلقائي":
                auto_model = auto_arima(
                    price_series,
                    seasonal=False,
                    trace=True,
                    suppress_warnings=True,
                    stepwise=True
                )
                order = auto_model.order
                st.success(f"تم اختيار معاملات ARIMA تلقائياً: {order}")
            else:
                order = arima_order

            model = ARIMA(price_series, order=order)
            fitted = model.fit()

            last_price = price_series.iloc[-1]
            forecast = fitted.forecast(steps=forecast_days)
            volatility = price_series.pct_change().std()
            forecast = np.clip(
                forecast,
                last_price * (1 - 2 * volatility),
                last_price * (1 + 2 * volatility)
            )

            forecast_dates = pd.date_range(
                start=df['Date'].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_days
            )
            forecast_df = pd.DataFrame({
                'التاريخ': forecast_dates,
                'السعر المتوقع': forecast.round(2),
                'التغير %': ((forecast / last_price - 1) * 100).round(2),
                'الإشارة': np.where(forecast > last_price, '📈 صعود', '📉 هبوط')
            })

            # النتائج
            st.success("✅ تم الانتهاء من التحليل بنجاح!")
            current_rsi = df['RSI_14'].iloc[-1]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("السعر الحالي", f"${last_price:,.2f}")
            with col2:
                st.metric("RSI (14)", f"{current_rsi:.1f}",
                          "تشبع شراء" if current_rsi > 70 else "تشبع بيع" if current_rsi < 30 else "حيادي")
            with col3:
                st.metric("التقلب الأخير", f"{(volatility * 100):.2f}%")

            st.subheader(f"📅 توقعات السعر لـ {forecast_days} يوم القادمة")
            st.dataframe(
                forecast_df.style.format({
                    'السعر المتوقع': "${:,.2f}",
                    'التغير %': "{:.2f}%"
                }),
                hide_index=True
            )

            st.subheader("\U0001F4CA التحليل الفني والتنبؤات")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
            ax1.plot(df['Date'], df['price'], label="السعر", color='#1f77b4')
            ax1.plot(df['Date'], df['EMA_7'], label="EMA 7", linestyle="--", color='#ff7f0e')
            ax1.plot(df['Date'], df['EMA_14'], label="EMA 14", linestyle="--", color='#2ca02c')
            ax1.fill_between(df['Date'], df['bb_lower'], df['bb_upper'], color='gray', alpha=0.1, label="نطاق بولينجر")
            ax1.plot(forecast_dates, forecast, 'ro--', label="التنبؤ")
            ax1.set_title("تحليل السعر والمؤشرات")
            ax1.legend(loc='upper left')

            ax2.plot(df['Date'], df['RSI_14'], label="RSI 14", color='#9467bd')
            ax2.axhline(70, linestyle='--', color='red', alpha=0.3)
            ax2.axhline(30, linestyle='--', color='green', alpha=0.3)
            ax2.set_title("مؤشر RSI")
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("\U0001F4DD ملخص التحليل")
            latest = df.iloc[-1]
            analysis = f"""
            - **الاتجاه العام:** {'صاعد' if latest['price'] > latest['EMA_14'] else 'هابط'}
            - **تقييم RSI (14):** {current_rsi:.1f} → {'🔴 تشبع شراء' if current_rsi > 70 else '🟢 تشبع بيع' if current_rsi < 30 else '🟡 منطقة محايدة'}
            - **نطاق بولينجر:** {'🟢 الجزء السفلي' if latest['price'] < latest['bb_lower'] else '🔴 الجزء العلوي' if latest['price'] > latest['bb_upper'] else '🟡 المنطقة الوسطى'}
            - **إشارة MACD:** {'🟢 إيجابية' if latest['MACD'] > latest['MACD_signal'] else '🔴 سلبية'}
            """
            st.markdown(analysis)

            st.download_button(
                label="\U0001F4E5 تحميل بيانات التحليل",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{ticker}_analysis.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ حدث خطأ أثناء التحليل:\n\n{str(e)}")
            st.stop()

# تذييل
st.markdown("---")
st.caption("""
تم تطوير هذا التطبيق باستخدام Python (Streamlit, yfinance, statsmodels, ta-lib).  
المعلومات المقدمة ليست نصيحة مالية. استخدمها على مسؤوليتك الخاصة.
""")
