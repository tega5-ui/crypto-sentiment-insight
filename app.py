import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from prophet import Prophet
from textblob import TextBlob
from fpdf import FPDF
import datetime

st.set_page_config(page_title="📈 نظام التحليل الفني المتقدم", layout="wide", page_icon="💹")
st.title("💹 نظام التحليل الفني وتوقع الأسعار للعملات الرقمية")

@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        return None

default_tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]
custom_tickers = st.session_state.get('custom_tickers', [])
all_tickers = default_tickers + custom_tickers

with st.sidebar:
    st.header("⚙️ الإعدادات الرئيسية")
    st.subheader("🪙 إدارة العملات")
    with st.expander("➕ إضافة عملة جديدة"):
        new_ticker = st.text_input("رمز العملة (مثل: XRP-USD):", key="new_ticker")
        if st.button("إضافة العملة"):
            if new_ticker and new_ticker not in all_tickers:
                custom_tickers.append(new_ticker)
                st.session_state.custom_tickers = custom_tickers
                st.success(f"تمت إضافة {new_ticker}")
            elif new_ticker in all_tickers:
                st.warning("هذه العملة مضافه مسبقاً")

    st.subheader("📅 إعدادات الفترة")
    start_date = st.date_input("تاريخ البداية", datetime.date(2023, 1, 1),
                               max_value=datetime.date.today() - datetime.timedelta(days=7))
    end_date = st.date_input("تاريخ النهاية", datetime.date.today(),
                             min_value=start_date + datetime.timedelta(days=7),
                             max_value=datetime.date.today())

    st.subheader("🔍 خيارات التحليل")
    forecast_days = st.slider("أيام التنبؤ المستقبلي:", 1, 90, 14)
    enable_prophet = st.checkbox("تفعيل تنبؤات Prophet", True)
    enable_sentiment = st.checkbox("تفعيل تحليل المشاعر", False)

tab1, tab2, tab3 = st.tabs(["📊 التحليل الفني", "🔮 التنبؤ المستقبلي", "⚙️ الإعدادات المتقدمة"])

with tab1:
    st.header("📊 التحليل الفني المتقدم")
    selected_ticker = st.selectbox("اختر العملة للتحليل:", all_tickers, index=0)

    if st.button("🚀 تنفيذ التحليل", key="analyze_btn"):
        with st.spinner("جاري تحليل البيانات، يرجى الانتظار..."):
            try:
                df = load_data(selected_ticker, start_date, end_date)
                if df is None or df.empty:
                    st.error("لا توجد بيانات متاحة للفترة المحددة")
                    st.stop()

                df = df[['Close']].reset_index()
                df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

                price_series = df['y'].dropna().values.flatten()
                index = df.index

                df['EMA_7'] = pd.Series(price_series, index=index).ewm(span=7).mean().values
                df['EMA_14'] = pd.Series(price_series, index=index).ewm(span=14).mean().values
                df['SMA_20'] = pd.Series(price_series, index=index).rolling(20).mean().values

                rsi = ta.momentum.RSIIndicator(close=pd.Series(price_series, index=index), window=14)
                df['RSI'] = rsi.rsi().values

                bb = ta.volatility.BollingerBands(close=pd.Series(price_series, index=index), window=20, window_dev=2)
                df['BB_upper'] = bb.bollinger_hband().values
                df['BB_lower'] = bb.bollinger_lband().values

                macd = ta.trend.MACD(close=pd.Series(price_series, index=index))
                df['MACD'] = macd.macd().values
                df['MACD_signal'] = macd.macd_signal().values

                df = df.dropna()
                latest = df.iloc[-1]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("السعر الحالي", f"${float(latest['y']):.2f}")
                    st.metric("المتوسط المتحرك 7 أيام", f"${float(latest['EMA_7']):.2f}")
                with col2:
                    st.metric("المتوسط المتحرك 14 يوم", f"${float(latest['EMA_14']):.2f}")
                    st.metric("المتوسط المتحرك 20 يوم", f"${float(latest['SMA_20']):.2f}")
                with col3:
                    st.metric("مؤشر RSI", f"{float(latest['RSI']):.2f}")
                    st.metric("نطاق بولينجر", f"{float(latest['BB_lower']):.2f} - {float(latest['BB_upper']):.2f}")

                trend = "صاعد" if float(latest['y']) > float(latest['EMA_14']) else "هابط"

                st.subheader("📈 الرسم البياني")
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(df['ds'], df['y'], label='السعر', color='blue')
                ax.plot(df['ds'], df['EMA_7'], label='EMA 7', linestyle='--')
                ax.plot(df['ds'], df['EMA_14'], label='EMA 14', linestyle='--')
                ax.plot(df['ds'], df['SMA_20'], label='SMA 20', linestyle='-.')
                ax.fill_between(df['ds'], df['BB_lower'], df['BB_upper'], alpha=0.1, label='نطاق بولينجر')
                ax.legend()
                st.pyplot(fig)

                st.subheader("📉 MACD")
                fig_macd, ax_macd = plt.subplots(figsize=(14, 3))
                ax_macd.plot(df['ds'], df['MACD'], label='MACD', color='blue')
                ax_macd.plot(df['ds'], df['MACD_signal'], label='خط الإشارة', color='red')
                ax_macd.axhline(0, color='gray', linestyle='--')
                ax_macd.legend()
                st.pyplot(fig_macd)

                st.subheader("🔔 التنبيهات الفنية")
                if float(latest['RSI']) > 70:
                    st.warning("⚠️ مؤشر RSI في منطقة التشبع الشرائي")
                elif float(latest['RSI']) < 30:
                    st.info("ℹ️ مؤشر RSI في منطقة التشبع البيعي")

                if float(latest['y']) < float(latest['BB_lower']):
                    st.success("💡 فرصة شراء: السعر تحت نطاق بولينجر")
                elif float(latest['y']) > float(latest['BB_upper']):
                    st.warning("⚠️ السعر فوق نطاق بولينجر")

            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

# يمكن تضمين التبويبين الآخرين Prophet و PDF عند الطلب
