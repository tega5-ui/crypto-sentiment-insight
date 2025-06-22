import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import ta
from datetime import datetime, timedelta

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="📈 نظام التحليل الفني المتقدم",
    layout="wide",
    page_icon="💹"
)

# عنوان التطبيق
st.title("💹 نظام التحليل الفني وتوقع الأسعار للعملات الرقمية")

# تحميل البيانات مع كاش
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        return None

# القائمة الأساسية للعملات
default_tickers = ["BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "SOL-USD"]

# واجهة المستخدم
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # إدارة العملات
    st.subheader("🪙 إدارة العملات")
    custom_tickers = st.session_state.get('custom_tickers', [])
    all_tickers = default_tickers + custom_tickers
    
    with st.expander("➕ إضافة عملة جديدة"):
        new_ticker = st.text_input("رمز العملة (مثل: XRP-USD):")
        if st.button("إضافة العملة"):
            if new_ticker and new_ticker not in all_tickers:
                custom_tickers.append(new_ticker)
                st.session_state.custom_tickers = custom_tickers
                st.success(f"تمت إضافة {new_ticker}")
            elif new_ticker in all_tickers:
                st.warning("هذه العملة مضافه مسبقاً")

    # إعدادات الفترة الزمنية
    st.subheader("📅 إعدادات الفترة الزمنية")
    start_date = st.date_input(
        "تاريخ البداية",
        datetime(2023, 1, 1).date(),
        max_value=datetime.today().date() - timedelta(days=7)
    )
    end_date = st.date_input(
        "تاريخ النهاية",
        datetime.today().date(),
        min_value=start_date + timedelta(days=7),
        max_value=datetime.today().date()
    )
    
    # إعدادات التحليل
    st.subheader("🔍 خيارات التحليل")
    forecast_days = st.slider("أيام التنبؤ المستقبلي:", 1, 90, 14)
    enable_technical = st.checkbox("تفعيل التحليل الفني", True)

# الواجهة الرئيسية
selected_ticker = st.selectbox(
    "اختر العملة للتحليل:",
    all_tickers,
    index=0
)

if st.button("🚀 تنفيذ التحليل"):
    with st.spinner("جاري تحليل البيانات، يرجى الانتظار..."):
        try:
            # تحميل البيانات الأساسية
            df = load_data(selected_ticker, start_date, end_date)
            if df is None or df.empty:
                st.error("لا توجد بيانات متاحة للفترة المحددة")
                st.stop()
            
            df = df[['Close']].reset_index()
            df.rename(columns={'Date': 'ds', 'Close': 'price'}, inplace=True)
            
            # التحليل الفني
            price_series = df['price'].dropna().values.flatten()
            
            if enable_technical:
                # حساب المؤشرات الفنية
                df['EMA_7'] = pd.Series(price_series).ewm(span=7, adjust=False).mean().values
                df['EMA_14'] = pd.Series(price_series).ewm(span=14, adjust=False).mean().values
                df['SMA_20'] = pd.Series(price_series).rolling(20).mean().values
                
                # مؤشر RSI
                rsi_indicator = ta.momentum.RSIIndicator(close=pd.Series(price_series), window=14)
                df['RSI'] = rsi_indicator.rsi().values
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(close=pd.Series(price_series), window=20, window_dev=2)
                df['BB_upper'] = bb.bollinger_hband().values
                df['BB_middle'] = bb.bollinger_mavg().values
                df['BB_lower'] = bb.bollinger_lband().values
                
                # MACD
                macd = ta.trend.MACD(close=pd.Series(price_series))
                df['MACD'] = macd.macd().values
                df['MACD_signal'] = macd.macd_signal().values
            
            # عرض النتائج
            latest = df.iloc[-1]
            
            # إنشاء بطاقات المقاييس
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("السعر الحالي", f"{latest['price']:.2f} $")
                if enable_technical:
                    st.metric("المتوسط المتحرك 7 أيام", f"{latest['EMA_7']:.2f} $")
            with col2:
                if enable_technical:
                    st.metric("المتوسط المتحرك 14 يوم", f"{latest['EMA_14']:.2f} $")
                    st.metric("المتوسط المتحرك 20 يوم", f"{latest['SMA_20']:.2f} $")
            with col3:
                if enable_technical:
                    st.metric("مؤشر RSI", f"{latest['RSI']:.2f}")
                    st.metric("نطاق بولينجر", f"{latest['BB_lower']:.2f} - {latest['BB_upper']:.2f} $")
            
            if enable_technical:
                # تحليل الاتجاه
                trend = "صاعد" if latest['price'] > latest['EMA_14'] else "هابط"
                rsi_status = "تشبع شراء" if latest['RSI'] > 70 else "تشبع بيع" if latest['RSI'] < 30 else "حيادي"
                
                st.subheader("📈 الرسم البياني التفاعلي")
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df['ds'], df['price'], label='السعر', color='blue')
                ax.plot(df['ds'], df['EMA_7'], label='EMA 7', linestyle='--', color='orange')
                ax.plot(df['ds'], df['EMA_14'], label='EMA 14', linestyle='--', color='green')
                ax.plot(df['ds'], df['SMA_20'], label='SMA 20', linestyle='--', color='purple')
                ax.fill_between(df['ds'], df['BB_lower'], df['BB_upper'], color='gray', alpha=0.2, label='نطاق بولينجر')
                ax.set_title(f"تحليل سعر {selected_ticker}")
                ax.legend()
                st.pyplot(fig)
                
                # تحليل MACD
                st.subheader("📉 تحليل مؤشر MACD")
                fig_macd, ax_macd = plt.subplots(figsize=(14, 4))
                ax_macd.plot(df['ds'], df['MACD'], label='MACD', color='blue')
                ax_macd.plot(df['ds'], df['MACD_signal'], label='خط الإشارة', color='red')
                ax_macd.axhline(0, color='gray', linestyle='--')
                ax_macd.set_title("مؤشر MACD")
                ax_macd.legend()
                st.pyplot(fig_macd)
                
                # التنبيهات الفنية
                st.subheader("🔔 التنبيهات الفنية")
                if latest['RSI'] > 70:
                    st.warning("⚠️ تحذير: مؤشر RSI في منطقة التشبع الشرائي (فوق 70)")
                elif latest['RSI'] < 30:
                    st.info("ℹ️ انتباه: مؤشر RSI في منطقة التشبع البيعي (تحت 30)")
                
                if latest['price'] < latest['BB_lower']:
                    st.success("💡 فرصة شراء: السعر عند الحد الأدنى لنطاق بولينجر")
                elif latest['price'] > latest['BB_upper']:
                    st.warning("⚠️ انتباه: السعر عند الحد الأعلى لنطاق بولينجر")
            
            # التنبؤات المستقبلية (مثال مبسط)
            st.subheader("🔮 توقعات مستقبلية")
            st.write("سيتم إضافة نماذج التنبؤ في التحديثات القادمة")
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

# تذييل الصفحة
st.markdown("---")
st.caption("""
تم تطوير هذا النظام باستخدام Python (Streamlit, yfinance, TA-Lib).  
البيانات المقدمة لأغراض تعليمية فقط وليست نصيحة مالية.
""")
