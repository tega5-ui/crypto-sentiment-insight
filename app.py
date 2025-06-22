import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from datetime import date, datetime, timedelta  # الاستيراد الصحيح للتاريخ

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="📈 نظام التحليل الفني المتقدم",
    layout="wide",
    page_icon="💹"
)

# عنوان التطبيق
st.title("💹 نظام التحليل الفني للعملات الرقمية")

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

# واجهة المستخدم في الشريط الجانبي
with st.sidebar:
    st.header("⚙️ الإعدادات الرئيسية")
    
    # إدارة العملات
    st.subheader("🪙 إدارة العملات")
    custom_tickers = st.session_state.get('custom_tickers', [])
    all_tickers = default_tickers + custom_tickers
    
    with st.expander("➕ إضافة عملة جديدة"):
        new_ticker = st.text_input("رمز العملة (مثل: XRP-USD):", key="new_ticker")
        if st.button("إضافة العملة"):
            if new_ticker and new_ticker not in all_tickers:
                custom_tickers.append(new_ticker)
                st.session_state.custom_tickers = custom_tickers
                st.success(f"تمت إضافة {new_ticker}")
            elif new_ticker in all_tickers:
                st.warning("هذه العملة مضافه مسبقاً")

    # إعدادات الفترة الزمنية
    st.subheader("📅 إعدادات الفترة الزمنية")
    
    # اختيار نوع الفترة
    period_type = st.radio("نوع الفترة:", ["مخصص", "اختيار سريع"])
    
    if period_type == "اختيار سريع":
        quick_period = st.selectbox("اختر فترة سريعة:", 
                                  ["آخر أسبوع", "آخر شهر", "آخر 3 أشهر", "آخر سنة", "آخر سنتين"])
        
        end_date = date.today()
        
        if quick_period == "آخر أسبوع":
            start_date = end_date - timedelta(days=7)
        elif quick_period == "آخر شهر":
            start_date = end_date - timedelta(days=30)
        elif quick_period == "آخر 3 أشهر":
            start_date = end_date - timedelta(days=90)
        elif quick_period == "آخر سنة":
            start_date = end_date - timedelta(days=365)
        else: # آخر سنتين
            start_date = end_date - timedelta(days=730)
    else:
        # الفترة المخصصة
        start_date = st.date_input(
            "تاريخ البداية",
            date(2023, 1, 1),
            max_value=date.today() - timedelta(days=1)
        )
        end_date = st.date_input(
            "تاريخ النهاية",
            date.today(),
            min_value=start_date + timedelta(days=1),
            max_value=date.today()
        )

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
            df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
            
            # التحليل الفني
            price_series = df['y'].dropna().values.flatten()
            
            # حساب المؤشرات
            df['EMA_7'] = pd.Series(price_series).ewm(span=7).mean()
            df['SMA_14'] = pd.Series(price_series).rolling(14).mean()
            df['SMA_50'] = pd.Series(price_series).rolling(50).mean()
            
            # مؤشر RSI
            rsi_indicator = ta.momentum.RSIIndicator(close=pd.Series(price_series), window=14)
            df['RSI'] = rsi_indicator.rsi()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=pd.Series(price_series), window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            
            # عرض النتائج
            latest = df.iloc[-1]
            
            # إنشاء بطاقات المقاييس
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("السعر الحالي", f"{latest['y']:.2f} $")
                st.metric("المتوسط المتحرك 7 أيام", f"{latest['EMA_7']:.2f} $")
            with col2:
                st.metric("المتوسط المتحرك 14 يوم", f"{latest['SMA_14']:.2f} $")
                st.metric("المتوسط المتحرك 50 يوم", f"{latest['SMA_50']:.2f} $")
            with col3:
                st.metric("مؤشر RSI", f"{latest['RSI']:.2f}")
                st.metric("نطاق بولينجر", f"{latest['BB_lower']:.2f} - {latest['BB_upper']:.2f} $")
            
            # تحليل الاتجاه
            trend = "صاعد" if latest['y'] > latest['SMA_50'] else "هابط"
            rsi_status = "تشبع شراء" if latest['RSI'] > 70 else "تشبع بيع" if latest['RSI'] < 30 else "حيادي"
            
            st.subheader("📈 الرسم البياني التفاعلي")
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df['ds'], df['y'], label='السعر', color='blue')
            ax.plot(df['ds'], df['EMA_7'], label='EMA 7', linestyle='--', color='orange')
            ax.plot(df['ds'], df['SMA_14'], label='SMA 14', linestyle='--', color='green')
            ax.plot(df['ds'], df['SMA_50'], label='SMA 50', linestyle='--', color='purple')
            ax.fill_between(df['ds'], df['BB_lower'], df['BB_upper'], color='gray', alpha=0.2, label='نطاق بولينجر')
            ax.set_title(f"تحليل سعر {selected_ticker} من {start_date} إلى {end_date}")
            ax.legend()
            st.pyplot(fig)
            
            # التنبيهات الفنية
            st.subheader("🔔 التنبيهات الفنية")
            if latest['RSI'] > 70:
                st.warning("⚠️ تحذير: مؤشر RSI في منطقة التشبع الشرائي (فوق 70)")
            elif latest['RSI'] < 30:
                st.info("ℹ️ انتباه: مؤشر RSI في منطقة التشبع البيعي (تحت 30)")
            
            if latest['y'] < latest['BB_lower']:
                st.success("💡 فرصة شراء: السعر عند الحد الأدنى لنطاق بولينجر")
            elif latest['y'] > latest['BB_upper']:
                st.warning("⚠️ انتباه: السعر عند الحد الأعلى لنطاق بولينجر")
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

# تذييل الصفحة
st.markdown("---")
st.caption("""
تم تطوير هذا النظام باستخدام Python (Streamlit, yfinance, TA-Lib).  
البيانات المقدمة لأغراض تعليمية فقط وليست نصيحة مالية.
""")
