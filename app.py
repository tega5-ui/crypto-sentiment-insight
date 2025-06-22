import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta
from prophet import Prophet
from textblob import TextBlob
from fpdf import FPDF
import datetime

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
    st.subheader("📅 إعدادات الفترة")
    start_date = st.date_input(
        "تاريخ البداية",
        datetime.date(2023, 1, 1),
        max_value=datetime.date.today() - datetime.timedelta(days=7)
    )
    end_date = st.date_input(
        "تاريخ النهاية",
        datetime.date.today(),
        min_value=start_date + datetime.timedelta(days=7),
        max_value=datetime.date.today()
    )

    # إعدادات التحليل
    st.subheader("🔍 خيارات التحليل")
    forecast_days = st.slider("أيام التنبؤ المستقبلي:", 1, 90, 14)
    enable_prophet = st.checkbox("تفعيل تنبؤات Prophet", True)
    enable_sentiment = st.checkbox("تفعيل تحليل المشاعر", False)

# الواجهة الرئيسية
tab1, tab2, tab3 = st.tabs(["📊 التحليل الفني", "🔮 التنبؤ المستقبلي", "⚙️ الإعدادات المتقدمة"])

with tab1:
    st.header("📊 التحليل الفني المتقدم")
    
    selected_ticker = st.selectbox(
        "اختر العملة للتحليل:",
        all_tickers,
        index=0
    )
    
    if st.button("🚀 تنفيذ التحليل", key="analyze_btn"):
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
                df['EMA_7'] = pd.Series(price_series).ewm(span=7).mean().values
                df['EMA_14'] = pd.Series(price_series).ewm(span=14).mean().values
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
                    st.metric("السعر الحالي", f"${latest['y']:.2f}")
                    st.metric("المتوسط المتحرك 7 أيام", f"${latest['EMA_7']:.2f}")
                with col2:
                    st.metric("المتوسط المتحرك 14 يوم", f"${latest['EMA_14']:.2f}")
                    st.metric("المتوسط المتحرك 20 يوم", f"${latest['SMA_20']:.2f}")
                with col3:
                    st.metric("مؤشر RSI", f"{latest['RSI']:.2f}")
                    st.metric("نطاق بولينجر", f"{latest['BB_lower']:.2f} - {latest['BB_upper']:.2f}")
                
                # تحليل الاتجاه
                trend = "صاعد" if latest['y'] > latest['EMA_14'] else "هابط"
                rsi_status = "تشبع شراء" if latest['RSI'] > 70 else "تشبع بيع" if latest['RSI'] < 30 else "حيادي"
                
                st.subheader("📈 الرسم البياني التفاعلي")
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df['ds'], df['y'], label='السعر', color='blue')
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
                
                if latest['y'] < latest['BB_lower']:
                    st.success("💡 فرصة شراء: السعر عند الحد الأدنى لنطاق بولينجر")
                elif latest['y'] > latest['BB_upper']:
                    st.warning("⚠️ انتباه: السعر عند الحد الأعلى لنطاق بولينجر")
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

with tab2:
    st.header("🔮 التنبؤ المستقبلي")
    
    if 'df' not in locals():
        st.warning("الرجاء تنفيذ التحليل الفني أولاً من تبويب التحليل الفني")
    else:
        if enable_prophet:
            with st.spinner("جاري إنشاء التنبؤات المستقبلية..."):
                try:
                    # تحضير البيانات للنموذج النبوي
                    prophet_df = df[['ds', 'y']].copy()
                    prophet_df.columns = ['ds', 'y']
                    
                    # إنشاء وتدريب النموذج
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    model.fit(prophet_df)
                    
                    # إنشاء إطار البيانات المستقبلي
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)
                    
                    # عرض النتائج
                    st.subheader(f"📅 توقعات السعر لـ {forecast_days} يوم القادمة")
                    
                    # عرض آخر التنبؤات
                    st.dataframe(
                        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).style.format({
                            'yhat': "${:.2f}",
                            'yhat_lower': "${:.2f}",
                            'yhat_upper': "${:.2f}"
                        }),
                        height=400
                    )
                    
                    # رسم التنبؤات
                    st.subheader("📈 رسم بياني للتنبؤات")
                    fig_forecast = model.plot(forecast)
                    st.pyplot(fig_forecast)
                    
                    # رسم المكونات
                    st.subheader("🧩 مكونات التنبؤ")
                    fig_components = model.plot_components(forecast)
                    st.pyplot(fig_components)
                    
                    # تحليل الاتجاه المستقبلي
                    last_trend = forecast['trend'].iloc[-1]
                    first_trend = forecast['trend'].iloc[-forecast_days]
                    trend_direction = "صاعد" if last_trend > first_trend else "هابط"
                    
                    st.metric("الاتجاه المتوقع", trend_direction)
                    
                except Exception as e:
                    st.error(f"حدث خطأ في التنبؤ: {str(e)}")
        else:
            st.info("تفعيل خيار التنبؤات في الإعدادات لعرض التنبؤات المستقبلية")

with tab3:
    st.header("⚙️ الإعدادات المتقدمة")
    
    if enable_sentiment:
        st.subheader("📰 تحليل المشاعر من الأخبار")
        news_text = st.text_area("أدخل نص الخبر أو التغريدة عن العملة:")
        
        if news_text:
            analysis = TextBlob(news_text)
            sentiment_score = analysis.sentiment.polarity
            
            if sentiment_score > 0.2:
                sentiment = "🟢 إيجابي"
                st.success(f"تحليل المشاعر: {sentiment} (نقاط: {sentiment_score:.2f})")
            elif sentiment_score < -0.2:
                sentiment = "🔴 سلبي"
                st.error(f"تحليل المشاعر: {sentiment} (نقاط: {sentiment_score:.2f})")
            else:
                sentiment = "⚪ محايد"
                st.info(f"تحليل المشاعر: {sentiment} (نقاط: {sentiment_score:.2f})")
    
    st.subheader("📤 تصدير النتائج")
    if st.button("🖨️ إنشاء تقرير PDF"):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font('Arial', '', 'arial.ttf', uni=True)
            pdf.set_font("Arial", size=12)
            
            pdf.cell(200, 10, txt=f"تقرير التحليل الفني لـ {selected_ticker}", ln=True, align='C')
            pdf.ln(10)
            
            pdf.cell(200, 10, txt=f"الفترة من {start_date} إلى {end_date}", ln=True, align='C')
            pdf.ln(15)
            
            # إضافة البيانات الأساسية
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"السعر الحالي: ${latest['y']:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"الاتجاه العام: {'صاعد' if latest['y'] > latest['EMA_14'] else 'هابط'}", ln=True)
            pdf.cell(200, 10, txt=f"مؤشر RSI: {latest['RSI']:.2f} ({'تشبع شراء' if latest['RSI'] > 70 else 'تشبع بيع' if latest['RSI'] < 30 else 'حيادي'})", ln=True)
            pdf.ln(10)
            
            # إضافة التنبيهات
            pdf.set_font("Arial", size=10, style='B')
            pdf.cell(200, 10, txt="الملاحظات والتنبيهات:", ln=True)
            pdf.set_font("Arial", size=10)
            
            if latest['RSI'] > 70:
                pdf.cell(200, 10, txt="- تحذير: مؤشر RSI في منطقة التشبع الشرائي", ln=True)
            elif latest['RSI'] < 30:
                pdf.cell(200, 10, txt="- انتباه: مؤشر RSI في منطقة التشبع البيعي", ln=True)
                
            if latest['y'] < latest['BB_lower']:
                pdf.cell(200, 10, txt="- فرصة شراء: السعر عند الحد الأدنى لنطاق بولينجر", ln=True)
            elif latest['y'] > latest['BB_upper']:
                pdf.cell(200, 10, txt="- انتباه: السعر عند الحد الأعلى لنطاق بولينجر", ln=True)
            
            pdf.output("technical_analysis_report.pdf")
            st.success("تم إنشاء التقرير بنجاح (technical_analysis_report.pdf)")
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء إنشاء التقرير: {str(e)}")

# تذييل الصفحة
st.markdown("---")
st.caption("""
تم تطوير هذا النظام باستخدام Python (Streamlit, yfinance, TA-Lib, Prophet).  
البيانات المقدمة لأغراض تعليمية فقط وليست نصيحة مالية.
""")
