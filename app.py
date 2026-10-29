import streamlit as st
import pandas as pd
import time
import random
import re
import plotly.express as px

# ---------------------------------------------------------
# ตั้งค่าหน้าเพจ (ต้องอยู่บรรทัดแรกสุดของคำสั่ง Streamlit)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thai Sentiment Analysis & CRM",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. ฟังก์ชันจัดการข้อมูล (Text Preprocessing)
# ---------------------------------------------------------
def clean_text(text):
    """ทำความสะอาดข้อความก่อนส่งเข้าโมเดลตามแผนการดำเนินงาน"""
    if not isinstance(text, str):
        return ""
    
    # 1. ลบ URL หรือลิงก์ต่างๆ
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 2. ลบอักขระพิเศษและ Emoji (อนุญาตเฉพาะตัวอักษรไทย อังกฤษ ตัวเลข และช่องว่าง)
    text = re.sub(r'[^\w\sก-๙]', '', text)
    # 3. จัดการช่องว่างที่ซ้ำซ้อนให้เหลือช่องเดียว และลบช่องว่างหัวท้าย
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ---------------------------------------------------------
# 2. ส่วนจำลองการทำงานของโมเดล
# ---------------------------------------------------------
def predict_sentiment(text, model_name):
    # นำข้อความไปทำความสะอาดก่อน
    cleaned_text = clean_text(text)
    
    # จำลองการทำนาย (แทนที่ด้วย PyTorch/HuggingFace ของจริงในอนาคต)
    classes = ["เชิงบวก", "เชิงกลาง", "เชิงลบ"]
    predicted_class = random.choice(classes)
    confidence = random.uniform(0.80, 0.99) 
    
    return predicted_class, confidence, cleaned_text

# ---------------------------------------------------------
# 3. ส่วนอินเทอร์เฟซ (UI) & กราฟิกที่ทันสมัย
# ---------------------------------------------------------
# ส่วนหัวโปรแกรม (Header) ตกแต่งด้วย HTML/CSS เล็กน้อย
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>✨ ระบบวิเคราะห์ความรู้สึกรีวิวลูกค้าภาษาไทย</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>แพลตฟอร์ม AI สำหรับทีม CRM เพื่อคัดกรองปัญหาและยกระดับประสบการณ์ลูกค้า</p>", unsafe_allow_html=True)
st.divider()

# แถบด้านข้าง (Sidebar) สำหรับตั้งค่า
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=100) # โลโก้จำลอง
    st.header("⚙️ ตั้งค่าโมเดล")
    model_choice = st.selectbox(
        "เลือกสถาปัตยกรรมโมเดล AI",
        ("WangchanBERTa (Transformer)", "Bi-directional LSTM")
    )
    st.info(f"**สถานะ:** พร้อมใช้งาน\n\n**โมเดลที่เลือก:** {model_choice}")
    st.caption("พัฒนาโดย: กลุ่มนักศึกษา CS461")

# แบ่งหน้าจอเป็น 2 แท็บหลัก
tab1, tab2 = st.tabs(["✍️ วิเคราะห์ข้อความแบบเดี่ยว (Single Text)", "📁 วิเคราะห์ข้อมูลกลุ่มแบบสถิติ (Batch & Analytics)"])

# ---------------------------------------------------------
# แท็บที่ 1: การกรอกข้อความเดี่ยว
# ---------------------------------------------------------
with tab1:
    col_input, col_result = st.columns([1.5, 1]) # แบ่งสัดส่วนคอลัมน์ซ้ายใหญ่กว่าขวา
    
    with col_input:
        st.subheader("กรอกข้อความรีวิว")
        user_input = st.text_area("ข้อความต้นฉบับจากลูกค้า:", height=150, placeholder="พิมพ์รีวิว หรือ ข้อร้องเรียนที่นี่... (เช่น สินค้าส่งไวมาก แต่กล่องบุบ 😡 www.test.com)")
        analyze_btn = st.button("🔍 วิเคราะห์ความรู้สึก", type="primary", use_container_width=True)
    
    with col_result:
        st.subheader("ผลลัพธ์การวิเคราะห์")
        if analyze_btn:
            if user_input.strip() == "":
                st.warning("⚠️ กรุณากรอกข้อความก่อนทำการวิเคราะห์")
            else:
                with st.spinner('กำลังใช้ AI ประมวลผล...'):
                    time.sleep(1) # จำลองเวลาประมวลผล
                    sentiment, conf_score, cleaned = predict_sentiment(user_input, model_choice)
                    
                    # กล่องแสดงผลลัพธ์สีสันสดใส
                    if sentiment == "เชิงบวก":
                        st.success(f"### 🟢 {sentiment}")
                    elif sentiment == "เชิงลบ":
                        st.error(f"### 🔴 {sentiment}")
                    else:
                        st.info(f"### 🟡 {sentiment}")
                    
                    st.metric(label="ความแม่นยำ (Confidence)", value=f"{conf_score:.2%}")
                    
                    # ส่วนขยายดูข้อความที่ถูกทำความสะอาดแล้ว
                    with st.expander("🛠️ ดูข้อความหลังผ่าน Preprocessing"):
                        st.code(cleaned, language="text")

# ---------------------------------------------------------
# แท็บที่ 2: การอัปโหลดไฟล์ CSV และแสดงสถิติ (Data Visualization)
# ---------------------------------------------------------
with tab2:
    st.subheader("อัปโหลดไฟล์รีวิวเพื่อวิเคราะห์และดูสถิติ")
    uploaded_file = st.file_uploader("ลากไฟล์ หรือ เลือกไฟล์ .csv ที่มีคอลัมน์เก็บข้อความ", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("**📌 ตัวอย่างข้อมูลเบื้องต้น:**")
        st.dataframe(df.head(3), use_container_width=True)
        
        col_select, col_empty = st.columns([1, 2])
        with col_select:
            text_column = st.selectbox("เลือกคอลัมน์ที่เก็บ 'ข้อความรีวิว':", df.columns)
            process_btn = st.button("🚀 เริ่มวิเคราะห์ชุดข้อมูลทั้งหมด", type="primary", use_container_width=True)
        
        if process_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            confidences = []
            cleaned_texts = []
            
            total_rows = len(df)
            
            # วนลูปวิเคราะห์พร้อมอัปเดตหลอด Progress
            for i, text in enumerate(df[text_column]):
                status_text.text(f"กำลังประมวลผลแถวที่ {i+1} / {total_rows}...")
                sent, conf, cleaned = predict_sentiment(str(text), model_choice)
                results.append(sent)
                confidences.append(conf)
                cleaned_texts.append(cleaned)
                
                # อัปเดตหลอดความคืบหน้า (Progress Bar)
                progress_bar.progress((i + 1) / total_rows)
                time.sleep(0.05) # หน่วงเวลาจำลองให้เห็นหลอดวิ่ง
            
            status_text.empty()
            progress_bar.empty()
            
            # เก็บผลลัพธ์ลง DataFrame
            df['Cleaned_Text'] = cleaned_texts
            df['Sentiment'] = results
            df['Confidence'] = confidences
            
            st.success("✅ วิเคราะห์ข้อมูลเสร็จสิ้น!")
            
            # --- ส่วนแสดงกราฟสถิติ (Analytics Dashboard) ---
            st.divider()
            st.subheader("📊 แดชบอร์ดสรุปผล (Analytics Dashboard)")
            
            # แบ่งเป็น 2 คอลัมน์สำหรับแสดงกราฟ
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # กราฟ Pie Chart แสดงสัดส่วนความรู้สึก
                sentiment_counts = df['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig_pie = px.pie(
                    sentiment_counts, 
                    names='Sentiment', 
                    values='Count', 
                    title="สัดส่วนความรู้สึกของลูกค้า (Sentiment Distribution)",
                    color='Sentiment',
                    color_discrete_map={'เชิงบวก':'#2ca02c', 'เชิงลบ':'#d62728', 'เชิงกลาง':'#ff7f0e'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with chart_col2:
                # กราฟ Bar Chart แสดงค่าความมั่นใจเฉลี่ย
                avg_conf = df.groupby('Sentiment')['Confidence'].mean().reset_index()
                fig_bar = px.bar(
                    avg_conf, 
                    x='Sentiment', 
                    y='Confidence', 
                    title="ค่าความมั่นใจเฉลี่ยของโมเดลแต่ละคลาส (Avg Confidence)",
                    text_auto='.2%',
                    color='Sentiment',
                    color_discrete_map={'เชิงบวก':'#2ca02c', 'เชิงลบ':'#d62728', 'เชิงกลาง':'#ff7f0e'}
                )
                fig_bar.update_layout(yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # --- ตารางผลลัพธ์สุดท้ายและปุ่มดาวน์โหลด ---
            st.write("**📋 ตารางผลลัพธ์ที่วิเคราะห์แล้ว:**")
            st.dataframe(df, use_container_width=True)
            
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดผลลัพธ์ทั้งหมด (CSV)",
                data=csv_data,
                file_name='sentiment_analyzed_with_stats.csv',
                mime='text/csv',
                type="primary"
            )