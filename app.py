import streamlit as st
import pandas as pd
import time
import random # ใช้สำหรับจำลองค่า (Mockup) ในช่วงที่ยังไม่ได้เชื่อมต่อโมเดลจริง

# ตั้งค่าหน้าเพจ
st.set_page_config(
    page_title="Thai Sentiment Analysis & CRM",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# ส่วนจำลองการทำงานของโมเดล (แทนที่ด้วยโมเดลจริงในภายหลัง)
# ---------------------------------------------------------
def load_model(model_name):
    # ฟังก์ชันสำหรับโหลดโมเดล Bi-LSTM หรือ WangchanBERTa
    return model_name

def predict_sentiment(text, model):
    # ฟังก์ชันจำลองการทำนายผลลัพธ์ คลาสที่ได้คือ เชิงบวก เชิงกลาง และเชิงลบ
    classes = ["เชิงบวก (Positive)", "เชิงกลาง (Neutral)", "เชิงลบ (Negative)"]
    predicted_class = random.choice(classes)
    confidence = random.uniform(0.80, 0.99) # จำลองความแม่นยำเป้าหมายที่เกิน 80%
    return predicted_class, confidence

# ---------------------------------------------------------
# ส่วนอินเทอร์เฟซ (UI)
# ---------------------------------------------------------
st.title("📊 ระบบวิเคราะห์ความรู้สึกรีวิวลูกค้าภาษาไทย")
st.markdown("แอปพลิเคชันสำหรับทีมบริการลูกค้าและ CRM เพื่อคัดกรองปัญหาและจัดลำดับความสำคัญของข้อความรีวิว")

# แถบด้านข้างสำหรับตั้งค่าโมเดล
with st.sidebar:
    st.header("⚙️ การตั้งค่าโมเดล")
    # เลือกสถาปัตยกรรมที่ต้องการเปรียบเทียบตามโครงงาน
    model_choice = st.selectbox(
        "เลือกสถาปัตยกรรมโมเดล",
        ("WangchanBERTa (Transformer)", "Bi-directional LSTM (Bi-LSTM)")
    )
    st.info(f"กำลังใช้งาน: **{model_choice}**")

# แบ่งหน้าจอเป็น 2 แท็บหลัก: กรอกข้อความ และ อัปโหลดไฟล์
tab1, tab2 = st.tabs(["✍️ วิเคราะห์จากข้อความ (Text Input)", "📁 อัปโหลดไฟล์รีวิว (CSV Upload)"])

# แท็บที่ 1: การกรอกข้อความเดี่ยว
with tab1:
    st.subheader("ทดสอบวิเคราะห์ข้อความรีวิว")
    user_input = st.text_area("กรอกข้อความรีวิวของลูกค้าที่นี่:", height=150, placeholder="เช่น สินค้าจัดส่งรวดเร็วมาก แพ็คมาอย่างดี...")
    
    if st.button("🔍 วิเคราะห์ความรู้สึก", type="primary"):
        if user_input.strip() == "":
            st.warning("กรุณากรอกข้อความก่อนทำการวิเคราะห์")
        else:
            with st.spinner('กำลังประมวลผล...'):
                time.sleep(1) # จำลองเวลาประมวลผล
                sentiment, conf_score = predict_sentiment(user_input, model_choice)
                
                st.markdown("### ผลการวิเคราะห์:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if "บวก" in sentiment:
                        st.success(f"**ระดับความรู้สึก:** {sentiment}")
                    elif "ลบ" in sentiment:
                        st.error(f"**ระดับความรู้สึก:** {sentiment}")
                    else:
                        st.info(f"**ระดับความรู้สึก:** {sentiment}")
                
                with col2:
                    st.metric(label="ค่าความมั่นใจ (Confidence Score)", value=f"{conf_score:.2%}")

# แท็บที่ 2: การอัปโหลดไฟล์ CSV
with tab2:
    st.subheader("วิเคราะห์ข้อมูลรีวิวจำนวนมาก")
    st.markdown("อัปโหลดไฟล์ `.csv` ที่มีคอลัมน์ชื่อ `review` หรือ `text` เพื่อทำนายผลลัพธ์แบบกลุ่ม")
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("ตัวอย่างข้อมูลที่อัปโหลด:")
        st.dataframe(df.head())
        
        # สมมติว่าคอลัมน์ที่เก็บข้อความชื่อ 'text'
        text_column = st.selectbox("เลือกคอลัมน์ที่เก็บข้อความรีวิว:", df.columns)
        
        if st.button("🚀 เริ่มวิเคราะห์ชุดข้อมูล"):
            with st.spinner('กำลังวิเคราะห์ข้อมูลทั้งหมด...'):
                results = []
                confidences = []
                
                for text in df[text_column]:
                    sent, conf = predict_sentiment(str(text), model_choice)
                    results.append(sent)
                    confidences.append(conf)
                
                df['Sentiment Prediction'] = results
                df['Confidence Score'] = confidences
                
                st.success("วิเคราะห์ข้อมูลเสร็จสิ้น!")
                st.dataframe(df)
                
                # ปุ่มสำหรับดาวน์โหลดผลลัพธ์
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 ดาวน์โหลดผลการวิเคราะห์ (CSV)",
                    data=csv,
                    file_name='sentiment_analysis_results.csv',
                    mime='text/csv',
                )