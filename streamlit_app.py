import streamlit as st
import google.generativeai as genai

# --- 1. การตั้งค่า Gemini API Key ---
# ให้ Streamlit อ่าน API Key จาก st.secrets
# st.secrets จะดึงค่าจากไฟล์ .streamlit/secrets.toml
# หรือจาก Streamlit Cloud Dashboard (แนะนำให้ตั้งค่าใน Dashboard เมื่อ deploy)
try:
    api_key = "AIzaSyBV__STA0thnAS6-RVol-VJDIZ-yhdqM2M"
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Gemini API Key ไม่ถูกตั้งค่า! โปรดเพิ่ม 'gemini_api_key' ใน Streamlit Secrets หรือไฟล์ .streamlit/secrets.toml")
    st.stop() # หยุดการทำงานของแอปถ้าไม่มี API Key

# --- 2. การตั้งค่าหน้าจอ Streamlit ---
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Chatbot ขับเคลื่อนโดย Gemini")
st.caption("พิมพ์คำถามของคุณเพื่อเริ่มต้นการสนทนา")

# --- 3. การจัดการ Session State สำหรับประวัติการสนทนา ---
# ตรวจสอบว่ามี "messages" ใน session_state หรือไม่ ถ้าไม่มีให้สร้างลิสต์ว่าง
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงผลข้อความที่มีอยู่ในประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- 4. การรับ input จากผู้ใช้ ---
prompt = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")

if prompt:
    # เพิ่มข้อความของผู้ใช้ลงใน session state และแสดงผลทันที
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- 5. การเรียกใช้ Gemini API ---
    try:
        # สร้างโมเดล Gemini. 'gemini-pro' เหมาะสำหรับการสนทนาข้อความ
        model = genai.GenerativeModel('gemini-1.5-flash')

        # เตรียมประวัติการสนทนาสำหรับ Gemini API
        # Gemini API คาดหวัง role เป็น "user" หรือ "model"
        # และ content เป็น list ของ dictionaries (parts)
        chat_history_for_gemini = []
        for message in st.session_state.messages:
            # แปลง "assistant" role ใน Streamlit ให้เป็น "model" สำหรับ Gemini
            gemini_role = "model" if message["role"] == "assistant" else message["role"]
            chat_history_for_gemini.append({"role": gemini_role, "parts": [message["content"]]})

        # เริ่มต้นการสนทนา (chat session) กับ Gemini
        # ส่งประวัติการสนทนาทั้งหมด ยกเว้นข้อความล่าสุดของผู้ใช้
        # เพราะข้อความล่าสุดจะถูกส่งเป็น input แยกต่างหากใน send_message
        chat = model.start_chat(history=chat_history_for_gemini[:-1])

        # ส่งข้อความของผู้ใช้ไปยัง Gemini และรอรับคำตอบ
        response = chat.send_message(prompt)

        # ดึงข้อความตอบกลับจาก Gemini
        assistant_response = response.text

    except Exception as e:
        assistant_response = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini หรือ API: {e}\nโปรดลองอีกครั้งในภายหลัง"

    # --- 6. แสดงผลคำตอบของ Chatbot และเพิ่มลงใน Session State ---
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)