import streamlit as st
import google.generativeai as genai

try:
    api_key = st.secrets["gemini_api_key"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Gemini API Key ไม่ถูกตั้งค่า! โปรดเพิ่ม 'gemini_api_key' ใน Streamlit Secrets หรือไฟล์ .streamlit/secrets.toml")
    st.stop()


st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("Chatbot Gemini")
st.caption("พิมพ์")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        chat_history_for_gemini = []
        for message in st.session_state.messages:

            gemini_role = "model" if message["role"] == "assistant" else message["role"]
            chat_history_for_gemini.append({"role": gemini_role, "parts": [message["content"]]})

        chat = model.start_chat(history=chat_history_for_gemini[:-1])

        response = chat.send_message(prompt)

        assistant_response = response.text

    except Exception as e:
        assistant_response = f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini หรือ API: {e}\nโปรดลองอีกครั้งในภายหลัง"

    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)