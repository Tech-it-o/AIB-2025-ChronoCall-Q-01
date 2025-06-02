import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM # สมมติว่าคุณใช้ Hugging Face transformers

# 1. โหลด Tokenizer และ Model (ควรโหลดเพียงครั้งเดียว)
#    อาจใช้ st.cache_resource เพื่อประสิทธิภาพที่ดีกว่า
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("your_model_name_or_path") # เปลี่ยนเป็นชื่อโมเดลของคุณ
    model = AutoModelForCausalLM.from_pretrained("your_model_name_or_path") # เปลี่ยนเป็นชื่อโมเดลของคุณ
    # อาจมีการย้ายโมเดลไป GPU ถ้ามี
    # if torch.cuda.is_available():
    #     model.to("cuda")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# กำหนด TOOLS หากคุณต้องการใช้ (อันนี้ขึ้นอยู่กับโมเดลและการใช้งานของคุณ)
TOOLS = [] # ตัวอย่าง: TOOLS = [{"name": "tool_name", "description": "tool_description"}]

def generate_model_answer(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=TOOLS,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")#.to(model.device) # ตรวจสอบเรื่อง .to(model.device) ถ้าใช้ CPU
    outputs = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]
    return output_text

st.title("แชทบอทง่ายๆ ด้วย Streamlit")

# การจัดการข้อความแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงข้อความแชทที่มีอยู่
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ช่องรับ input จากผู้ใช้
prompt = st.chat_input("พูดคุยกับบอท...")
if prompt:
    # เพิ่มข้อความผู้ใช้ลงใน session_state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # สร้างข้อความตอบกลับจากโมเดล
    with st.chat_message("assistant"):
        with st.spinner("กำลังคิด..."):
            # แปลงข้อความใน session_state ให้อยู่ในรูปแบบที่ model_answer ต้องการ
            chat_messages = []
            for msg in st.session_state.messages:
                # ตรวจสอบบทบาท: user หรือ assistant
                if msg["role"] == "user":
                    chat_messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_messages.append({"role": "assistant", "content": msg["content"]})
            
            # เรียกใช้ฟังก์ชันที่สร้างคำตอบจากโมเดล
            response = generate_model_answer(chat_messages)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})