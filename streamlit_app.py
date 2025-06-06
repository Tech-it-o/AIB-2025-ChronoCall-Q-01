from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import json

st.set_page_config(page_title="Test-ChronoCall-Q", page_icon="🤖")
st.title("ChronoCall-Q")
st.caption("พิมพ์คำสั่งของคุณด้านล่างแล้วกด Enter หรือปุ่ม 'ยืนยัน'")

model_name_or_path = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData-Merged"

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_model_answer(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]
    return output_text

# --- Streamlit ---

st.markdown("""
    <style>
    /* เปลี่ยนกรอบหลักให้เป็นสีม่วง */
    div[data-baseweb="input"] > div {
        border: 2px solid #a020f0 !important;
        border-radius: 6px;
        padding: 2px;
        box-shadow: none !important;
    }

    /* ตอน focus แล้ว */
    div[data-baseweb="input"] > div:focus-within {
        border: 2px solid #a020f0 !important;
        box-shadow: 0 0 0 2px rgba(160, 32, 240, 0.3) !important;
    }

    /* input ด้านในไม่ให้แสดงเงาสีแดงเลย */
    input {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ลบพวกกรอบแดงที่แอบซ่อนอยู่ */
    .css-1cpxqw2, .css-1d391kg, .css-1y4p8pa {
        border: 2px solid #a020f0 !important;
        box-shadow: none !important;
    }

    /* force กล่อง input ให้ไม่ใช้สีแดงแม้จะ error */
    div:has(input:focus) {
        border-color: #a020f0 !important;
    }
    </style>
""", unsafe_allow_html=True)


if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
    ]

user_input = st.text_input("พิมพ์คำสั่งที่นี่ แล้วกด Enter หรือกดปุ่มยืนยัน", value=st.session_state.user_input, key="input")

submit_button = st.button("ยืนยัน")

if (user_input and user_input != st.session_state.user_input) or submit_button:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("โมเดลกำลังคิด..."):
        full_conversation_for_model = st.session_state.messages
        response = get_model_answer(full_conversation_for_model)

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.success(f"Qwen: {response}")
    st.session_state.user_input = ""