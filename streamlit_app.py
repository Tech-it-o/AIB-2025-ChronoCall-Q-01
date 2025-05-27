import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc # สำหรับการจัดการหน่วยความจำ

# --- 1. การตั้งค่าโมเดลและโหลดโมเดล (ส่วนสำคัญ) ---
# เลือกโมเดล Qwen ที่ต้องการใช้
# สำหรับการทดสอบบน CPU หรือ RAM จำกัด: "Qwen/Qwen1.5-0.5B-Chat"
# ถ้ามี GPU หรือ RAM มากขึ้น: "Qwen/Qwen1.5-1.8B-Chat" หรือ "Qwen/Qwen1.5-7B-Chat"
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat" # คุณสามารถเปลี่ยนตรงนี้ได้ตามต้องการ

# ตรวจสอบว่ามี GPU หรือไม่ และตั้งค่า precision
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 # หรือ torch.float16 สำหรับ GPU ที่รองรับ
    print(f"Using GPU: {DEVICE} with {DTYPE} precision")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32 # CPU มักจะรองรับ float32 ได้ดีกว่า
    print(f"Using CPU with {DTYPE} precision. Inference might be slow for larger models.")

@st.cache_resource
def load_qwen_model(model_name, device, dtype):
    """โหลด Tokenizer และ Model ของ Qwen และแคชไว้"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None, # "auto" ให้ transformers จัดการ GPU
            # load_in_4bit=True # Uncomment บรรทัดนี้และ pip install bitsandbytes ถ้า RAM/VRAM ไม่พอ
        )
        if device == "cpu":
            model.to(device) # ย้ายโมเดลไป CPU ถ้าไม่ได้ใช้ device_map="auto"
        model.eval() # ตั้งค่าโมเดลเป็น evaluation mode
        st.success(f"Model {model_name} loaded successfully on {device}.")
        return tokenizer, model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}. Please check model name and available resources.")
        st.info("Try a smaller model like 'Qwen/Qwen1.5-0.5B-Chat' or enable `load_in_4bit=True` if you have `bitsandbytes` installed.")
        st.stop() # หยุดการทำงานของ Streamlit ถ้าโหลดโมเดลไม่ได้

# โหลดโมเดลเมื่อแอปเริ่มต้น
tokenizer, model = load_qwen_model(MODEL_NAME, DEVICE, DTYPE)

# --- 2. ส่วนหัวของ Streamlit App ---
st.set_page_config(page_title="Qwen Chatbot")
st.title("🤖 Qwen Chatbot (Powered by Streamlit & Hugging Face)")
st.caption(f"🚀 Using model: **{MODEL_NAME}** on **{DEVICE.upper()}**")

# --- 3. การจัดการประวัติการสนทนา ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # เพิ่ม System message สำหรับ Qwen chat template
    st.session_state.messages.append({"role": "system", "content": "You are a helpful AI assistant."})
    # เพิ่มข้อความต้อนรับจาก Assistant
    st.session_state.messages.append({"role": "assistant", "content": "สวัสดีครับ! มีอะไรให้ช่วยไหมครับ?"})

# --- 4. แสดงประวัติการสนทนาใน UI ---
for message in st.session_state.messages:
    # ไม่ต้องแสดง system message ใน UI แต่เก็บไว้ใน session_state สำหรับโมเดล
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- 5. รับ Input จากผู้ใช้และสร้างการตอบกลับ ---
prompt = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")
if prompt:
    # 5.1 เพิ่มข้อความผู้ใช้เข้าในประวัติ
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5.2 แสดงข้อความผู้ใช้ใน UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5.3 สร้างการตอบกลับจาก Qwen
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # เตรียม List ของข้อความสำหรับโมเดล (รวม system message)
        chat_history_for_model = []
        for msg in st.session_state.messages:
            chat_history_for_model.append({"role": msg["role"], "content": msg["content"]})

        # ใช้ tokenizer.apply_chat_template เพื่อสร้าง prompt ที่ถูกต้องสำหรับ Qwen
        input_text = tokenizer.apply_chat_template(
            chat_history_for_model,
            tokenize=False,
            add_generation_prompt=True # สำคัญสำหรับ Qwen เพื่อให้โมเดลรู้ว่าถึงตา assistant แล้ว
        )

        # เตรียม input ให้กับโมเดล
        model_inputs = tokenizer([input_text], return_tensors="pt").to(DEVICE)

        try:
            # เริ่มต้นการสร้างข้อความตอบกลับจากโมเดล
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512, # กำหนดความยาวสูงสุดของคำตอบ
                do_sample=True,    # เปิดใช้งานการสุ่มเพื่อความหลากหลาย
                top_p=0.8,         # ควบคุมความหลากหลายของคำตอบ
                temperature=0.7,   # ควบคุมความคิดสร้างสรรค์
                repetition_penalty=1.0 # ป้องกันการตอบซ้ำซาก
            )

            # ถอดรหัสส่วนที่โมเดลสร้างขึ้นมาใหม่เท่านั้น
            generated_response_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
            bot_response = tokenizer.decode(generated_response_ids, skip_special_tokens=True)

            # Qwen บางครั้งอาจสร้าง output ที่มี tag หรือ system message ติดมาด้วย
            # เราต้องตัดส่วนที่ไม่ใช่คำตอบจริงๆ ออกไป
            if "<|im_end|>" in bot_response:
                bot_response = bot_response.split("<|im_end|>")[0].strip()
            if "<|im_start|>" in bot_response:
                 bot_response = bot_response.split("<|im_start|>")[-1].replace("assistant", "").strip()

            full_response = bot_response
            message_placeholder.markdown(full_response)

        except torch.cuda.OutOfMemoryError:
            st.error("GPU out of memory! Please try a smaller model or close other applications.")
            full_response = "Error: GPU out of memory. Cannot generate response."
            message_placeholder.markdown(full_response)
            torch.cuda.empty_cache()
            gc.collect() # พยายามเคลียร์หน่วยความจำ
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            full_response = f"Error: {e}"
            message_placeholder.markdown(full_response)

    # 5.4 เพิ่มข้อความ Chatbot เข้าในประวัติ
    st.session_state.messages.append({"role": "assistant", "content": full_response})