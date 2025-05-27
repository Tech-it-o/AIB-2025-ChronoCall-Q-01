import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- 1. การตั้งค่าโมเดลและโหลดโมเดล ---
# ใช้ Qwen1.5-0.5B-Chat ซึ่งเป็นโมเดลขนาดเล็กที่สุด
# และเป็นตัวเลือกที่ดีที่สุดสำหรับการ Deploy บน Streamlit Cloud Free Tier
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat" 

# บน Streamlit Cloud จะเป็น CPU และมักจะมี RAM จำกัด
# สำหรับ 0.5B อาจจะไม่ต้องใช้ 4-bit quantization ก็ได้
# แต่ถ้าแอปยัง Crash หรือทำงานช้ามาก ให้ลองตั้ง USE_4BIT = True
DEVICE = "cpu"
DTYPE = torch.float32 
USE_4BIT = False # เริ่มต้นด้วย False เพื่อความเร็วที่ดีกว่า (ถ้า RAM พอ)
                  # ถ้ามีปัญหา OOM ให้เปลี่ยนเป็น True และเพิ่ม bitsandbytes ใน requirements.txt

print(f"กำลังพยายามโหลดโมเดล: {MODEL_NAME}")
print(f"ใช้ Device: {DEVICE}")
if USE_4BIT:
    print("เปิดใช้งานการโหลดโมเดลแบบ 4-bit quantization (ต้องติดตั้ง bitsandbytes)")
else:
    print("ไม่ได้ใช้ 4-bit quantization")

@st.cache_resource
def load_qwen_model(model_name, device, dtype, use_4bit):
    """
    โหลด Tokenizer และ Model ของ Qwen และแคชไว้
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # กำหนด Quantization Config ถ้าใช้ 4-bit
        quantization_config = None
        if use_4bit:
            from transformers import BitsAndBytesConfig # import ที่นี่เพื่อหลีกเลี่ยง error ถ้าไม่ได้ใช้
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_use_double_quant=True,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None, # บน Streamlit Cloud จะไม่ใช้ cuda
            quantization_config=quantization_config, # ใส่ quantization config ที่นี่
        )

        model.eval() 
        
        st.success(f"✔️ โมเดล '{model_name}' โหลดสำเร็จบน {DEVICE.upper()} "
                   f"{'ด้วย 4-bit quantization' if use_4bit else ''}.")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.info(f"โมเดล '{model_name}' อาจยังใหญ่เกินไป หรือมีปัญหาการติดตั้ง")
        st.warning("โปรดตรวจสอบ:")
        st.markdown("- **`MODEL_NAME`** ถูกต้องหรือไม่")
        st.markdown("- หาก **`USE_4BIT = True`** ต้องแน่ใจว่า **`bitsandbytes`** อยู่ใน `requirements.txt`")
        st.markdown("- หากยังคงล้มเหลว, โมเดลอาจเกินขีดจำกัดหน่วยความจำของ Streamlit Cloud จริงๆ")
        st.stop()

# โหลดโมเดลเมื่อแอปเริ่มต้น
tokenizer, model = load_qwen_model(MODEL_NAME, DEVICE, DTYPE, USE_4BIT)

# --- 2. ส่วนหัวของ Streamlit App ---
st.set_page_config(page_title="Qwen Chatbot (Small)")
st.title("🤖 Qwen Chatbot")
st.caption(f"ขับเคลื่อนโดยโมเดล: **{MODEL_NAME}** บน **{DEVICE.upper()}** "
           f"{'ด้วย 4-bit quantization' if USE_4BIT else ''}")
st.write("สวัสดีครับ! มีอะไรให้ช่วยไหมครับ? (นี่คือ Qwen ขนาดเล็ก)")

# --- 3. การจัดการประวัติการสนทนา ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # เพิ่ม System message สำหรับ Qwen chat template
    st.session_state.messages.append({"role": "system", "content": "You are a helpful AI assistant."})
    # เพิ่มข้อความต้อนรับจาก Assistant
    st.session_state.messages.append({"role": "assistant", "content": "สวัสดีครับ! มีอะไรให้ช่วยไหมครับ?"})

# --- 4. แสดงประวัติการสนทนาใน UI ---
for message in st.session_state.messages:
    if message["role"] != "system": # ไม่ต้องแสดง system message ใน UI
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- 5. รับ Input จากผู้ใช้และสร้างการตอบกลับ ---
prompt = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() 
        full_response = ""
        
        chat_history_for_model = []
        for msg in st.session_state.messages:
            chat_history_for_model.append({"role": msg["role"], "content": msg["content"]})

        input_text = tokenizer.apply_chat_template(
            chat_history_for_model,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        with st.spinner("กำลังคิด..."): 
            try:
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512, 
                    do_sample=True,    
                    top_p=0.8,         
                    temperature=0.7,   
                    repetition_penalty=1.0 
                )

                generated_response_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
                bot_response = tokenizer.decode(generated_response_ids, skip_special_tokens=True)

                if "<|im_end|>" in bot_response:
                    bot_response = bot_response.split("<|im_end|>")[0].strip()
                if "<|im_start|>assistant\n" in bot_response:
                    bot_response = bot_response.split("<|im_start|>assistant\n")[-1].strip()
                bot_response = bot_response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

                full_response = bot_response
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}")
                full_response = f"ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล: {e}. " \
                                "นี่อาจเกิดจากขีดจำกัดหน่วยความจำบน Streamlit Cloud."
                message_placeholder.markdown(full_response)
                gc.collect() 

    st.session_state.messages.append({"role": "assistant", "content": full_response})