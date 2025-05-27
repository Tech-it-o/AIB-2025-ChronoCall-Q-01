import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

# --- 1. การตั้งค่าโมเดลและโหลดโมเดล (ส่วนสำคัญ) ---
# **เปลี่ยนชื่อโมเดลตรงนี้สำหรับ Qwen3-4B**
# ตรวจสอบชื่อที่ถูกต้องบน Hugging Face Hub: https://huggingface.co/Qwen
# หากมีเวอร์ชัน -Chat จะเหมาะกับการทำ Chatbot มากกว่า
MODEL_NAME = "Qwen/Qwen3-4B-Chat" # <--- เปลี่ยนเป็นชื่อโมเดล Qwen3-4B-Chat ที่คุณต้องการใช้ (หรือ Qwen/Qwen3-4B ถ้าเป็น Base)

# ตั้งค่า precision และ device
# บน Streamlit Cloud จะเป็น CPU และเราจะพึ่งพา load_in_4bit
DEVICE = "cpu"
DTYPE = torch.float32 # การใช้ 4-bit quantization ไม่ได้ขึ้นอยู่กับ DTYPE ตรงๆ แต่ควรตั้งให้เป็น float32
USE_4BIT = True # <--- สำคัญมาก: เปิดใช้งานการโหลดแบบ 4-bit quantization สำหรับ RAM จำกัด

print(f"กำลังพยายามโหลดโมเดล: {MODEL_NAME}")
print(f"ใช้ Device: {DEVICE}")
if USE_4BIT:
    print("เปิดใช้งานการโหลดโมเดลแบบ 4-bit quantization (ต้องติดตั้ง bitsandbytes)")
else:
    print("ไม่ได้ใช้ 4-bit quantization (อาจต้องการ RAM/VRAM สูงมาก)")

@st.cache_resource
def load_qwen_model(model_name, device, dtype, use_4bit):
    """
    โหลด Tokenizer และ Model ของ Qwen และแคชไว้
    ใช้ load_in_4bit สำหรับการประหยัดหน่วยความจำ
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # กำหนด Quantization Config ถ้าใช้ 4-bit
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # NormalFloat 4-bit
                bnb_4bit_compute_dtype=torch.bfloat16, # ใช้ bfloat16 สำหรับการคำนวณภายใน
                bnb_4bit_use_double_quant=True, # เปิดใช้งาน double quantization
            )
        
        # โหลดโมเดล
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            quantization_config=quantization_config, # ใส่ quantization config ที่นี่
        )

        model.eval() # ตั้งค่าโมเดลเป็น evaluation mode
        
        st.success(f"✔️ โมเดล '{model_name}' โหลดสำเร็จบน {DEVICE.upper()} "
                   f"{'ด้วย 4-bit quantization' if use_4bit else ''}.")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.info(f"**โมเดล '{model_name}' (4B) มีขนาดใหญ่มาก** สำหรับ Streamlit Cloud Free Tier.")
        st.warning("โปรดตรวจสอบ:")
        st.markdown("- **`MODEL_NAME`** ถูกต้องหรือไม่")
        st.markdown("- **`USE_4BIT = True`** และ **`bitsandbytes`** อยู่ใน `requirements.txt`")
        st.markdown("- หากยังคงล้มเหลว, โมเดลนี้อาจเกินขีดจำกัดหน่วยความจำของ Streamlit Cloud.")
        st.markdown("- พิจารณาใช้โมเดลขนาดเล็กกว่า (เช่น `Qwen/Qwen1.5-0.5B-Chat` หรือ `1.8B-Chat`) หรือ Deploy บน Hugging Face Spaces (ที่มี GPU) แทน.")
        st.stop() # หยุดการทำงานของ Streamlit ถ้าโหลดโมเดลไม่ได้

# โหลดโมเดลเมื่อแอปเริ่มต้น
tokenizer, model = load_qwen_model(MODEL_NAME, DEVICE, DTYPE, USE_4BIT)

# --- 2. ส่วนหัวของ Streamlit App ---
st.set_page_config(page_title="Qwen3 Chatbot")
st.title("🤖 Qwen3 Chatbot")
st.caption(f"ขับเคลื่อนโดยโมเดล: **{MODEL_NAME}** บน **{DEVICE.upper()}** "
           f"{'ด้วย 4-bit quantization' if USE_4BIT else ''}")
st.write("ลองพิมพ์ข้อความเพื่อเริ่มสนทนา!")

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
    # 5.1 เพิ่มข้อความผู้ใช้เข้าในประวัติ
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 5.2 แสดงข้อความผู้ใช้ใน UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5.3 สร้างการตอบกลับจาก Qwen
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # ใช้ placeholder เพื่ออัปเดตข้อความได้
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
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        with st.spinner("กำลังคิด..."): # เพิ่ม spinner ระหว่างที่โมเดลกำลังประมวลผล
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
                # กรณีโมเดลสร้าง assistant role ซ้ำ
                if "<|im_start|>assistant\n" in bot_response:
                    bot_response = bot_response.split("<|im_start|>assistant\n")[-1].strip()
                # กรณีอื่นๆ ที่อาจมี artifact
                bot_response = bot_response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


                full_response = bot_response
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการสร้างคำตอบ: {e}")
                full_response = f"ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล: {e}. " \
                                "นี่อาจเกิดจากขีดจำกัดหน่วยความจำบน Streamlit Cloud."
                message_placeholder.markdown(full_response)
                gc.collect() # พยายามเคลียร์หน่วยความจำ

    # 5.4 เพิ่มข้อความ Chatbot เข้าในประวัติ
    st.session_state.messages.append({"role": "assistant", "content": full_response})