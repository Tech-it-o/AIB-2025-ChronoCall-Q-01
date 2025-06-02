import streamlit as st
# import google.generativeai as genai # ไม่จำเป็นถ้าใช้ Qwen อย่างเดียว

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

# --- Load Model ---
model_name_or_path = "Qwen/Qwen3-0.6B"

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# --- Chat Function (model_answer) ---
def model_answer(messages_history): # เปลี่ยนชื่อ parameter ให้สื่อถึงประวัติ
    print("Model is running...")
    text = tokenizer.apply_chat_template(
        messages_history, # ใช้ messages_history ที่มี system prompt และ chat history
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    # ตัดส่วน prompt ออกจาก output เพื่อให้ได้เฉพาะส่วนที่โมเดลตอบ
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # หาส่วนของ text ที่โมเดลสร้างขึ้นใหม่
    # วิธีที่ง่ายที่สุดคือการหาส่วนที่เพิ่มเข้ามาหลังจาก input text
    # เนื่องจาก apply_chat_template จะรวมบทสนทนาทั้งหมดไว้
    # เราจึงต้องหาส่วนที่ต่อท้ายจากบทสนทนาสุดท้ายที่ถูกป้อนเข้าไป
    
    # เพื่อให้แน่ใจว่าได้เฉพาะส่วนที่โมเดลตอบ เราจะใช้การหาส่วนที่อยู่หลัง user prompt ล่าสุด
    # ซึ่งโดยปกติแล้ว tokenized output จะมีโครงสร้างคล้ายๆ แบบนี้:
    # <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n
    # เราต้องการส่วนที่อยู่หลัง <|im_start|>assistant\n
    
    # หาก Qwen ใช้ template ที่แตกต่างไป อาจต้องปรับการตัดข้อความ
    # แต่โดยทั่วไปแล้ว, tokenizer.decode(outputs[0])[len(text):] ก็น่าจะใช้ได้หาก model.generate
    # ไม่ได้คืนค่า prompt กลับมาด้วย หรือเราต้องการแค่ส่วนที่โมเดลเพิ่มเข้ามาจริงๆ
    
    # สำหรับความปลอดภัย เราจะลองหาตำแหน่งของข้อความที่อยู่หลังบทสนทนาล่าสุด
    # โดยการตัดส่วนที่โมเดล "เห็น" ออกไป
    
    # ส่วนใหญ่ tokenizer.batch_decode(outputs)[0][len(text):] ก็น่าจะใช้ได้
    # แต่หากมีปัญหาเรื่องการตัด text ลองพิจารณารูปแบบ output ของ Qwen โดยละเอียด
    # ในกรณีนี้ ผมจะใช้ len(text): เหมือนเดิม เพราะเป็นวิธีมาตรฐาน
    output_text_only_response = output_text[len(text):].strip()

    return output_text_only_response

# --- Chatbot UI ---
st.set_page_config(page_title="ChronoCall-Q Chatbot", page_icon="🤖")
st.title("Chatbot ChronoCall-Q")
st.caption("พิมพ์ตรงนี้ ")

if "messages" not in st.session_state:
    # เริ่มต้นด้วย system prompt
    st.session_state.messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-06-02.\n\nCurrent Day: Monday."}
    ]

# แสดงข้อความแชทในประวัติ (ยกเว้น system prompt)
for message in st.session_state.messages:
    # ไม่ต้องแสดง system message ใน UI เพราะมันเป็นเหมือนการตั้งค่าเบื้องหลัง
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

prompt = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")

if prompt:
    # เพิ่มข้อความผู้ใช้เข้าในประวัติ
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # เรียก model_answer ด้วยประวัติทั้งหมด (รวม system prompt)
    # เนื่องจาก model_answer ต้องการประวัติทั้งหมดเพื่อสร้าง context
    assistant_response = model_answer(st.session_state.messages)

    # เพิ่มข้อความจากผู้ช่วย (โมเดล) เข้าในประวัติ
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.write(assistant_response)