from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # ต้อง import torch ด้วย
import streamlit as st
import json

st.set_page_config(page_title="Test-ChronoCall-Q Output", page_icon="🤖")
st.title("ChronoCall-Q Output")
st.caption("พิมพ์ข้อความของคุณด้านล่างแล้วกด Enter หรือปุ่ม 'ส่ง'")

# model_name_or_path = "Qwen/Qwen3-0.6B"

# # โหลด TOOLS จาก tools.json (ควรมีไฟล์นี้อยู่ในไดเรกทอรีเดียวกัน)
# with open('tools.json', 'r', encoding='utf-8') as f:
#     TOOLS = json.load(f)

# # โหลด Tokenizer และ Model (จะทำแค่ครั้งเดียวเมื่อแอปเริ่มทำงาน)
# @st.cache_resource # ใช้แคชเพื่อไม่ให้โหลดโมเดลซ้ำทุกครั้งที่มีการรีเฟรช
# def load_model_and_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         torch_dtype=torch.bfloat16, # ระบุ dtype ที่ชัดเจน
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     return tokenizer, model

# tokenizer, model = load_model_and_tokenizer()

# def get_model_answer(messages):
#     # print("Model is running...") # อันนี้จะแสดงใน console ที่รัน streamlit
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         tools=TOOLS,
#         enable_thinking=False
#     )
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     # ใช้ no_grad เพื่อประหยัดหน่วยความจำเมื่อไม่ต้องการคำนวณ gradient
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=512)
#     output_text = tokenizer.batch_decode(outputs)[0][len(text):]
#     return output_text

# # --- ส่วนของการจัดการแชทใน Streamlit ---

# # เริ่มต้นประวัติการสนทนาหากยังไม่มี
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
#         # ไม่ต้องแสดงข้อความ system นี้ในหน้าจอ แต่จำเป็นสำหรับโมเดล
#     ]

# # แสดงประวัติการสนทนาบนหน้าจอ
# for message in st.session_state.messages:
#     if message["role"] != "system": # ไม่แสดงข้อความ system บนหน้าจอ
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# # ช่องป้อนข้อความสำหรับผู้ใช้
# if prompt := st.chat_input("พิมพ์ข้อความของคุณที่นี่..."):
#     # เพิ่มข้อความผู้ใช้เข้าในประวัติการสนทนา
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # แสดงข้อความผู้ใช้บนหน้าจอ
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # ให้โมเดลตอบกลับ
#     with st.spinner("โมเดลกำลังคิด..."): # แสดงสถานะว่ากำลังโหลด
#         # ส่งประวัติการสนทนาทั้งหมดให้โมเดลเพื่อรักษา context
#         full_conversation_for_model = st.session_state.messages
#         response = get_model_answer(full_conversation_for_model)

#     # เพิ่มข้อความจากโมเดลเข้าในประวัติการสนทนา
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     # แสดงข้อความจากโมเดลบนหน้าจอ
#     with st.chat_message("assistant"):
#         st.markdown(response)