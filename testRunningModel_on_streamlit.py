from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import json
from peft import PeftModel, PeftConfig

st.set_page_config(page_title="Test-ChronoCall-Q Output", page_icon="🤖")
st.title("Output ChronoCall-Q")
st.caption("พิมพ์ตรงนี้ ")

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

# 1. กำหนดชื่อ Base Model และ LoRA ของคุณ
base_model_id = "Qwen/Qwen3-0.6B" # หรือ "Qwen/Qwen3-0.6B-Chat"
lora_model_id = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData"

# 2. โหลด Tokenizer ของ Base Model
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. โหลด Base Model
# กำหนด device ที่จะใช้ (cuda สำหรับ GPU, cpu สำหรับ CPU)
device = "cpu"
print(f"Using device: {device}")

# สำหรับ Qwen3 อาจจะต้องใช้ trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype= torch.float32,
    device_map="auto", # ให้ PEFT จัดการการกระจายโมเดล
    trust_remote_code=True # Qwen บางเวอร์ชันต้องการตัวนี้
)

# 4. โหลด LoRA และผนวกเข้ากับ Base Model
# เนื่องจากคุณ push เฉพาะ LoRA ขึ้นไป LoRA ของคุณจึงทำงานเป็น PEFT adapter
model = PeftModel.from_pretrained(model, lora_model_id)

def model_answer(messages):
    print("Model is running...") # This will still print to the console where Streamlit is run
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]

    return(output_text)

if __name__ == "__main__":
    print("Testing the model...") # This will still print to the console where Streamlit is run
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
        {"role": "user", "content": "เอานัดทำการบ้านกับเพื่อนพฤหัสที่จะถึงนี้ออก"},
    ]
    
    response = model_answer(messages)
    
    # Display the response in Streamlit
    st.write(response)
    # Or for a more prominent display:
    # st.success(response) # For a green success box
    # st.info(response) # For a blue info box
    # st.markdown(f"**Model Response:** {response}") # If you want to use Markdown for formatting