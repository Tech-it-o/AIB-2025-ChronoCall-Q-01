import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig # นำเข้า PeftModel และ PeftConfig
import json

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

# (Optional) หากคุณต้องการรวม LoRA เข้ากับ Base Model อย่างถาวร (ทำให้ได้โมเดลใหม่ที่ใหญ่ขึ้น)
# แต่คุณอาจจะยังไม่จำเป็นต้องทำขั้นตอนนี้หากต้องการแค่รัน Inference
# model = model.merge_and_unload()

def model_answer(messages):
    print("Model is running...")
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
    print("Testing the model...")
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
        {"role": "user", "content": "เอานัดทำการบ้านกับเพื่อนพฤหัสที่จะถึงนี้ออก"},
    ]
    
    response = model_answer(messages)
    print(response)