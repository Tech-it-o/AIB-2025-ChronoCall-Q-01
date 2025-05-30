from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name_or_path = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData-2"

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

def model_answer(messages):
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
    messages = [
        {"role": "system", "content": "คุณเป็นผู้ช่วยที่ตอบคำถามเกี่ยวกับข้อมูลที่ฉันให้ชื่อว่า อัลฟ่า"},
        {"role": "user", "content": "คุณชื่ออะไร"},
    ]
    
    response = model_answer(messages)
    print(response)