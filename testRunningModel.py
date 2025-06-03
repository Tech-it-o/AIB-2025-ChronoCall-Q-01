from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name_or_path = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData"

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