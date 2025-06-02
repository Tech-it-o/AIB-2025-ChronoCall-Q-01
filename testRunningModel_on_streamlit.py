from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import json

st.set_page_config(page_title="Test-ChronoCall-Q Output", page_icon="🤖")
st.title("Output ChronoCall-Q")
st.caption("พิมพ์ตรงนี้ ")

model_name_or_path = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData"

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

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