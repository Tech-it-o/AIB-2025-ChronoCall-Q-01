from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
import json
from peft import PeftModel, PeftConfig

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="ChronoCall-Q Chatbot", page_icon="🤖")
st.title("ChronoCall-Q Chatbot")
st.caption("Ask me anything! I am a helpful assistant from Alibaba Cloud.")

# --- Load TOOLS from JSON ---
# Ensure 'tools.json' exists in the same directory as this script.
# If your model doesn't strictly use tools, this can be an empty JSON array `[]`.
try:
    with open('tools.json', 'r', encoding='utf-8') as f:
        TOOLS = json.load(f)
except FileNotFoundError:
    st.error("Error: tools.json not found. Please ensure it's in the same directory.")
    TOOLS = [] # Fallback to an empty list if file is not found

# --- Model Loading (with Streamlit caching for efficiency) ---
# @st.cache_resource ensures the model is loaded only once across reruns,
# which is crucial for performance in Streamlit apps.
@st.cache_resource
def load_model_and_tokenizer():
    base_model_id = "Qwen/Qwen3-0.6B"
    lora_model_id = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData"

    st.write(f"Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    st.write(f"Loading base model from: {base_model_id}")
    # Using 'auto' for device_map lets transformers handle placement,
    # often leveraging GPU if available, or CPU otherwise.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32, # Ensure consistency if you had issues with 'auto'
        device_map="auto",
        trust_remote_code=True
    )

    st.write(f"Loading LoRA adapter from: {lora_model_id}")
    # Merge LoRA adapter into the base model
    model = PeftModel.from_pretrained(model, lora_model_id)
    # Optional: If you want to merge the LoRA weights into the base model permanently
    # model = model.merge_and_unload()
    
    st.success("Model and Tokenizer loaded successfully!")
    return tokenizer, model

# Load model and tokenizer when the app starts or refreshes
tokenizer, model = load_model_and_tokenizer()

# --- Model Inference Function ---
def model_answer(messages):
    # Apply chat template for Qwen, including tool definitions
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS, # Pass the loaded tools here
        enable_thinking=False
    )
    
    # Prepare inputs for the model
    # Ensure inputs are on the same device as the model
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate model output
    outputs = model.generate(**inputs, max_new_tokens=512)
    
    # Decode the generated tokens to text
    # [len(text):] slices the output to remove the input prompt,
    # showing only the model's generated response.
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]
    
    return output_text

# --- Streamlit Chat Interface ---

# Initialize chat history in Streamlit's session state if it doesn't exist.
# This ensures that the conversation persists across user inputs.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
        {"role": "assistant", "content": "Hello! How can I help you today?"} # Initial greeting from the assistant
    ]

# Display existing chat messages from the history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# --- Chat Input for User ---
# st.chat_input creates a persistent input field at the bottom of the page.
if prompt := st.chat_input("Type your message here..."):
    # Add user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get model's response
    with st.chat_message("assistant"):
        with st.spinner("ChronoCall-Q is thinking..."):
            # Pass the *entire* chat history to the model to maintain context
            full_response = model_answer(st.session_state.messages)
            st.markdown(full_response) # Display the model's response
    
    # Add assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})