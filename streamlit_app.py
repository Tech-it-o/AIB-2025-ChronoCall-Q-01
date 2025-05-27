import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# st.write("Hello, Streamlit!")

# with st.chat_message("user"):
#     st.write("Hello, Streamlit!")

st.title("Streamlit Chat Example....")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type a message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Simulate a response from the assistant
    response = f"Echo: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)