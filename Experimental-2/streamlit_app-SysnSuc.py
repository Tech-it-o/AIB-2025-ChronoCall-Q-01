import streamlit as st
import datetime
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import ast

st.set_page_config(page_title="Test-ChronoCall-Q", page_icon="🗓️")

st.markdown("""
    <style>
    /* เปลี่ยนกรอบหลักให้เป็นสีม่วง */
    div[data-baseweb="input"] > div {
        border: 2px solid #a020f0 !important;
        border-radius: 6px;
        padding: 2px;
        box-shadow: none !important;
    }

    /* ตอน focus แล้ว */
    div[data-baseweb="input"] > div:focus-within {
        border: 2px solid #a020f0 !important;
        box-shadow: 0 0 0 2px rgba(160, 32, 240, 0.3) !important;
    }

    /* input ด้านในไม่ให้แสดงเงาสีแดงเลย */
    input {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ลบพวกกรอบแดงที่แอบซ่อนอยู่ */
    .css-1cpxqw2, .css-1d391kg, .css-1y4p8pa {
        border: 2px solid #a020f0 !important;
        box-shadow: none !important;
    }

    /* force กล่อง input ให้ไม่ใช้สีแดงแม้จะ error */
    div:has(input:focus) {
        border-color: #a020f0 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- Model ---

def convert_to_dict(text):
    match = re.search(r"<tool_call>\n(.*?)\n</tool_call>", text)

    if match:
        tool_dict_str = match.group(1)
        try:
            result = ast.literal_eval(tool_dict_str)
            return(result)
        except Exception as e:
            return({})
    else:
        return({})

# --- Model ---

model_name_or_path = "TechitoTamani/Qwen3-0.6B_FinetuneWithMyData-Merged"

with open('tools.json', 'r', encoding='utf-8') as f:
    TOOLS = json.load(f)

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16, # ระบุ dtype ที่ชัดเจน
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def get_model_answer(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]
    return output_text

# --- Calendar ---

SCOPES = ['https://www.googleapis.com/auth/calendar']

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_flow():
    return Flow.from_client_config(
        {
            "web": {
                "client_id": st.secrets["client_id"],
                "client_secret": st.secrets["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [st.secrets["redirect_uri"]],
            }
        },
        scopes=SCOPES,
        redirect_uri=st.secrets["redirect_uri"]
    )

def generate_auth_url(flow):
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline', include_granted_scopes='true')
    return auth_url

def create_service(creds):
    return build("calendar", "v3", credentials=creds)

# --- Streamlit ---

def main():

    # ดึง query params
    params = st.query_params
    code = params.get("code", None)

    # ถ้ายังไม่ได้ login
    if "credentials" not in st.session_state:
        if code:
            flow = create_flow()
            try:
                flow.fetch_token(code=code)
                creds = flow.credentials
                st.session_state["credentials"] = {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                    "token_uri": creds.token_uri,
                    "client_id": creds.client_id,
                    "client_secret": creds.client_secret,
                    "scopes": creds.scopes
                }
                st.success("🎉 ล็อกอินสำเร็จ! พร้อมใช้งานแล้ว")
                st.rerun()
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
                return
        else:
            flow = create_flow()
            auth_url = generate_auth_url(flow)

            st.markdown(f"""
                <!-- ชื่อ -->
                <div class="fade-in-title custom-title">ChronoCall-Q</div>

                <!-- ปุ่ม ล่าช้า -->
                <div class="fade-in-button login-button">
                    <a href="{auth_url}" target="_blank" rel="noopener noreferrer">
                        <button>Login with Google</button>
                    </a>
                </div>
            """, unsafe_allow_html=True)

            st.stop()

    # ถ้า login แล้ว
    creds = Credentials(**st.session_state["credentials"])
    service = create_service(creds)

    st.title("ChronoCall-Q")
    st.caption("พิมพ์คำสั่งของคุณด้านล่างแล้วกด Enter หรือปุ่ม 'ยืนยัน'")

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2025-02-01.\n\nCurrent Day: Saturday."},
        ]

    user_input = st.text_input("พิมพ์คำสั่งที่นี่ แล้วกด Enter หรือกดปุ่มยืนยัน", value=st.session_state.user_input, key="input")

    submit_button = st.button("ยืนยัน")

    if (user_input and user_input != st.session_state.user_input) or submit_button:

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("โมเดลกำลังคิด..."):
            full_conversation_for_model = st.session_state.messages
            response = get_model_answer(full_conversation_for_model)

        st.session_state.messages.append({"role": "assistant", "content": response})

        func_call_dict = convert_to_dict(response)
        st.success(f"Qwen: {func_call_dict}")
        st.session_state.user_input = ""

    # with st.form("event_form"):
    #     summary = st.text_input("หัวข้อกิจกรรม", "ประชุมทีม")
    #     location = st.text_input("สถานที่", "Google Meet")
    #     start_date = st.date_input("วันที่เริ่ม", datetime.date.today())
    #     end_date = st.date_input("วันที่สิ้นสุด", datetime.date.today())
    #     submitted = st.form_submit_button("เพิ่มกิจกรรม")

    # if submitted:
    #     event = {
    #         'summary': summary,
    #         'location': location,
    #         'start': {
    #             'date': start_date.strftime("%Y-%m-%d"),
    #             'timeZone': 'Asia/Bangkok',
    #         },
    #         'end': {
    #             'date': end_date.strftime("%Y-%m-%d"),
    #             'timeZone': 'Asia/Bangkok',
    #         },
    #     }
    #     created_event = service.events().insert(calendarId='primary', body=event).execute()
    #     st.success(f"✅ เพิ่มกิจกรรมสำเร็จ: [คลิกดูใน Calendar]({created_event.get('htmlLink')})")

if __name__ == "__main__":
    main()