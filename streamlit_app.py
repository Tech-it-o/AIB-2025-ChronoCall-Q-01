import streamlit as st
import datetime
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ตั้ง Scope ที่ใช้เข้าถึง Google Calendar
SCOPES = ['https://www.googleapis.com/auth/calendar']

# ฟังก์ชัน login เพื่อรับ credentials
def login():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret_58410467583-f3udl0q7m082pejkjg01l40gsa8qngqn.apps.googleusercontent.com.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds

# UI สำหรับกรอกข้อมูล
st.title("🗓️ เพิ่มกิจกรรมลง Google Calendar")

with st.form("event_form"):
    summary = st.text_input("หัวข้อกิจกรรม", "ประชุมทีม")
    location = st.text_input("สถานที่", "Google Meet")
    start_date = st.date_input("วันที่เริ่ม")
    end_date = st.date_input("วันที่สิ้นสุด")
    submitted = st.form_submit_button("เพิ่มกิจกรรม")

if submitted:
    creds = login()
    service = build("calendar", "v3", credentials=creds)

    event = {
        'summary': summary,
        'location': location,
        'start': {
            'date': start_date.strftime("%Y-%m-%d"),
            'timeZone': 'Asia/Bangkok',
        },
        'end': {
            'date': end_date.strftime("%Y-%m-%d"),
            'timeZone': 'Asia/Bangkok',
        },
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    st.success(f"✅ เพิ่มกิจกรรมสำเร็จ: [คลิกดูใน Calendar]({created_event.get('htmlLink')})")
