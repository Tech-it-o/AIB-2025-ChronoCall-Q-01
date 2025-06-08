import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# อ่าน secrets
sender = st.secrets["email"]["sender"]
receiver = st.secrets["email"]["receiver"]
password = st.secrets["email"]["password"]

def send_email(user_email):
    subject = "New Beta Tester - ChronoCall-Q"
    body = f"""\
    มีผู้สนใจเข้าร่วม Beta Test
    อีเมล: {user_email}
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        return True
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        return False

# UI
st.title("ขอเข้าร่วม Beta Test - ChronoCall-Q")

email_input = st.text_input("กรอกอีเมลของคุณ")

if st.button("ส่งคำขอ"):
    if email_input:
        if send_email(email_input):
            st.success("ส่งคำขอเรียบร้อยแล้ว! เราจะติดต่อกลับเร็ว ๆ นี้ ❤️")
    else:
        st.warning("กรุณากรอกอีเมลก่อนกดส่ง")
