import streamlit as st
import requests
import time

# =====================================
# PAGE CONFIG + GLOBAL THEME
# =====================================
st.set_page_config(
    page_title="Smart Chatbot",
    layout="wide",
)

# =====================================
# Custom CSS (Professional Dashboard Style)
# =====================================
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
    font-family: "Segoe UI", sans-serif;
}

.header-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: -10px;
    color: #1f2937;
}

.subtext {
    color: #6b7280;
    margin-bottom: 25px;
}

.chat-card {
    background-color: white;
    padding: 18px 24px;
    border-radius: 14px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
}

.user-bubble {
    background-color: #3b82f6;
    padding: 12px 15px;
    color: white;
    max-width: 60%;
    border-radius: 14px;
    margin: 6px 0px 6px auto;
    font-size: 16px;
}

.bot-bubble {
    background-color: #e5e7eb;
    padding: 12px 15px;
    color: #111827;
    max-width: 60%;
    border-radius: 14px;
    margin: 6px 0px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# =====================================
# HF API CALL (Light & Free)
# =====================================
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}
MODEL = "google/flan-t5-base"

def ask_inference(prompt):
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    payload = {"inputs": prompt}

    for _ in range(3):  # retries
        r = requests.post(url, headers=HEADERS, json=payload)
        if r.status_code == 503:
            time.sleep(2)
            continue
        r.raise_for_status()
        output = r.json()
        if isinstance(output, list):
            return output[0]["generated_text"]
        return str(output)
    return "الخدمة مشغولة حالياً."


# =====================================
# UI HEADER
# =====================================
st.markdown('<div class="header-title">💬 Smart Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">واجهة احترافية مستوحاة من Dashboard الخاصة بك.</div>', unsafe_allow_html=True)


# =====================================
# Chat State
# =====================================
if "history" not in st.session_state:
    st.session_state.history = []


# =====================================
# Chat Display
# =====================================
chat_area = st.container()

with chat_area:
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)


# =====================================
# User Input
# =====================================
user_msg = st.text_input("اكتب رسالتك هنا:", "")

if user_msg:
    st.session_state.history.append(("user", user_msg))

    with st.spinner("جاري التفكير..."):
        prompt = f"جاوب بالعربي الواضح: {user_msg}"
        bot_reply = ask_inference(prompt)

    st.session_state.history.append(("bot", bot_reply))
    st.rerun()
