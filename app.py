import os, json, time
import requests
import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")
st.title("💬 Messenger-Style Chatbot (Free)")

# --------- Theme toggle ---------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
toggle = st.toggle("🌗 Dark / Light Mode", value=(st.session_state.theme=="dark"))
st.session_state.theme = "dark" if toggle else "light"

DARK_CSS = """
<style>
body { background:#0e1117; color:#fff; }
.user-bubble { background:#1f6feb; color:#fff; padding:10px 14px; margin:8px; max-width:70%;
               border-radius:18px 18px 4px 18px; align-self:flex-end; }
.bot-bubble  { background:#2d333b; color:#fff; padding:10px 14px; margin:8px; max-width:70%;
               border-radius:18px 18px 18px 4px; align-self:flex-start; }
.chat-wrap   { display:flex; flex-direction:column; }
</style>
"""
LIGHT_CSS = """
<style>
body { background:#f5f5f5; color:#000; }
.user-bubble { background:#0084ff; color:#fff; padding:10px 14px; margin:8px; max-width:70%;
               border-radius:18px 18px 4px 18px; align-self:flex-end; }
.bot-bubble  { background:#e4e6eb; color:#000; padding:10px 14px; margin:8px; max-width:70%;
               border-radius:18px 18px 18px 4px; align-self:flex-start; }
.chat-wrap   { display:flex; flex-direction:column; }
</style>
"""
st.markdown(DARK_CSS if st.session_state.theme=="dark" else LIGHT_CSS, unsafe_allow_html=True)

# --------- HF Inference API settings ---------
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODEL = st.sidebar.selectbox(
    "الموديل المستضاف (مجاني بحدود):",
    ["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    index=0,
)
MAX_NEW_TOKENS = st.sidebar.slider("طول الرد", 32, 256, 160, 16)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
TOP_P = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

st.caption("يعمل عبر Hugging Face Inference API لتقليل استهلاك الذاكرة في Streamlit Cloud.")

API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def call_hf(prompt: str, retries: int = 3) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(MAX_NEW_TOKENS),
            "temperature": float(TEMPERATURE),
            "top_p": float(TOP_P),
            "return_full_text": False
        }
    }
    for i in range(retries):
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 503:
            # model is warming up; wait and retry
            time.sleep(3)
            continue
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return out[0]["generated_text"]
        return json.dumps(out)[:300]
    return "الخدمة مشغولة حاليًا، جرّبي بعد لحظات."

# --------- Chat state ---------
if "history" not in st.session_state:
    st.session_state.history = []

with st.container():
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    for role, msg in st.session_state.history:
        klass = "user-bubble" if role == "user" else "bot-bubble"
        st.markdown(f"<div class='{klass}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

user_msg = st.text_input("اكتبي رسالتك هنا…", placeholder="مثال: عرف نفسك باختصار")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    prompt = f"انت مساعد ودود بترد بالعربي المختصر.\nUser: {user_msg}\nAssistant:"
    with st.spinner("بيفكر…"):
        try:
            reply = call_hf(prompt) if HF_TOKEN else "من فضلك أضيفي HF_TOKEN في Secrets."
        except Exception as e:
            reply = f"حصل خطأ: {e}"
    st.session_state.history.append(("bot", reply))
    st.rerun()

# أدوات إضافية
col1, col2 = st.columns(2)
if col1.button("🧹 مسح المحادثة"):
    st.session_state.clear()
    st.rerun()
if col2.download_button(
    "⬇️ تنزيل المحادثة",
    data="\n\n".join([f"{r.upper()}: {m}" for r, m in st.session_state.history]),
    file_name="chat_history.txt",
    mime="text/plain"
):
    pass
