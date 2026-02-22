import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ================================
#           PAGE CONFIG
# ================================
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")

# ================================
#        THEME TOGGLE
# ================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme = st.session_state.theme

toggle = st.toggle("🌗 Dark / Light Mode", value=(theme == "dark"))
st.session_state.theme = "dark" if toggle else "light"
theme = st.session_state.theme


# ================================
#  CUSTOM CSS FOR FLAT BUBBLES
# ================================
dark_css = """
<style>
body {
    background-color: #0e1117;
    color: white;
}
.chat-container {
    padding: 10px;
    border-radius: 10px;
}
.user-bubble {
    background-color: #1f6feb;
    color: white;
    padding: 10px 14px;
    margin: 10px;
    max-width: 70%;
    border-radius: 18px 18px 4px 18px;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #2d333b;
    color: white;
    padding: 10px 14px;
    margin: 10px;
    max-width: 70%;
    border-radius: 18px 18px 18px 4px;
    align-self: flex-start;
}
</style>
"""

light_css = """
<style>
body {
    background-color: #f5f5f5;
    color: black;
}
.chat-container {
    padding: 10px;
    border-radius: 10px;
}
.user-bubble {
    background-color: #0084ff;
    color: white;
    padding: 10px 14px;
    margin: 10px;
    max-width: 70%;
    border-radius: 18px 18px 4px 18px;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #e4e6eb;
    color: black;
    padding: 10px 14px;
    margin: 10px;
    max-width: 70%;
    border-radius: 18px 18px 18px 4px;
    align-self: flex-start;
}
</style>
"""

st.markdown(dark_css if theme == "dark" else light_css, unsafe_allow_html=True)


# ================================
#       LOAD CHAT MODEL
# ================================
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # سريع ومفتوح المصدر

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    return tokenizer, model

tokenizer, model = load_model()


# ================================
#       CHAT HISTORY
# ================================
if "history" not in st.session_state:
    st.session_state.history = []

st.title("💬 Messenger-Style Free Chatbot")
st.caption("واجهة دردشة بابلز + ثيم Dark/Light — شغال 100% Free على Streamlit Cloud.")

chat_container = st.container()


# ================================
#      DISPLAY CHAT BUBBLES
# ================================
with chat_container:
    for role, msg in st.session_state.history:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)


# ================================
#       USER INPUT
# ================================
user_msg = st.text_input("اكتب رسالتك هنا...", key="input_field")

if user_msg:
    # Add user msg
    st.session_state.history.append(("user", user_msg))

    # Build prompt
    prompt = f"انت مساعد ذكي بترد بالعربي. المستخدم قال: {user_msg}\nالرد:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "الرد:" in reply:
        reply = reply.split("الرد:")[-1].strip()

    # Add bot reply
    st.session_state.history.append(("bot", reply))

    st.rerun()
