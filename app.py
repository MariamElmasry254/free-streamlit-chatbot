import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================= UI =================
st.set_page_config(page_title="Smart Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.chat-container {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
}
.user-msg {
    background-color: #2E2E2E;
    padding: 10px;
    margin: 8px;
    border-radius: 10px;
}
.bot-msg {
    background-color: #333333;
    padding: 10px;
    margin: 8px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Smart Free Chatbot")
st.write("Chatbot يعمل بالكامل مجانًا باستخدام Streamlit Cloud.")

# ================= Model =================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # سريع ومفتوح المصدر
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    return tokenizer, model

tokenizer, model = load_model()

# ================= Chat Logic =================
if "history" not in st.session_state:
    st.session_state.history = []

# Show history
for sender, msg in st.session_state.history:
    role_class = "user-msg" if sender == "user" else "bot-msg"
    st.markdown(f"<div class='{role_class}'>{msg}</div>", unsafe_allow_html=True)

# Input
user_input = st.text_input("اكتب سؤالك هنا:", placeholder="اكتب أي سؤال...")

if user_input:
    st.session_state.history.append(("user", user_input))

    inputs = tokenizer(
        f"You are an assistant who replies in Arabic. User: {user_input}\nAssistant:",
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.split("Assistant:")[-1].strip()

    st.session_state.history.append(("bot", reply))
    st.rerun()
