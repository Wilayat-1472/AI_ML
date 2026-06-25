import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Mental Health Support Chatbot", page_icon="💙")
st.title("💙 Mental Health Support Chatbot")
st.markdown("*A supportive chatbot for emotional wellness*")

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("./mental-health-chatbot-final")
    tokenizer = AutoTokenizer.from_pretrained("./mental-health-chatbot-final")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

def get_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_length=100, temperature=0.7,
            top_k=50, top_p=0.95, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Chatbot:" in response:
        response = response.split("Chatbot:", 1)[-1].strip()
    return response

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! How are you feeling today?"})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Share how you're feeling..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        formatted = f"User: {prompt}\nChatbot:"
        response = get_response(formatted)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
