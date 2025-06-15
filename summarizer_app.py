
import os
os.environ["USE_TF"] = "0"  # Use PyTorch only

import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("summarization", model="t5-small")

summarizer = load_model()

st.title("🧠 AI Text Summarizer")
st.write("Paste any long text below and get a concise summary!")

text = st.text_area("Enter your text here", height=300)

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
            st.success("✅ Summary:")
            st.write(summary[0]['summary_text'])
