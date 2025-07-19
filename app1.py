import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env
load_dotenv()

# UI setup
st.set_page_config(page_title="CloudCrimBot 🔍")
st.title("CloudCrimBot 🔍")
st.markdown("Ask any question about Tamil Nadu crime data.")

# Load FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    "crime_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_input := st.chat_input("Ask about Tamil Nadu crime data..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve relevant chunks
    docs = db.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Use the following crime records to answer the question:
{context}

Question: {user_input}

If the answer is not in the data, say "Not available".
Answer:"""

    # Call TinyLlama via Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        answer = result.get("response", "⚠️ No answer.")
    except Exception as e:
        answer = f"⚠️ Error calling TinyLlama: {str(e)}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
