import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import requests
import json

# Configuration
def load_config():
    return {
        "OPENROUTER_API_KEY": "",
        "CHROMA_DB_PATH": "medical_db"
    }

config = load_config()

# Utility Functions
def get_relevant_chunks(query, collection, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]


def generate_response(query, context):
    headers = {
        "X-Title": "Medical Chatbot",
        "Authorization": f"Bearer {config['OPENROUTER_API_KEY']}"
    }
    prompt = f"""You are a helpful AI assistant that answers questions based strictly on the provided medical context.
If the answer cannot be found in the context, say 'I cannot answer this question based on the provided document.'

Context:\n{' '.join(context)}\n
Question: {query}\n
Answer:"""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "meta-llama/llama-3.3-8b-instruct:free",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.set_page_config(page_title="Doctor's AI Assistant", layout="centered")

# Custom CSS for minimal subtle theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .chat-box {
        background-color: #2d2d2d;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .user {
        background-color: #2c3e50;
        align-self: flex-end;
    }
    .bot {
        background-color: #34495e;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Greeting and Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("👩‍⚕️ Dr. AI Assistant")
name = st.text_input("Enter your name", value="Doctor")
greeting = f"Hello Dr. {name}, how can I assist you today?"
st.info(greeting)

# Initialize DB
client = chromadb.PersistentClient(path=config['CHROMA_DB_PATH'])
collection = client.get_collection("pdf_collection")

# Chat Input
user_query = st.chat_input("Type your question...")
if user_query:
    with st.spinner("Generating response..."):
        chunks = get_relevant_chunks(user_query, collection)
        bot_response = generate_response(user_query, chunks)
        # Append messages
        st.session_state.messages.append((user_query, bot_response))

# Display chat
for user_msg, bot_msg in st.session_state.messages:
    st.markdown(f"<div class='chat-box user'>**You:** {user_msg}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-box bot'>**Assistant:** {bot_msg}</div>", unsafe_allow_html=True)
