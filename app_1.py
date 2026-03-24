import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunking import embedding_model
import os
from dotenv import load_dotenv
from openai import OpenAI


st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")
st.title("Legal RAG Chatbot")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_rag_system():

    vectorstore = FAISS.load_local(
        "Section_FAISS_Index_Openai",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore

vectorstore = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question ")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # RAG Retrieval
    results = vectorstore.similarity_search(query, k=4)

    context_blocks = []
    for doc in results:
        title = doc.metadata.get("section_title", "Unknown Section")
        context_blocks.append(f"[{title}]\n{doc.page_content}")

    context = "\n\n".join(context_blocks)

    # RAG Prompt Synthesis 
    prompt = f"""
You are a legal assistant chatbot.

Use only the context below to answer.
If the answer is not in the context, say:
"information not found in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    # -------- LLM Reasoning --------
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    answer = response.output_text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
