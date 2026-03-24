import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from chunking import embedding_model

st.set_page_config(page_title="Legal RAG Chatbot")
st.title("Legal RAG Chatbot")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_rag_system():

    vectorstore = FAISS.load_local(
        "SemanticSliding_FAISS_Index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore

vectorstore = load_rag_system()

query = st.text_input("Ask a question :")

if st.button("Search") and query:
    results = vectorstore.similarity_search(query, k=8)

    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""
You are a legal assistant.

Use only the context below to answer.
If not found, say: "information not found in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.responses.create(
        model="gpt-4.0-mini",
        input=prompt
    )

    answer = response.output_text

    st.subheader("Answer:")
    st.write(answer)
