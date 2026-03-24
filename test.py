from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load vector DB
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    "Section_FAISS_Index_Openai",
    embedding_model,
    allow_dangerous_deserialization=True
)

questions = [
    "What is considered Confidential Information?",
    "What are the vendor’s disaster recovery obligations?",
    "How quickly must the vendor notify Fiserv after an incident?",
    "Is a pandemic response plan required?",
    "How often must the business continuity plan be tested?",
    "What happens if vendor loses customer data?"
]

def rag_answer(query):
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
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


print("\n=== RAG Evaluation ===\n")

for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}\n")
    ans = rag_answer(q)
    print("Answer:\n", ans)
    print("\n" + "-"*70)
