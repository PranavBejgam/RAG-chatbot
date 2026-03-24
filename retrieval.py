from langchain_community.vectorstores import FAISS
import faiss
from chunking import embedding_model

vectorstore = FAISS.load_local(
    "SemanticSliding_FAISS_Index",
    embedding_model,
    allow_dangerous_deserialization=True
)

query = input("Enter your Question here : ")

results = vectorstore.similarity_search(query,k=3)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n")
    print(doc.page_content)
    print("\nMetadata:", doc.metadata)