from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json
import os
from dotenv import load_dotenv

load_dotenv()

file_path = "Fiserv_XUS_CSA_11Sept2023_11Sept2024_Signed 1 1.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

semantic_chunks = semantic_splitter.split_documents(docs)

final_chunks = []
metadata_export = []

for i in range(len(semantic_chunks)):
    current = semantic_chunks[i]

    combined_text = current.page_content

    if i + 1 < len(semantic_chunks):
        combined_text += "\n\n" + semantic_chunks[i + 1].page_content

    metadata = {
        "source": file_path,
        "chunk_id": i,
        "semantic_group": i,
        "page": current.metadata.get("page"),
        "start_index": current.metadata.get("start_index")
    }

    final_chunks.append(
        Document(
            page_content=combined_text,
            metadata=metadata
        )
    )

    metadata_export.append({
        "chunk_id": i,
        "text_preview": combined_text[:300],
        "metadata": metadata
    })

vectorstore = FAISS.from_documents(final_chunks, embeddings)
vectorstore.save_local("SemanticSliding_FAISS_Index")

with open("semantic_chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_export, f, indent=2)

print("Semantic sliding FAISS index saved!")

