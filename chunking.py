from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import re
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

file_path = "Fiserv_XUS_CSA_11Sept2023_11Sept2024_Signed 1 1.pdf"

load_dotenv()

loader = PyPDFLoader(file_path)
docs = loader.load()

# Checking 
# print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# Recursive Character Chunking
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index=True

)"""

# chunks = text_splitter.split_documents(docs)

# print(len(chunks))
def section_chunking(docs, source_name="Fiserv_Contract.pdf"):
    full_text = "\n".join([doc.page_content for doc in docs])

    pattern = r'(ARTICLE\s+\d+[:.]|ANNEX\s+[A-Z0-9]+|EXHIBIT\s+[A-Z0-9]+|SCHEDULE\s+\d+|TABLE\s+\d+[:.]|[0-9]+(?:\.[0-9]+)+)'

    parts = re.split(pattern, full_text)

    chunks = []
    current = ""
    section_id = 0
    chunk_id = 0

    for part in parts:
        if re.match(pattern, part):
            if current.strip():
                title = current.strip().split("\n")[0][:120]

                chunks.append(
                    Document(
                        page_content=current.strip(),
                        metadata={
                            "source": source_name,
                            "section_id": section_id,
                            "section_title": title,
                            "chunk_id": chunk_id
                        }
                    )
                )

                section_id += 1
                chunk_id += 1

            current = part
        else:
            current += " " + part

    # save last section
    if current.strip():
        title = current.strip().split("\n")[0][:120]

        chunks.append(
            Document(
                page_content=current.strip(),
                metadata={
                    "source": source_name,
                    "section_id": section_id,
                    "section_title": title,
                    "chunk_id": chunk_id
                }
            )
        )

    return chunks

chunks = section_chunking(docs)

# BGE Embeddings
'''
embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5"
)

# for each chunk:
#     embed chunk
#     store vector in FAISS
#     attach metadata

vectorstore = FAISS.from_documents(chunks,embedding_model)

# saving faiss index
vectorstore.save_local("Section_FAISS_Index")

print("Successfully saved")
'''

# Open AI Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(chunks, embedding_model)

vectorstore.save_local("Section_FAISS_Index_Openai")

print("OpenAI FAISS index saved successfully!")