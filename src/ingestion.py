import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Path input dokumen PDF & output FAISS
DATA_PATH = "data/raw"
INDEX_PATH = "artifacts/faiss_index"

def run_ingestion():
    print(f"Memuat dokumen PDF dari {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    print(f"Berhasil memuat {len(docs)} dokumen.")

    # 2. Chunking teks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    print(f"Berhasil memotong teks menjadi {len(chunks)} chunks.")

    # 3. Model Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 4. Buat Vectorstore dan Simpan
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"Indeks FAISS berhasil disimpan di {INDEX_PATH}")

if __name__ == "__main__":
    run_ingestion()