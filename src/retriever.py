# src/retriever.py
import os
import pickle
import logging
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class HybridParentRetriever:
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config

        embed_cfg = self.config.get("embeddings") or self.config.get("embedding", {})
        model_name = embed_cfg.get("model_name", "BAAI/bge-m3")
        device = embed_cfg.get("device", "cpu")

        logger.info(f"Memuat model embedding: {model_name} pada {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

        paths = self.config["paths"]
        faiss_dir = paths["faiss_index_dir"]
        docstore_path = paths.get("parent_docstore_path", "artifacts/parent_docstore.pkl")

        # 1. Load FAISS Vectorstore
        logger.info(f"Memuat FAISS index dari {faiss_dir}...")
        self.vectorstore = FAISS.load_local(
            faiss_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        top_k_faiss = self.config["retriever"].get("top_k_faiss", 10)
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k_faiss})

        # 2. Cek keberadaan Parent Docstore (BM25)
        if os.path.exists(docstore_path):
            logger.info(f"Memuat parent docstore dari {docstore_path}...")
            with open(docstore_path, "rb") as f:
                self.docstore = pickle.load(f)

            all_docs = list(self.docstore.values()) if isinstance(self.docstore, dict) else self.docstore
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = self.config["retriever"].get("top_k_bm25", 10)

            weights = self.config["retriever"].get("weights", [0.5, 0.5])
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=weights
            )
            logger.info("Hybrid (BM25 + FAISS) Retriever berhasil diinisialisasi.")
        else:
            logger.warning(
                f"File '{docstore_path}' tidak ditemukan. "
                "Sistem berjalan sementara dalam mode FAISS Dense Retriever saja."
            )
            self.ensemble_retriever = faiss_retriever

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.ensemble_retriever.invoke(query)

    def invoke(self, query: str) -> List[Document]:
        return self.ensemble_retriever.invoke(query)