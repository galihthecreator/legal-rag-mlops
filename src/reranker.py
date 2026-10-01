# src/reranker.py
import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from src.config_loader import load_config

logger = logging.getLogger(__name__)


class LegalReranker:
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config
        model_name = self.config["reranker"]["model_name"]
        self.top_k = self.config["reranker"]["top_k"]
        self.fallback_threshold = self.config["reranker"]["fallback_threshold"]

        logger.info(f"Memuat Reranker Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document]) -> Tuple[List[Document], bool]:
        """
        Menghitung ulang skor relevansi dokumen berdasarkan query menggunakan Cross-Encoder.
        Mengembalikan tuple: (list_dokumen_terurut, status_is_fallback)
        """
        if not documents:
            logger.warning("Tidak ada dokumen yang dikirim untuk di-rerank.")
            return [], True

        # Buat pasangan (query, document_content) untuk Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Hitung skor relevansi
        scores = self.model.predict(pairs)

        # Gabungkan dokumen dengan skornya
        doc_score_pairs = list(zip(documents, scores))
        
        # Urutkan berdasarkan skor tertinggi
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Ambil top-k dokumen
        top_ranked_pairs = doc_score_pairs[: self.top_k]
        
        # Cek apakah skor tertinggi memenuhi threshold minimum
        top_score = top_ranked_pairs[0][1] if top_ranked_pairs else 0.0
        is_fallback = top_score < self.fallback_threshold

        ranked_documents = [doc for doc, score in top_ranked_pairs]

        logger.info(f"Reranking selesai. Skor tertinggi: {top_score:.4f} | Fallback status: {is_fallback}")
        
        return ranked_documents, is_fallback


if __name__ == "__main__":
    # Test Reranker secara independen
    reranker = LegalReranker()
    sample_docs = [
        Document(page_content="Pasal 1: Setiap orang yang melanggar hukum akan dikenakan sanksi."),
        Document(page_content="Kucing adalah hewan mamalia pemakan daging.")
    ]
    docs, fallback = reranker.rerank("Apa sanksi melanggar hukum?", sample_docs)
    print(f"Hasil Rerank ({len(docs)} docs, fallback={fallback}):")
    for d in docs:
        print("-", d.page_content)