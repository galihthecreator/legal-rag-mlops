# src/pipeline.py
import torch
import logging
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from src.config_loader import load_config
from src.retriever import HybridParentRetriever
from src.reranker import LegalReranker

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LegalRAGPipeline:
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config
        
        # 1. Inisialisasi Retriever & Reranker
        logger.info("Inisialisasi Hybrid Retriever dan Reranker...")
        self.retriever = HybridParentRetriever(config=self.config)
        self.reranker = LegalReranker(config=self.config)

        # 2. Load Model Qwen2.5-7B (4-bit Quantization)
        model_name = self.config["model"]["name"]
        logger.info(f"Memuat LLM: {model_name} (4-bit load: {self.config['model']['load_in_4bit']})...")

        quantization_config = None
        if self.config["model"]["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Setup Pipeline HuggingFace
        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    def generate_hyde_document(self, query: str) -> str:
        """
        Tahap 1: HyDE (Hypothetical Document Embeddings)
        Membuat contoh jawaban/dokumen hukum hipotetis untuk meningkatkan kualitas pencarian dense vector.
        """
        logger.info("Membuat Hypothetical Document (HyDE)...")
        hyde_prompt = (
            f"Tuliskan pasal atau penjelasan hukum singkat dalam Bahasa Indonesia "
            f"yang secara langsung menjawab pertanyaan berikut.\n\nPertanyaan: {query}\n\nPenjelasan Hukum Hipotetis:"
        )
        output = self.llm(hyde_prompt)[0]["generated_text"]
        # Ambil hanya bagian hipotetis yang dihasilkan
        hypothetical_doc = output.replace(hyde_prompt, "").strip()
        return hypothetical_doc

    def _build_context(self, documents: List[Document]) -> str:
        """
        Menggabungkan dokumen konteks hasil reranking menjadi string tunggal.
        """
        context_blocks = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Dokumen Hukum")
            page = doc.metadata.get("page", "-")
            context_blocks.append(f"[Konteks {i} | Sumber: {source} Hal. {page}]\n{doc.page_content}")
        
        return "\n\n".join(context_blocks)

    def run(self, query: str, use_hyde: bool = True) -> Dict[str, Any]:
        """
        Menjalankan alur lengkap RAG: HyDE -> Hybrid Retrieval -> Reranking -> LLM Generation.
        """
        logger.info(f"Menerima Query: '{query}'")

        # 1. HyDE Transformation (Opsional/Konfigurabel)
        search_query = query
        if use_hyde:
            hypothetical_doc = self.generate_hyde_document(query)
            # Gabungkan query asli + dokumen hipotetis untuk pencarian kaya konteks
            search_query = f"{query} {hypothetical_doc}"

        # 2. Hybrid Retrieval (BM25 + FAISS pada Child Chunks -> Parent Docs)
        retrieved_docs = self.retriever.invoke(search_query)

        # 3. Reranking (Cross-Encoder BGE-M3)
        ranked_docs, is_fallback = self.reranker.rerank(query, retrieved_docs)

        # 4. Formulasi Prompt berdasarkan Kondisi Fallback
        if is_fallback or not ranked_docs:
            logger.warning("Memakai fallback prompt karena relevansi dokumen rendah.")
            system_prompt = (
                "Anda adalah asisten hukum Indonesia yang cermat. Informasi spesifik tidak ditemukan pada basis data hukum kami. "
                "Jawablah pertanyaan berikut secara umum berdasarkan pengetahuan hukum Anda, namun berikan penegasan/disclaimer "
                "bahwa jawaban ini tidak merujuk pada pasal tertentu dari dokumen internal.\n\n"
                f"Pertanyaan: {query}\n\nJawaban:"
            )
        else:
            context_str = self._build_context(ranked_docs)
            system_prompt = (
                "Anda adalah asisten hukum Indonesia yang profesional dan akurat. "
                "Jawab pertanyaan pengguna secara ringkas dan lugas HANYA berdasarkan konteks hukum di bawah ini. "
                "Sebutkan sumber pasal atau dokumennya jika tersedia dalam konteks.\n\n"
                f"--- KONTEKS HUKUM ---\n{context_str}\n---------------------\n\n"
                f"Pertanyaan: {query}\n\nJawaban Berdasarkan Konteks:"
            )

        # 5. LLM Inference
        logger.info("Menghasilkan jawaban dari LLM...")
        raw_output = self.llm(system_prompt)[0]["generated_text"]
        answer = raw_output.replace(system_prompt, "").strip()

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": ranked_docs,
            "is_fallback": is_fallback
        }


# Quick Test Pipeline
if __name__ == "__main__":
    rag_pipeline = LegalRAGPipeline()
    sample_query = "Apa sanksi pidana jika melanggar ketentuan pasal dalam undang-undang ini?"
    result = rag_pipeline.run(sample_query, use_hyde=True)

    print("\n" + "="*50)
    print(f"QUERY: {result['query']}")
    print(f"FALLBACK TRIGGERED: {result['is_fallback']}")
    print(f"JUMLAH KONTEKS DIGUNAKAN: {len(result['retrieved_documents'])}")
    print("="*50)
    print(f"JAWABAN LLM:\n{result['answer']}")
    print("="*50)