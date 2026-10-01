# src/api.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import QueryRequest, QueryResponse, RetrievedDocumentResponse
from src.pipeline import LegalRAGPipeline

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variable untuk menyimpan instance pipeline (Singleton Pattern)
rag_pipeline: LegalRAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager untuk memuat model berat (LLM, FAISS, Reranker)
    sekali saja saat server FastAPI dinyalakan.
    """
    global rag_pipeline
    logger.info("Mulai memuat Legal RAG Pipeline ke memori GPU/RAM...")
    try:
        rag_pipeline = LegalRAGPipeline()
        logger.info("Legal RAG Pipeline berhasil dimuat!")
    except Exception as e:
        logger.error(f"Gagal memuat RAG Pipeline: {e}")
        raise e
    yield
    # Cleanup saat server shutdown
    logger.info("Mematikan server FastAPI dan membersihkan resource...")


# Inisialisasi FastAPI App
app = FastAPI(
    title="Legal RAG API Service",
    description="API Service untuk sistem RAG Hukum Indonesia berbasis Qwen2.5-7B, Hybrid Search, dan Cross-Encoder Reranking.",
    version="1.0.0",
    lifespan=lifespan
)

# Setup CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah ke domain spesifik saat di lingkungan produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health Check"])
def root_check():
    """
    Endpoint sederhana untuk pengecekan ketersediaan service.
    """
    return {
        "status": "online",
        "service": "Legal RAG API Service",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Health check endpoint untuk Kubernetes / Docker container probing.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline belum siap atau gagal dimuat."
        )
    return {"status": "healthy", "pipeline_loaded": True}


@app.post("/api/v1/ask", response_model=QueryResponse, tags=["RAG Inference"])
def ask_legal_question(request: QueryRequest):
    """
    Endpoint utama untuk menjawab pertanyaan hukum pengguna.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline belum siap."
        )

    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query tidak boleh kosong."
        )

    try:
        logger.info(f"Menerima request API untuk query: '{request.query}'")
        
        # Jalankan RAG Pipeline
        result = rag_pipeline.run(query=request.query, use_hyde=request.use_hyde)

        # Format dokumen konteks untuk response
        formatted_docs = []
        for doc in result.get("retrieved_documents", []):
            formatted_docs.append(
                RetrievedDocumentResponse(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "Dokumen Hukum"),
                    page=doc.metadata.get("page", "-")
                )
            )

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            is_fallback=result["is_fallback"],
            retrieved_documents=formatted_docs
        )

    except Exception as e:
        logger.error(f"Error saat memproses query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan internal: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)