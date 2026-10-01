# src/schemas.py
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator


class DocumentSchema(BaseModel):
    """
    Schema untuk memvalidasi teks dokumen hukum sebelum di-chunk & di-embed.
    """
    page_content: str = Field(..., min_length=10, description="Teks mentah dari halaman PDF hukum")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dokumen (sumber, nomor halaman, dll)")

    @field_validator("page_content")
    @classmethod
    def validate_content_not_empty(cls, value: str) -> str:
        # Bersihkan whitespace
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Teks halaman PDF tidak boleh kosong atau hanya berisi whitespace.")
        return cleaned


class ChunkSchema(BaseModel):
    """
    Schema untuk memvalidasi hasil chunking sebelum disimpan ke FAISS.
    """
    chunk_id: str = Field(..., description="Unique ID untuk chunk (misal: doc1_page2_chunk0)")
    text: str = Field(..., min_length=15, description="Teks hasil chunking")
    source_file: str = Field(..., description="Nama file PDF sumber")
    page_number: int = Field(..., ge=1, description="Nomor halaman (mulai dari 1)")

    @field_validator("text")
    @classmethod
    def clean_chunk_text(cls, value: str) -> str:
        # Normalisasi karakter aneh / newline berlebih dari ekstraksi PDF
        return " ".join(value.split())

class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Pertanyaan hukum dari pengguna", 
        example="Apa sanksi pidana jika melanggar ketentuan pasal undang-undang?"
    )
    use_hyde: bool = Field(
        default=True, 
        description="Apakah ingin menggunakan HyDE expansion?"
    )


class RetrievedDocumentResponse(BaseModel):
    content: str
    source: Optional[str] = "Dokumen Hukum"
    page: Optional[Any] = "-"


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_fallback: bool
    retrieved_documents: List[RetrievedDocumentResponse]