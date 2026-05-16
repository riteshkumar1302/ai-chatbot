from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    pdf_folder: str = os.getenv("PDF_FOLDER", "pdfs")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    embedding_cache_folder: str = os.getenv("EMBEDDING_CACHE_FOLDER", ".cache/embeddings")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    keyword_top_k: int = int(os.getenv("KEYWORD_TOP_K", "5"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "5"))
    semantic_top_k: int = int(os.getenv("SEMANTIC_TOP_K", "6"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "3"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "2400"))
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "-5.0"))


def load_settings() -> AppSettings:
    return AppSettings()


def get_groq_api_key(secrets: Any | None = None) -> str | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key

    if secrets is None:
        return None

    try:
        value = secrets.get("GROQ_API_KEY", "")
    except Exception:
        return None

    value = str(value).strip()
    return value or None
