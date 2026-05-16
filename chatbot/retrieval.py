from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chatbot.config import AppSettings


STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "can",
    "could",
    "for",
    "help",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "where",
    "who",
    "why",
}

IGNORE_PHRASES = [
    "tell me in detail",
    "explain in detail",
    "give me details",
    "tell me more",
    "please explain",
    "please provide details",
    "in detail",
]

INTENT_EXPANSIONS = {
    "quit": ["resign", "resignation", "separation", "notice period", "full and final"],
    "masters": ["mtech", "m.tech", "higher education", "post graduate"],
    "mba": ["master of business administration"],
    "bonus": ["annual performance bonus", "apb", "incentive"],
    "insurance": ["mediclaim", "health insurance", "hospitalization"],
    "refer": ["referral", "employee referral", "referral bonus"],
    "complain": ["complaint", "grievance", "escalation"],
    "password": ["reset password", "laptop password", "login"],
}

SECTION_BOOST_TERMS = {
    "full and final",
    "higher education",
    "mba",
    "mtech",
    "notice period",
    "resignation",
    "separation",
}

IMPORTANT_KEYWORDS = {
    "cafeteria",
    "holiday",
    "jira",
    "menu",
    "onboarding",
    "password",
    "vpn",
}

CONTEXT_REMOVE_PATTERNS = [
    "QUICK REFERENCE",
    "RELATED TOPICS",
    "END OF RELATED TOPICS",
    "To find information about",
    "See Section",
]


def normalize_question(question: str) -> str:
    clean_question = question.lower()
    for phrase in IGNORE_PHRASES:
        clean_question = clean_question.replace(phrase, "")

    clean_question = re.sub(r"[^a-z0-9.\s-]", " ", clean_question)
    clean_question = re.sub(r"\s+", " ", clean_question).strip()

    words = [
        word
        for word in clean_question.split()
        if word not in STOP_WORDS and len(word) > 1
    ]
    return " ".join(words)


def expand_question(original_question: str, normalized_question: str) -> str:
    expanded_terms = [normalized_question]
    original_lower = original_question.lower()

    for intent, keywords in INTENT_EXPANSIONS.items():
        if intent in original_lower:
            expanded_terms.extend(keywords)

    return " ".join(term for term in expanded_terms if term).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.]+", text.lower())


def clean_context_for_llm(context: str) -> str:
    cleaned_lines = []
    for line in context.splitlines():
        if any(pattern.lower() in line.lower() for pattern in CONTEXT_REMOVE_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


class HybridRetriever:
    def __init__(
        self,
        chunks: list[str],
        vector_store: FAISS,
        bm25: BM25Okapi,
        reranker: CrossEncoder,
        settings: AppSettings,
    ) -> None:
        self.chunks = chunks
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings

    @classmethod
    def from_chunks(cls, chunks: list[str], settings: AppSettings) -> "HybridRetriever":
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            cache_folder=settings.embedding_cache_folder,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        index_path = Path(settings.faiss_index_path)
        index_signature = _index_signature(chunks, settings.embedding_model_name)
        if index_path.exists() and _index_signature_matches(index_path, index_signature):
            try:
                vector_store = FAISS.load_local(
                    str(index_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                vector_store = FAISS.from_texts(chunks, embeddings)
                vector_store.save_local(str(index_path))
                _write_index_signature(index_path, index_signature)
        else:
            vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store.save_local(str(index_path))
            _write_index_signature(index_path, index_signature)

        tokenized_chunks = [tokenize(chunk) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        reranker = CrossEncoder(settings.reranker_model_name)
        return cls(chunks, vector_store, bm25, reranker, settings)

    def retrieve_context(self, question: str) -> str:
        normalized_question = normalize_question(question)
        expanded_question = expand_question(question, normalized_question)
        query_tokens = tokenize(expanded_question)

        if not query_tokens:
            return ""

        candidate_chunks = self._collect_candidates(expanded_question, query_tokens)
        ranked_chunks = self._rerank(question, candidate_chunks)

        if not ranked_chunks:
            return ""

        context = "\n\n".join(chunk for chunk, _score in ranked_chunks)
        context = clean_context_for_llm(context)
        return context[: self.settings.max_context_chars]

    def _collect_candidates(self, expanded_question: str, query_tokens: list[str]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(chunk: str) -> None:
            if chunk and chunk not in seen:
                candidates.append(chunk)
                seen.add(chunk)

        keyword_matches = sorted(
            self._keyword_matches(expanded_question, query_tokens),
            key=lambda item: item[1],
            reverse=True,
        )
        for chunk, _score in keyword_matches[: self.settings.keyword_top_k]:
            add_candidate(chunk)

        bm25_scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.settings.bm25_top_k]
        for index in ranked_indices:
            if bm25_scores[index] > 0:
                add_candidate(self.chunks[index])

        docs = self.vector_store.similarity_search(
            expanded_question,
            k=self.settings.semantic_top_k,
        )
        for doc in docs:
            add_candidate(doc.page_content)

        return candidates

    def _keyword_matches(
        self,
        expanded_question: str,
        query_tokens: list[str],
    ) -> list[tuple[str, int]]:
        query_text = " ".join(query_tokens)
        meaningful_tokens = [token for token in query_tokens if len(token) > 2]
        matches: list[tuple[str, int]] = []

        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = 0

            for token in meaningful_tokens:
                if token in chunk_lower:
                    score += 1

            for term in SECTION_BOOST_TERMS:
                if term in expanded_question and term in chunk_lower:
                    score += 2

            if query_text and query_text in chunk_lower:
                score += 4

            for keyword in IMPORTANT_KEYWORDS:
                if keyword in expanded_question and keyword in chunk_lower:
                    score += 2

            if score >= 2:
                matches.append((chunk, score))

        return matches

    def _rerank(self, question: str, candidate_chunks: list[str]) -> list[tuple[str, float]]:
        if not candidate_chunks:
            return []

        pairs = [(question, chunk) for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(candidate_chunks, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        selected: list[tuple[str, float]] = []
        for chunk, score in ranked:
            score_value = float(score)
            if score_value < self.settings.min_rerank_score:
                continue
            selected.append((chunk, score_value))
            if len(selected) >= self.settings.rerank_top_k:
                break

        return selected


def _index_signature(chunks: list[str], embedding_model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(embedding_model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\0")
        digest.update(chunk.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _index_signature_matches(index_path: Path, signature: str) -> bool:
    metadata_path = index_path / "index_meta.json"
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return metadata.get("signature") == signature


def _write_index_signature(index_path: Path, signature: str) -> None:
    index_path.mkdir(parents=True, exist_ok=True)
    metadata_path = index_path / "index_meta.json"
    metadata_path.write_text(
        json.dumps({"signature": signature}, indent=2),
        encoding="utf-8",
    )
