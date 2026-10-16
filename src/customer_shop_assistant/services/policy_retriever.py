import logging
import math
import re
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Module-level cache to avoid redundant embedding calls to Ollama
_INDEX_CACHE: dict[str, dict[str, Any]] = {}


class PolicyChunk(BaseModel):
    """Retrieved policy knowledge chunk with source citation."""

    doc_name: str = Field(..., description="Source policy filename")
    doc_title: str = Field(default="", description="Main document title")
    doc_code: str | None = Field(
        default=None, description="Document code (e.g. POL-SHIP-01)"
    )
    section_title: str = Field(..., description="Markdown section heading")
    content: str = Field(..., description="Text content of policy section")
    score: float = Field(default=0.0, description="Relevance similarity score")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


VIETNAMESE_STOP_WORDS = {
    "là",
    "và",
    "của",
    "cho",
    "các",
    "có",
    "được",
    "bao",
    "nhiêu",
    "này",
    "khi",
    "trong",
    "với",
    "như",
    "thế",
    "nào",
    "gì",
    "sao",
    "đến",
    "từ",
    "ra",
    "vào",
    "lại",
    "đã",
    "sẽ",
    "đang",
    "mình",
    "bạn",
    "shop",
    "hỏi",
    "giúp",
}


class PolicyRetriever:
    """RAG retriever indexing and fetching store policy markdown documents.

    Uses dense semantic embeddings (nomic-embed-text via Ollama) with seamless
    fallback to sublinear TF-IDF vectorization.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        policies_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.policies_dir = policies_dir or (self.settings.DATA_DIR / "policies")
        self.chunks: list[PolicyChunk] = []
        self._dense_embeddings: list[list[float]] = []
        self._tfidf_embeddings: list[list[float]] = []
        self._vocabulary: list[str] = []
        self._idf: dict[str, float] = {}
        self._indexed = False
        self._using_dense = False
        self._embeddings_model: OllamaEmbeddings | None = None

    def index_documents(self) -> int:
        """Parse markdown files in policies directory into chunks and index them."""
        cache_key = f"{self.policies_dir}:{self.settings.EMBEDDING_PROVIDER}"
        if cache_key in _INDEX_CACHE:
            cached = _INDEX_CACHE[cache_key]
            self.chunks = cached["chunks"]
            self._dense_embeddings = cached["dense"]
            self._using_dense = cached["using_dense"]
            self._embeddings_model = cached["model"]
            self._vocabulary = cached["vocab"]
            self._idf = cached["idf"]
            self._tfidf_embeddings = cached["tfidf"]
            self._indexed = True
            return len(self.chunks)

        self.chunks = []
        if not self.policies_dir.exists():
            return 0

        all_texts: list[str] = []

        for md_file in sorted(self.policies_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            doc_name = md_file.name

            # Extract doc title and code if present
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            doc_title = title_match.group(1).strip() if title_match else doc_name

            code_match = re.search(r"Mã tài liệu:\s*([A-Z0-9\-]+)", content)
            doc_code = code_match.group(1) if code_match else None

            # Split by markdown headers
            sections = re.split(r"\n(?=##?\s+)", content)
            for section in sections:
                clean_sec = section.strip()
                if not clean_sec:
                    continue

                lines = clean_sec.splitlines()
                header_line = lines[0].strip() if lines else doc_name
                sec_title = re.sub(r"^#+\s*", "", header_line)

                # Skip header metadata block if it doesn't contain policy rules
                if (
                    sec_title == doc_title
                    and "Mã tài liệu:" in clean_sec
                    and len(lines) <= 7
                ):
                    continue

                chunk = PolicyChunk(
                    doc_name=doc_name,
                    doc_title=doc_title,
                    doc_code=doc_code,
                    section_title=sec_title,
                    content=clean_sec,
                )
                self.chunks.append(chunk)
                search_text = f"{doc_title} {sec_title} {clean_sec}"
                all_texts.append(search_text)

        total_docs = len(all_texts)
        if total_docs == 0:
            self._indexed = True
            return 0

        # 1. Try Dense Semantic Embeddings via Ollama (nomic-embed-text)
        if self.settings.EMBEDDING_PROVIDER == "ollama":
            try:
                self._embeddings_model = OllamaEmbeddings(
                    model=self.settings.OLLAMA_EMBEDDING_MODEL,
                    base_url=self.settings.OLLAMA_BASE_URL,
                )
                self._dense_embeddings = self._embeddings_model.embed_documents(
                    all_texts
                )
                self._using_dense = True
                logger.info(
                    "Indexed %d policy chunks with Ollama %s",
                    len(all_texts),
                    self.settings.OLLAMA_EMBEDDING_MODEL,
                )
            except Exception as exc:
                logger.warning(
                    "Could not generate Ollama embeddings (%s): %s. "
                    "Using TF-IDF fallback.",
                    self.settings.OLLAMA_EMBEDDING_MODEL,
                    exc,
                )
                self._using_dense = False

        # 2. Build TF-IDF vectors as robust fallback
        doc_freq: dict[str, int] = {}
        for text in all_texts:
            tokens = set(re.findall(r"\w+", text.lower()))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        self._vocabulary = sorted(doc_freq.keys())
        self._idf = {
            w: math.log(1.0 + (total_docs / float(freq)))
            for w, freq in doc_freq.items()
        }

        self._tfidf_embeddings = []
        for text in all_texts:
            tokens = re.findall(r"\w+", text.lower())
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            vec = [
                (1.0 + math.log(tf[w])) * self._idf.get(w, 1.0) if w in tf else 0.0
                for w in self._vocabulary
            ]
            self._tfidf_embeddings.append(vec)

        self._indexed = True

        # Cache in memory
        _INDEX_CACHE[cache_key] = {
            "chunks": self.chunks,
            "dense": self._dense_embeddings,
            "using_dense": self._using_dense,
            "model": self._embeddings_model,
            "vocab": self._vocabulary,
            "idf": self._idf,
            "tfidf": self._tfidf_embeddings,
        }

        return len(self.chunks)

    def retrieve(self, query: str, k: int = 3) -> list[PolicyChunk]:
        """Retrieve top-k relevant policy chunks with source citations."""
        if not self._indexed:
            self.index_documents()

        if not self.chunks:
            return []

        q_tokens = re.findall(r"\w+", query.lower())
        unique_q_words = {
            w for w in q_tokens if len(w) >= 3 and w not in VIETNAMESE_STOP_WORDS
        }

        # A. Try Dense Semantic Search using nomic-embed-text
        if self._using_dense and self._embeddings_model is not None:
            try:
                dense_q_vec = self._embeddings_model.embed_query(query)
                scored_dense: list[tuple[float, PolicyChunk]] = []

                for chunk, vec in zip(
                    self.chunks, self._dense_embeddings, strict=False
                ):
                    sim = cosine_similarity(dense_q_vec, vec)

                    sec_lower = chunk.section_title.lower()
                    doc_lower = f"{chunk.doc_name} {chunk.doc_title}".lower()
                    content_lower = chunk.content.lower()

                    sec_matches = [w for w in unique_q_words if w in sec_lower]
                    doc_matches = [w for w in unique_q_words if w in doc_lower]
                    content_matches = [w for w in unique_q_words if w in content_lower]

                    coverage_boost = (
                        (len(content_matches) / len(unique_q_words)) * 0.4
                        if unique_q_words
                        else 0.0
                    )
                    content_boost = 0.06 * len(content_matches)

                    sim += 0.08 * len(sec_matches)
                    sim += 0.05 * len(doc_matches)
                    sim += coverage_boost + content_boost

                    scored_dense.append((sim, chunk))

                scored_dense.sort(key=lambda x: x[0], reverse=True)
                results: list[PolicyChunk] = []
                for score, chunk in scored_dense[:k]:
                    res_chunk = chunk.model_copy()
                    res_chunk.score = round(score, 4)
                    results.append(res_chunk)
                return results
            except Exception as exc:
                logger.warning(
                    "Dense query embedding failed (%s), falling back to TF-IDF",
                    exc,
                )

        # B. Fallback: TF-IDF Retrieval
        if not self._vocabulary:
            return []

        q_tf: dict[str, int] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        query_vec = [
            (1.0 + math.log(q_tf[w])) * self._idf.get(w, 1.0) if w in q_tf else 0.0
            for w in self._vocabulary
        ]

        scored: list[tuple[float, PolicyChunk]] = []
        for chunk, vec in zip(self.chunks, self._tfidf_embeddings, strict=False):
            sim = cosine_similarity(query_vec, vec)

            sec_lower = chunk.section_title.lower()
            doc_lower = f"{chunk.doc_name} {chunk.doc_title}".lower()
            content_lower = chunk.content.lower()

            sec_matches = [w for w in unique_q_words if w in sec_lower]
            doc_matches = [w for w in unique_q_words if w in doc_lower]
            content_matches = [w for w in unique_q_words if w in content_lower]

            coverage_boost = (
                (len(content_matches) / len(unique_q_words)) * 0.3
                if unique_q_words
                else 0.0
            )

            sim += 0.15 * len(sec_matches)
            sim += 0.05 * len(doc_matches)
            sim += coverage_boost

            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        results_fallback: list[PolicyChunk] = []
        for score, chunk in scored[:k]:
            res_chunk = chunk.model_copy()
            res_chunk.score = round(score, 4)
            results_fallback.append(res_chunk)

        return results_fallback

    def format_citation(self, chunk: PolicyChunk) -> str:
        """Format a human-readable and standard citation string."""
        code_str = f" ({chunk.doc_code})" if chunk.doc_code else ""
        return f"[{chunk.doc_name}{code_str}: {chunk.section_title}]"
