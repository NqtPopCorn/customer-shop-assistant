import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Module-level cache to avoid redundant embedding calls
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


class PolicyRetriever:
    """RAG retriever indexing and fetching store policy markdown documents.

    Uses dense semantic embeddings (nomic-embed-text via Ollama) and
    standard Okapi BM25, fused with Reciprocal Rank Fusion (RRF).
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
        self._indexed = False
        self._using_dense = False
        self._embeddings_model: OllamaEmbeddings | None = None

        # BM25 index structures
        self._doc_token_counts: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}

    def _compute_policies_hash(self) -> str:
        """Compute MD5 hash across file modification times and sizes in policies_dir."""
        if not self.policies_dir.exists():
            return "empty"
        file_stats: list[str] = []
        for md_file in sorted(self.policies_dir.glob("*.md")):
            stat = md_file.stat()
            file_stats.append(f"{md_file.name}:{stat.st_mtime}:{stat.st_size}")
        return hashlib.md5("".join(file_stats).encode()).hexdigest()

    def index_documents(self) -> int:
        """Parse markdown files in policies directory into chunks and index them."""
        model_name = (
            self.settings.OLLAMA_EMBEDDING_MODEL
            if self.settings.EMBEDDING_PROVIDER == "ollama"
            else "none"
        )
        cache_key = (
            f"{self.policies_dir}:{self._compute_policies_hash()}:"
            f"{self.settings.EMBEDDING_PROVIDER}:{model_name}"
        )

        if cache_key in _INDEX_CACHE:
            cached = _INDEX_CACHE[cache_key]
            self.chunks = cached["chunks"]
            self._dense_embeddings = cached["dense"]
            self._using_dense = cached["using_dense"]
            self._embeddings_model = cached["model"]
            self._doc_token_counts = cached["doc_token_counts"]
            self._doc_lengths = cached["doc_lengths"]
            self._avg_doc_len = cached["avg_doc_len"]
            self._idf = cached["idf"]
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
                search_text = f"{doc_title}\n{sec_title}\n{sec_title}\n{clean_sec}"
                all_texts.append(search_text)

        total_docs = len(all_texts)
        if total_docs == 0:
            self._indexed = True
            return 0

        # 1. Dense Semantic Embeddings via Ollama (nomic-embed-text)
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
                    "Could not generate Ollama embeddings (%s): %s. Using BM25.",
                    self.settings.OLLAMA_EMBEDDING_MODEL,
                    exc,
                )
                self._using_dense = False

        # 2. Standard Okapi BM25 Indexing
        self._doc_token_counts = []
        self._doc_lengths = []
        doc_freq: dict[str, int] = {}

        for text in all_texts:
            tokens = re.findall(r"\w+", text.lower())
            self._doc_lengths.append(len(tokens))
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_token_counts.append(tf)

            for unique_token in tf.keys():
                doc_freq[unique_token] = doc_freq.get(unique_token, 0) + 1

        self._avg_doc_len = sum(self._doc_lengths) / float(total_docs)
        self._idf = {
            w: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for w, freq in doc_freq.items()
        }

        self._indexed = True

        _INDEX_CACHE[cache_key] = {
            "chunks": self.chunks,
            "dense": self._dense_embeddings,
            "using_dense": self._using_dense,
            "model": self._embeddings_model,
            "doc_token_counts": self._doc_token_counts,
            "doc_lengths": self._doc_lengths,
            "avg_doc_len": self._avg_doc_len,
            "idf": self._idf,
        }

        return len(self.chunks)

    def retrieve(self, query: str, k: int = 3) -> list[PolicyChunk]:
        """Retrieve top-k relevant policy chunks using dense + BM25 RRF fusion."""
        if not self._indexed:
            self.index_documents()

        if not self.chunks:
            return []

        # 1. BM25 Scores
        q_tokens = re.findall(r"\w+", query.lower())
        k1 = 1.5
        b = 0.75

        bm25_scores: list[float] = []
        unique_q_tokens = set(q_tokens)
        for _i, (tf_map, doc_len) in enumerate(
            zip(self._doc_token_counts, self._doc_lengths, strict=False)
        ):
            score = 0.0
            for t in unique_q_tokens:
                if t in tf_map:
                    freq = tf_map[t]
                    idf_val = self._idf.get(t, 0.0)
                    denom = freq + k1 * (1.0 - b + b * (doc_len / self._avg_doc_len))
                    score += idf_val * (freq * (k1 + 1.0)) / denom
            bm25_scores.append(score)

        # Ranked by BM25
        bm25_ranked = sorted(
            range(len(self.chunks)), key=lambda idx: bm25_scores[idx], reverse=True
        )

        # 2. Dense Semantic Search (if available)
        dense_ranked: list[int] = []
        dense_sims: list[float] = [0.0] * len(self.chunks)

        if self._using_dense and self._embeddings_model is not None:
            try:
                dense_q_vec = self._embeddings_model.embed_query(query)
                for idx, vec in enumerate(self._dense_embeddings):
                    dense_sims[idx] = cosine_similarity(dense_q_vec, vec)
                dense_ranked = sorted(
                    range(len(self.chunks)),
                    key=lambda idx: dense_sims[idx],
                    reverse=True,
                )
            except Exception as exc:
                logger.warning("Dense query embedding failed: %s", exc)
                dense_ranked = []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: list[float] = [0.0] * len(self.chunks)
        c = 60.0

        for rank, chunk_idx in enumerate(bm25_ranked):
            if bm25_scores[chunk_idx] > 0.0:
                rrf_scores[chunk_idx] += 1.0 / (c + rank + 1)

        if dense_ranked:
            for rank, chunk_idx in enumerate(dense_ranked):
                if dense_sims[chunk_idx] > 0.0:
                    rrf_scores[chunk_idx] += 1.0 / (c + rank + 1)

        final_ranked = sorted(
            range(len(self.chunks)), key=lambda idx: rrf_scores[idx], reverse=True
        )

        results: list[PolicyChunk] = []
        for idx in final_ranked[:k]:
            if rrf_scores[idx] <= 0.0 and bm25_scores[idx] <= 0.0:
                continue
            chunk = self.chunks[idx].model_copy()
            chunk.score = round(rrf_scores[idx], 4)
            results.append(chunk)

        return results

    def format_citation(self, chunk: PolicyChunk) -> str:
        """Format a human-readable and standard citation string."""
        code_str = f" ({chunk.doc_code})" if chunk.doc_code else ""
        return f"[{chunk.doc_name}{code_str}: {chunk.section_title}]"
