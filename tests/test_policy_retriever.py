import pytest

from customer_shop_assistant.services.policy_retriever import (
    PolicyRetriever,
    cosine_similarity,
)


def test_cosine_similarity() -> None:
    vec_a = [1.0, 0.0, 1.0]
    vec_b = [1.0, 0.0, 1.0]
    vec_c = [0.0, 1.0, 0.0]

    assert pytest.approx(cosine_similarity(vec_a, vec_b)) == 1.0
    assert pytest.approx(cosine_similarity(vec_a, vec_c)) == 0.0


def test_index_and_retrieve_shipping_policy() -> None:
    retriever = PolicyRetriever()
    count = retriever.index_documents()
    assert count > 0

    query = "Phí giao hàng là bao nhiêu và đơn bao nhiêu được freeship?"
    results = retriever.retrieve(query, k=2)
    assert len(results) > 0

    top_result = results[0]
    assert "shipping.md" in top_result.doc_name
    assert "30.000" in top_result.content
    assert "500.000" in top_result.content

    citation = retriever.format_citation(top_result)
    assert "shipping.md" in citation


def test_retrieve_returns_policy() -> None:
    retriever = PolicyRetriever()
    results = retriever.retrieve("Thời hạn đổi trả hàng trong mấy ngày?", k=2)
    assert len(results) > 0

    # Must contain returns.md
    doc_names = [r.doc_name for r in results]
    assert "returns.md" in doc_names
    returns_chunk = next(r for r in results if r.doc_name == "returns.md")
    assert "7 ngày" in returns_chunk.content


def test_retrieve_payment_policy() -> None:
    retriever = PolicyRetriever()
    results = retriever.retrieve("Shop có hỗ trợ thanh toán VietQR MBBank không?", k=2)
    assert len(results) > 0

    doc_names = [r.doc_name for r in results]
    assert "payment.md" in doc_names
    payment_chunk = next(r for r in results if r.doc_name == "payment.md")
    assert "VietQR" in payment_chunk.content
    assert "MBBank" in payment_chunk.content


def test_dense_embedding_nomic_embed_text() -> None:
    retriever = PolicyRetriever()
    retriever.index_documents()
    assert len(retriever.chunks) > 0
    if retriever._using_dense:
        assert len(retriever._dense_embeddings) == len(retriever.chunks)
        assert len(retriever._dense_embeddings[0]) == 768


def test_cache_key_hash_and_model() -> None:
    retriever = PolicyRetriever()
    hash1 = retriever._compute_policies_hash()
    assert len(hash1) == 32  # Valid MD5 hex digest
