"""Tests for OutfitRetrievalAndReranker on Fixed 1,000-Product Snapshot."""

import pytest

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)
from customer_shop_assistant.services.outfit_retrieval import (
    OutfitRetrievalAndReranker,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)


def _make_reranker() -> OutfitRetrievalAndReranker:
    return OutfitRetrievalAndReranker(
        kg_repo=InMemoryOutfitKGRepository(),
        snapshot_index=get_snapshot_catalog_index(get_settings()),
    )


def test_retrieval_respects_hard_constraints_and_attaches_rule_ids() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="đi làm",
        style="Smart Casual",
        disliked_colors=["Đen"],
        max_price=600_000.0,
        limit=3,
    )
    assert res.error is None
    assert len(res.outfits) >= 1
    for outfit in res.outfits:
        assert outfit.top.in_stock is True
        assert outfit.bottom.in_stock is True
        assert outfit.top.price <= 600_000.0
        assert outfit.bottom.price <= 600_000.0
        assert outfit.top.color != "Đen"
        assert outfit.bottom.color != "Đen"
        assert outfit.applied_rule_ids
        assert outfit.reasons


def test_retrieval_multi_gender_balance_when_gender_missing() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        top_category="Áo sơ mi",
        top_color="Trắng",
        limit=4,
    )
    assert res.error is None
    assert len(res.outfits) >= 2
    assert any("cân bằng" in n.lower() for n in res.notes)


@pytest.mark.parametrize(
    "oos_cat",
    ["Váy liền", "Đầm dạ hội", "Túi xách", "Mũ nón", "Thắt lưng"],
)
def test_retrieval_rejects_five_non_blueprint_categories(oos_cat: str) -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(top_category=oos_cat)
    assert res.error == "out_of_scope"
    assert res.outfits == []
