"""Tests for OutfitRetrievalAndReranker on Fixed 1,000-Product Snapshot."""

import pytest

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.shop import (
    CategorySchema,
    GenderTarget,
    ProductSchema,
    ProductVariantSchema,
)
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


def test_retrieval_attaches_color_score_breakdown_and_70_30_blend() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="đi làm",
        style="minimalist",
        limit=3,
    )
    assert res.error is None
    assert res.outfits
    for outfit in res.outfits:
        assert outfit.blueprint_score is not None
        assert outfit.color_score_breakdown is not None
        bd = outfit.color_score_breakdown
        assert bd.unmapped_colors == []
        assert 0.0 <= bd.color_score <= 1.0
        expected_final = round(0.70 * outfit.blueprint_score + 0.30 * bd.color_score, 4)
        assert outfit.score == pytest.approx(expected_final, abs=1e-4)


def test_retrieval_degrades_gracefully_on_runtime_unmapped_color() -> None:
    products = [
        ProductSchema(
            id=9001,
            name="Áo sơ mi cầu vồng",
            slug="ao-so-mi-cau-vong",
            description="Test product with unknown runtime color",
            category=CategorySchema(id=9001, name="Áo sơ mi"),
            genderTarget=GenderTarget.MALE,
            variants=[
                ProductVariantSchema(
                    id=9101,
                    sku="SKU-TOP-1",
                    size="M",
                    color="Màu cầu vồng XYZ",
                    listPrice=350000.0,
                    sellingPrice=350000.0,
                    finalPrice=350000.0,
                    stock=5,
                )
            ],
        ),
        ProductSchema(
            id=9002,
            name="Quần âu đen",
            slug="quan-au-den",
            description="Standard black trousers",
            category=CategorySchema(id=9002, name="Quần âu"),
            genderTarget=GenderTarget.MALE,
            variants=[
                ProductVariantSchema(
                    id=9102,
                    sku="SKU-BOT-1",
                    size="30",
                    color="Đen",
                    listPrice=450000.0,
                    sellingPrice=450000.0,
                    finalPrice=450000.0,
                    stock=5,
                )
            ],
        ),
    ]
    reranker = OutfitRetrievalAndReranker(kg_repo=InMemoryOutfitKGRepository())
    res = reranker.retrieve_outfits(
        live_products=products,
        target_gender="MALE",
        occasion="đi làm",
        style="minimalist",
        limit=2,
    )
    assert res.error is None
    assert len(res.outfits) >= 1
    bd = res.outfits[0].color_score_breakdown
    assert bd is not None
    assert "Màu cầu vồng XYZ" in bd.unmapped_colors
    assert bd.pair_harmony == pytest.approx(0.50)
