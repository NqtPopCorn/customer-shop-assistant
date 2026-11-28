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


def test_retrieval_office_male_option_a_diversity() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="công sở",
        limit=3,
    )
    assert res.error is None
    assert len(res.outfits) == 3

    blueprints = {(o.top.category, o.bottom.category) for o in res.outfits}
    assert len(blueprints) == 3

    top_ids = [o.top.product_id for o in res.outfits]
    bot_ids = [o.bottom.product_id for o in res.outfits]
    assert len(set(top_ids) | set(bot_ids)) == 6

    color_pairs = {(o.top.color, o.bottom.color) for o in res.outfits}
    assert len(color_pairs) >= 2

    for outfit in res.outfits:
        assert outfit.blueprint_score is not None
        assert outfit.color_score_breakdown is not None
        expected = round(
            0.70 * outfit.blueprint_score
            + 0.30 * outfit.color_score_breakdown.color_score,
            4,
        )
        assert outfit.score == pytest.approx(expected, abs=1e-4)


def test_retrieval_locked_categories_still_diversify_skus_and_colors() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="công sở",
        top_category="Áo sơ mi",
        bottom_category="Quần âu",
        limit=3,
    )
    assert res.error is None
    assert len(res.outfits) == 3
    top_ids = [o.top.product_id for o in res.outfits]
    bot_ids = [o.bottom.product_id for o in res.outfits]
    assert len(set(top_ids) | set(bot_ids)) == 6
    color_pairs = {(o.top.color, o.bottom.color) for o in res.outfits}
    assert len(color_pairs) >= 2


def test_retrieval_anchor_product_keeps_anchor_and_varies_bottoms() -> None:
    reranker = _make_reranker()
    res = reranker.retrieve_outfits(
        product_id=53,
        target_gender="MALE",
        occasion="công sở",
        limit=3,
    )
    assert res.error is None
    assert len(res.outfits) == 3
    assert all(o.top.product_id == 53 for o in res.outfits)
    bot_ids = [o.bottom.product_id for o in res.outfits]
    assert len(set(bot_ids)) == 3
    bot_cats = {o.bottom.category for o in res.outfits}
    assert len(bot_cats) >= 2


def test_retrieval_cross_turn_exclusion_and_graceful_degradation() -> None:
    reranker = _make_reranker()
    turn1 = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="công sở",
        limit=3,
    )
    shown_pairs = [(o.top.product_id, o.bottom.product_id) for o in turn1.outfits]
    shown_pids = [
        pid for o in turn1.outfits for pid in (o.top.product_id, o.bottom.product_id)
    ]

    turn2 = reranker.retrieve_outfits(
        target_gender="MALE",
        occasion="công sở",
        limit=3,
        exclude_pairs=shown_pairs,
        exclude_product_ids=shown_pids,
    )
    assert len(turn2.outfits) == 3
    turn2_pairs = {(o.top.product_id, o.bottom.product_id) for o in turn2.outfits}
    turn2_pids = {
        pid for o in turn2.outfits for pid in (o.top.product_id, o.bottom.product_id)
    }
    assert turn2_pairs.isdisjoint(set(shown_pairs))
    assert turn2_pids.isdisjoint(set(shown_pids))

    # Graceful degradation when small live catalog is already in exclusion set
    tiny = [
        ProductSchema(
            id=8001,
            name="Áo sơ mi trắng",
            slug="ao-so-mi-trang",
            category=CategorySchema(id=8001, name="Áo sơ mi"),
            genderTarget=GenderTarget.MALE,
            variants=[
                ProductVariantSchema(
                    id=8101,
                    sku="SKU-8001",
                    size="M",
                    color="Trắng",
                    listPrice=300000.0,
                    sellingPrice=300000.0,
                    finalPrice=300000.0,
                    stock=5,
                )
            ],
        ),
        ProductSchema(
            id=8002,
            name="Quần âu đen",
            slug="quan-au-den",
            category=CategorySchema(id=8002, name="Quần âu"),
            genderTarget=GenderTarget.MALE,
            variants=[
                ProductVariantSchema(
                    id=8102,
                    sku="SKU-8002",
                    size="30",
                    color="Đen",
                    listPrice=400000.0,
                    sellingPrice=400000.0,
                    finalPrice=400000.0,
                    stock=5,
                )
            ],
        ),
    ]
    degraded = reranker.retrieve_outfits(
        live_products=tiny,
        target_gender="MALE",
        occasion="công sở",
        exclude_pairs=[(8001, 8002)],
        exclude_product_ids=[8001, 8002],
        limit=1,
    )
    assert len(degraded.outfits) == 1
    assert degraded.outfits[0].top.product_id == 8001


def test_retrieval_balanced_determinism_and_session_rotation() -> None:
    reranker = _make_reranker()
    r_fixed_1 = reranker.retrieve_outfits(
        target_gender="MALE", occasion="công sở", limit=3, rotation_seed=None
    )
    r_fixed_2 = reranker.retrieve_outfits(
        target_gender="MALE", occasion="công sở", limit=3, rotation_seed=None
    )
    assert [(o.top.product_id, o.bottom.product_id) for o in r_fixed_1.outfits] == [
        (o.top.product_id, o.bottom.product_id) for o in r_fixed_2.outfits
    ]

    r_seed_a1 = reranker.retrieve_outfits(
        target_gender="MALE", occasion="công sở", limit=3, rotation_seed="sess-alpha"
    )
    r_seed_a2 = reranker.retrieve_outfits(
        target_gender="MALE", occasion="công sở", limit=3, rotation_seed="sess-alpha"
    )
    pairs_a = [(o.top.product_id, o.bottom.product_id) for o in r_seed_a1.outfits]
    assert pairs_a == [
        (o.top.product_id, o.bottom.product_id) for o in r_seed_a2.outfits
    ]

    rotated_sets = {
        tuple(
            (o.top.product_id, o.bottom.product_id)
            for o in reranker.retrieve_outfits(
                target_gender="MALE",
                occasion="công sở",
                limit=3,
                rotation_seed=seed,
            ).outfits
        )
        for seed in ("sess-alpha", "sess-beta", "sess-gamma", "sess-delta")
    }
    assert len(rotated_sets) >= 2
