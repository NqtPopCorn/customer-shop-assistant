"""Tests for Method A zero-double-counting scoring in InMemoryOutfitKGRepository."""

from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)


def test_branch_1_style_known_excludes_suits_category() -> None:
    repo = InMemoryOutfitKGRepository()
    bps = repo.get_outfit_blueprints(
        style_id="smart_casual", occasion_id="office", limit=5
    )
    assert bps
    top_bp = bps[0]
    assert "pair" in top_bp.score_breakdown
    assert "style_top" in top_bp.score_breakdown
    assert "style_bottom" in top_bp.score_breakdown
    assert "occasion_style" in top_bp.score_breakdown
    assert "occasion_top" not in top_bp.score_breakdown
    # Zero double-counting: Branch 1 never uses SC_* (SUITS_CATEGORY) rule IDs
    assert all(not rid.startswith("SC_") for rid in top_bp.applied_rule_ids)
    assert any(rid.startswith("RC_") for rid in top_bp.applied_rule_ids)
    assert any(rid.startswith("PW_") for rid in top_bp.applied_rule_ids)

    expected = round(
        0.45 * top_bp.score_breakdown["pair"]
        + 0.25 * top_bp.score_breakdown["style_top"]
        + 0.25 * top_bp.score_breakdown["style_bottom"]
        + 0.05 * top_bp.score_breakdown["occasion_style"],
        4,
    )
    assert top_bp.score == expected


def test_branch_2_occasion_only_excludes_recommends_category() -> None:
    repo = InMemoryOutfitKGRepository()
    bps = repo.get_outfit_blueprints(style_id=None, occasion_id="office", limit=5)
    assert bps
    top_bp = bps[0]
    assert "occasion_top" in top_bp.score_breakdown
    assert "occasion_bottom" in top_bp.score_breakdown
    assert "style_top" not in top_bp.score_breakdown
    assert all(not rid.startswith("RC_") for rid in top_bp.applied_rule_ids)
    assert any(rid.startswith("SC_") for rid in top_bp.applied_rule_ids)

    expected = round(
        0.50 * top_bp.score_breakdown["pair"]
        + 0.25 * top_bp.score_breakdown["occasion_top"]
        + 0.25 * top_bp.score_breakdown["occasion_bottom"],
        4,
    )
    assert top_bp.score == expected


def test_branch_3_and_bidirectional_anchor_normalization() -> None:
    repo = InMemoryOutfitKGRepository()
    bps = repo.get_outfit_blueprints(anchor_category_id="chan_vay", limit=5)
    assert bps
    assert all(bp.bottom_category_id == "chan_vay" for bp in bps)
    assert all(bp.top_category_id != "chan_vay" for bp in bps)
    assert all(len(bp.applied_rule_ids) == 1 for bp in bps)
    assert all(bp.applied_rule_ids[0].startswith("PW_") for bp in bps)

    # Non-blueprint category returns empty blueprints
    assert repo.get_outfit_blueprints(anchor_category_id="vay_lien") == []


def test_color_harmony_is_decoupled_and_optional() -> None:
    repo = InMemoryOutfitKGRepository()
    navy = repo.get_color_harmony("xanh_navy")
    assert navy.available is True
    assert "be_sua" in navy.recommended_color_ids
    assert "trang" in navy.recommended_color_ids
    assert any(r.startswith("CH_") for r in navy.applied_rule_ids)

    unknown = repo.get_color_harmony("mau_khong_ton_tai")
    assert unknown.available is False
    assert unknown.recommended_color_ids == []
    # Blueprint query remains unaffected even without valid color
    assert repo.get_outfit_blueprints(occasion_id="party")
