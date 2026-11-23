"""Unit tests for Outfit Knowledge Graph rules and pure scoring functions."""

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.outfit_rules import (
    ProductAttrs,
    classify_pattern,
    combine_scores,
    compute_formality,
    load_outfit_rules,
    normalize_occasion,
    passes_hard_filters,
    score_colour,
    score_formality,
    score_pattern,
)


def _rules():
    return load_outfit_rules(get_settings().DATA_DIR / "outfit_kg")


def test_colour_harmony_classic_vs_loud_clash() -> None:
    rules = _rules()
    navy_beige, reasons_good = score_colour("Xanh navy", "Be sữa", rules)
    red_green, reasons_bad = score_colour("Đỏ tươi", "Xanh lá", rules)

    assert navy_beige > red_green
    assert navy_beige >= 0.9
    assert any("kinh điển" in r for r in reasons_good)
    assert red_green < 0.3
    assert any("rực" in r for r in reasons_bad)


def test_pattern_scoring_matrix() -> None:
    assert score_pattern("PATTERN", "Stripe", "PATTERN", "Check")[0] == 0.2
    assert score_pattern("SOLID", "Solid", "PATTERN", "Stripe")[0] == 0.9
    assert score_pattern("SOLID", "Denim", "SOLID", "Denim")[0] == 0.6
    assert score_pattern("SOLID", "Solid", "SOLID", "Solid")[0] == 1.0


def test_classify_pattern_and_compute_formality_clamping() -> None:
    rules = _rules()
    assert classify_pattern("Solid", rules) == "SOLID"
    assert classify_pattern("Denim", rules) == "SOLID"
    assert classify_pattern("Melange", rules) == "SOLID"
    assert classify_pattern("Stripe", rules) == "PATTERN"
    assert classify_pattern(None, rules) == "SOLID"

    # Clamped at [1, 5]
    assert compute_formality("Quần âu", "Đi họp công sở lịch sự", rules) == 5
    assert compute_formality("Áo thun", "Mặc đi biển năng động", rules) == 1
    assert compute_formality("Áo polo", "Đi làm văn phòng công sở", rules) == 4
    assert compute_formality("Áo polo", "Phong cách retro dạo phố", rules) == 2
    # Both plus and minus keywords present -> neutral delta
    assert compute_formality("Áo polo", "Công sở nhưng vẫn thể thao", rules) == 3


def test_hard_filters_gender_role_formality_disliked() -> None:
    top = ProductAttrs(
        slug="top-1",
        role="TOP",
        category="Áo polo",
        pattern_raw="Solid",
        pattern_class="SOLID",
        formality=3,
    )
    skirt = ProductAttrs(
        slug="bot-skirt",
        role="BOTTOM",
        category="Chân váy",
        pattern_raw="Solid",
        pattern_class="SOLID",
        formality=3,
    )
    trousers = ProductAttrs(
        slug="bot-trousers",
        role="BOTTOM",
        category="Quần âu",
        pattern_raw="Solid",
        pattern_class="SOLID",
        formality=5,
    )
    tshirt = ProductAttrs(
        slug="top-tee",
        role="TOP",
        category="Áo thun",
        pattern_raw="Solid",
        pattern_class="SOLID",
        formality=1,
    )

    # Skirt rejected for MALE even if tagged UNISEX
    assert not passes_hard_filters(
        top,
        skirt,
        GenderTarget.MALE,
        GenderTarget.UNISEX,
        GenderTarget.MALE,
        None,
        set(),
        "Trắng",
        "Đen",
    )
    # Valid for FEMALE
    assert passes_hard_filters(
        top,
        skirt,
        GenderTarget.UNISEX,
        GenderTarget.FEMALE,
        GenderTarget.FEMALE,
        None,
        set(),
        "Trắng",
        "Đen",
    )
    # TOP + TOP rejected
    assert not passes_hard_filters(
        top,
        tshirt,
        GenderTarget.MALE,
        GenderTarget.MALE,
        GenderTarget.MALE,
        None,
        set(),
        "Trắng",
        "Đen",
    )
    # Formality diff 4 > 2 rejected (tshirt=1 vs trousers=5)
    assert not passes_hard_filters(
        tshirt,
        trousers,
        GenderTarget.MALE,
        GenderTarget.MALE,
        GenderTarget.MALE,
        None,
        set(),
        "Trắng",
        "Đen",
    )
    # Disliked colour rejected (case-insensitive)
    assert not passes_hard_filters(
        top,
        trousers,
        GenderTarget.MALE,
        GenderTarget.MALE,
        GenderTarget.MALE,
        None,
        {"đen"},
        "Trắng",
        "Đen",
    )
    # Occasion range (1, 2) excludes formality 5 (outside [0, 3])
    assert not passes_hard_filters(
        top,
        trousers,
        GenderTarget.MALE,
        GenderTarget.MALE,
        GenderTarget.MALE,
        (1, 2),
        set(),
        "Trắng",
        "Xám tiêu",
    )


def test_normalize_occasion_and_combine() -> None:
    rules = _rules()
    assert normalize_occasion("đi làm văn phòng", rules) == (3, 5)
    assert normalize_occasion("đi biển cuối tuần", rules) == (1, 2)
    assert normalize_occasion("không rõ dịp gì", rules) is None

    f_score, _ = score_formality(3, 3, (3, 5))
    assert f_score == 1.0
    combined = combine_scores(1.0, 0.9, 1.0, 1.0, True, rules.weights)
    assert combined == 0.97
