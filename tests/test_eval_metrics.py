from customer_shop_assistant.eval.dataset import EvaluationCase, GroundTruth
from customer_shop_assistant.eval.metrics import evaluate_response
from customer_shop_assistant.schemas.chat import ClarificationMode


def test_metric_intent_and_tool_match() -> None:
    case = EvaluationCase(
        id="TEST_01",
        category="product_suitability",
        question="Tìm áo thun cotton mát mẻ",
        clarification_mode=ClarificationMode.MINIMAL_CLARIFICATION,
        ground_truth=GroundTruth(
            expected_intent="CATALOG_SEARCH",
            expected_tool="search_products",
            required_keywords=["áo thun", "cotton"],
            should_clarify=False,
        ),
    )

    res = evaluate_response(
        case=case,
        reply="Shop có mẫu Áo thun cotton Compact 100% cực mát giá 250.000đ.",
        intent="CATALOG_SEARCH",
        tools_used=["search_products"],
        clarification_asked=False,
        latency_ms=120.0,
        estimated_tokens=50,
    )

    assert res.passed
    assert res.intent_match
    assert res.tool_match
    assert res.clarification_compliance
    assert res.keyword_score == 1.0


def test_metric_clarification_compliance() -> None:
    case = EvaluationCase(
        id="TEST_02",
        category="clarification_strategy",
        question="Tư vấn đồ cho mình với",
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        ground_truth=GroundTruth(
            expected_intent="PRODUCT_ADVICE",
            should_clarify=True,
        ),
    )

    # Success case: clarification asked
    res_success = evaluate_response(
        case=case,
        reply=(
            "Bạn muốn tìm đồ đi làm hay đi chơi, "
            "và chiều cao cân nặng của bạn là bao nhiêu?"
        ),
        intent="PRODUCT_ADVICE",
        tools_used=[],
        clarification_asked=True,
        latency_ms=80.0,
        estimated_tokens=30,
    )
    assert res_success.clarification_compliance
    assert res_success.passed

    # Failure case: gave immediate answer instead of clarifying
    res_fail = evaluate_response(
        case=case,
        reply="Shop có áo thun này đẹp lắm mua đi bạn.",
        intent="PRODUCT_ADVICE",
        tools_used=["search_products"],
        clarification_asked=False,
        latency_ms=80.0,
        estimated_tokens=30,
    )
    assert not res_fail.clarification_compliance
    assert not res_fail.passed


def test_metric_policy_citation() -> None:
    case = EvaluationCase(
        id="TEST_03",
        category="policy_faq",
        question="Phí ship thế nào?",
        ground_truth=GroundTruth(
            expected_intent="POLICY_INQUIRY",
            expected_policy_doc="shipping.md",
        ),
    )

    # Valid citation
    res_valid = evaluate_response(
        case=case,
        reply="Theo quy định tại [shipping.md: Biểu phí], đơn trên 500k được freeship.",
        intent="POLICY_INQUIRY",
        tools_used=["search_policy"],
        clarification_asked=False,
        latency_ms=90.0,
        estimated_tokens=40,
    )
    assert res_valid.policy_citation_match
    assert res_valid.passed

    # Missing citation
    res_invalid = evaluate_response(
        case=case,
        reply="Shop freeship đơn trên 500k nhé.",
        intent="POLICY_INQUIRY",
        tools_used=["search_policy"],
        clarification_asked=False,
        latency_ms=90.0,
        estimated_tokens=30,
    )
    assert not res_invalid.policy_citation_match
    assert not res_invalid.passed


def test_metric_out_of_scope_rejection() -> None:
    case = EvaluationCase(
        id="TEST_04",
        category="out_of_scope",
        question="Hủy đơn ORD-1234 giúp mình",
        ground_truth=GroundTruth(
            expected_intent="OUT_OF_SCOPE",
            is_out_of_scope=True,
        ),
    )

    res = evaluate_response(
        case=case,
        reply=(
            "Hiện tại mình chưa hỗ trợ thao tác hủy đơn cá nhân. "
            "Bạn vui lòng liên hệ nhân viên shop để được hỗ trợ nhé."
        ),
        intent="OUT_OF_SCOPE",
        tools_used=[],
        clarification_asked=False,
        latency_ms=50.0,
        estimated_tokens=25,
    )
    assert res.out_of_scope_match
    assert res.passed


def test_metric_outfit_returned_and_violation_rate() -> None:
    from customer_shop_assistant.schemas.chat import (
        OutfitItem,
        OutfitSuggestion,
        UserProfile,
    )
    from customer_shop_assistant.schemas.shop import GenderTarget

    case = EvaluationCase(
        id="OUTFIT_TEST",
        category="product_suitability",
        sub_category="mix_and_match",
        question="Gợi ý set đồ đi làm cho nam",
        user_profile=UserProfile(gender=GenderTarget.MALE),
        ground_truth=GroundTruth(
            expected_intent="PRODUCT_ADVICE",
            expected_tool="suggest_outfit",
            required_keywords=["áo", "quần"],
        ),
    )

    valid_outfit = OutfitSuggestion(
        top=OutfitItem(
            product_id=1,
            name="Áo polo nam",
            category="Áo polo",
            color="Xanh navy",
            in_stock=True,
            price=350000.0,
        ),
        bottom=OutfitItem(
            product_id=2,
            name="Quần kaki nam",
            category="Quần kaki",
            color="Be sữa",
            in_stock=True,
            price=450000.0,
        ),
        score=0.9,
        reasons=["Hợp"],
    )
    bad_skirt_for_male = OutfitSuggestion(
        top=valid_outfit.top,
        bottom=OutfitItem(
            product_id=3,
            name="Chân váy xếp ly",
            category="Chân váy",
            color="Đen",
            in_stock=True,
            price=380000.0,
        ),
        score=0.8,
        reasons=["Sai giới tính"],
    )

    res_ok = evaluate_response(
        case=case,
        reply="Gợi ý áo polo và quần kaki cho bạn.",
        intent="PRODUCT_ADVICE",
        tools_used=["suggest_outfit"],
        clarification_asked=False,
        latency_ms=100.0,
        estimated_tokens=40,
        outfits=[valid_outfit, bad_skirt_for_male],
    )
    assert res_ok.outfit_returned is True
    assert res_ok.outfit_rule_violation_rate == 0.5
