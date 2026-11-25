from customer_shop_assistant.eval.dataset import EvaluationCase, GroundTruth
from customer_shop_assistant.eval.metrics import (
    compute_slot_accuracy,
    evaluate_response,
    evaluate_turn_structure_and_grounding,
    extract_token_usage,
)
from customer_shop_assistant.schemas.chat import (
    ChatRecommendedProduct,
    ClarificationMode,
    LangChainUsageMetadata,
    OutfitItem,
    OutfitSuggestion,
    TurnDebugMeta,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget


def test_metric_intent_tool_and_soft_keywords() -> None:
    case = EvaluationCase(
        id="TEST_01",
        category="product_suitability",
        question="Tìm áo thun cotton mát mẻ",
        clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        ground_truth=GroundTruth(
            expected_intent="PRODUCT_SEARCH",
            expected_tool="search_products",
            required_keywords=["từ_khóa_không_có", "cotton"],
            should_clarify=False,
        ),
    )

    res = evaluate_response(
        case=case,
        reply="Shop có mẫu Áo phông thoáng mát 100% tự nhiên giá 250.000đ.",
        intent="PRODUCT_SEARCH",
        tools_used=["search_products"],
        clarification_asked=False,
        latency_ms=120.0,
        allow_token_estimate=True,
    )

    assert res.passed
    assert res.intent_match
    assert res.tool_match
    assert res.clarification_compliance
    assert res.keyword_score == 0.0  # soft diagnostic only


def test_forbidden_keyword_fails_passed() -> None:
    case = EvaluationCase(
        id="TEST_FORBID",
        category="product_suitability",
        question="Phối đồ nam đi làm",
        user_profile=UserProfile(gender=GenderTarget.MALE),
        ground_truth=GroundTruth(
            expected_intent="OUTFIT_ADVICE",
            expected_tool="suggest_outfit",
            forbidden_keywords=["chân váy"],
        ),
    )

    res = evaluate_response(
        case=case,
        reply="Bạn có thể phối áo sơ mi với chân váy công sở nhé.",
        intent="OUTFIT_ADVICE",
        tools_used=["suggest_outfit"],
        clarification_asked=False,
        latency_ms=90.0,
    )
    assert not res.forbidden_keyword_ok
    assert not res.passed


def test_metric_clarification_compliance() -> None:
    case = EvaluationCase(
        id="TEST_02",
        category="clarification_strategy",
        question="Tư vấn đồ cho mình với",
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        ground_truth=GroundTruth(
            expected_intent="PRODUCT_SEARCH",
            should_clarify=True,
        ),
    )

    res_success = evaluate_response(
        case=case,
        reply="Bạn muốn tìm đồ đi làm hay đi chơi ạ?",
        intent="PRODUCT_SEARCH",
        tools_used=[],
        clarification_asked=True,
        latency_ms=80.0,
    )
    assert res_success.clarification_compliance
    assert res_success.passed

    res_fail = evaluate_response(
        case=case,
        reply="Shop có áo thun này đẹp lắm mua đi bạn.",
        intent="PRODUCT_SEARCH",
        tools_used=["search_products"],
        clarification_asked=False,
        latency_ms=80.0,
    )
    assert not res_fail.clarification_compliance
    assert not res_fail.passed


def test_metric_policy_citation_and_oos() -> None:
    policy_case = EvaluationCase(
        id="TEST_03",
        category="policy_faq",
        question="Phí ship thế nào?",
        ground_truth=GroundTruth(
            expected_intent="NON_SHOPPING",
            expected_policy_doc="shipping.md",
        ),
    )
    res_valid = evaluate_response(
        case=policy_case,
        reply="Theo quy định tại [shipping.md: Biểu phí], đơn trên 500k freeship.",
        intent="NON_SHOPPING",
        tools_used=["search_policy"],
        clarification_asked=False,
        latency_ms=90.0,
    )
    assert res_valid.policy_citation_match
    assert res_valid.passed

    res_invalid = evaluate_response(
        case=policy_case,
        reply="Shop freeship đơn trên 500k nhé.",
        intent="NON_SHOPPING",
        tools_used=["search_policy"],
        clarification_asked=False,
        latency_ms=90.0,
    )
    assert not res_invalid.policy_citation_match
    assert not res_invalid.passed

    oos_case = EvaluationCase(
        id="TEST_04",
        category="out_of_scope",
        question="Hủy đơn ORD-1234 giúp mình",
        ground_truth=GroundTruth(
            expected_intent="NON_SHOPPING",
            is_out_of_scope=True,
        ),
    )
    res_oos = evaluate_response(
        case=oos_case,
        reply=(
            "Hiện tại mình chưa hỗ trợ thao tác hủy đơn cá nhân. "
            "Bạn vui lòng liên hệ nhân viên shop để được hỗ trợ nhé."
        ),
        intent="NON_SHOPPING",
        tools_used=[],
        clarification_asked=False,
        latency_ms=50.0,
    )
    assert res_oos.out_of_scope_match
    assert res_oos.passed


def test_slot_accuracy_and_token_extraction() -> None:
    empty_res = compute_slot_accuracy([], [])
    assert empty_res.precision is None
    assert empty_res.recall is None

    exact = compute_slot_accuracy(
        ["target_gender", "occasion"], ["occasion", "target_gender"]
    )
    assert exact.precision == 1.0
    assert exact.recall == 1.0
    assert exact.exact_match is True

    partial = compute_slot_accuracy(
        ["target_gender", "occasion"], ["target_gender", "budget"]
    )
    assert partial.precision == 0.5
    assert partial.recall == 0.5
    assert partial.exact_match is False

    meta = TurnDebugMeta(
        mode="CONDITIONAL_CLARIFICATION",
        phase="OPEN",
        gated_turns=0,
        tools_available=[],
        latency_ms=100,
        usage_metadata=LangChainUsageMetadata(
            input_tokens=320, output_tokens=80, total_tokens=400
        ),
    )
    tok = extract_token_usage(meta)
    assert tok.input_tokens == 320
    assert tok.output_tokens == 80
    assert tok.total_tokens == 400
    assert not tok.is_estimated


def test_structural_and_grounding_checks() -> None:
    # Mode A leaked clarification vs allowed polite closing
    leak = evaluate_turn_structure_and_grounding(
        mode=ClarificationMode.IMMEDIATE_ANSWER,
        reply="Bạn muốn tìm sản phẩm cho nam hay nữ vậy?",
        clarification_asked=False,
    )
    assert not leak.mode_compliant

    closing_ok = evaluate_turn_structure_and_grounding(
        mode=ClarificationMode.IMMEDIATE_ANSWER,
        reply=(
            "Shop gợi ý mẫu [10] Áo sơ mi trắng giá 350.000đ. "
            "Bạn cần mình hỗ trợ thêm gì không?"
        ),
        clarification_asked=False,
        products=[
            ChatRecommendedProduct(
                id=10, name="Áo sơ mi trắng", price=350000.0, category="Áo sơ mi"
            )
        ],
    )
    assert closing_ok.mode_compliant
    assert closing_ok.grounding_ok

    # Phantom form reference
    phantom = evaluate_turn_structure_and_grounding(
        mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        reply="Bạn vui lòng tích chọn nhanh vài tiêu chí ở bảng dưới đây nhé:",
        clarification_asked=True,
        clarification_form=None,
    )
    assert not phantom.form_integrity_ok

    # Hallucinated ID and mismatched price
    bad_grounding = evaluate_turn_structure_and_grounding(
        mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        reply="Shop gợi ý [999] Áo thun giá 999.000đ.",
        products=[
            ChatRecommendedProduct(
                id=10, name="Áo thun", price=250000.0, category="Áo thun"
            )
        ],
    )
    assert not bad_grounding.id_grounding_ok
    assert not bad_grounding.price_grounding_ok
    assert not bad_grounding.grounding_ok


def test_metric_outfit_returned_and_violation_rate() -> None:
    case = EvaluationCase(
        id="OUTFIT_TEST",
        category="product_suitability",
        sub_category="mix_and_match",
        question="Gợi ý set đồ đi làm cho nam",
        user_profile=UserProfile(gender=GenderTarget.MALE),
        ground_truth=GroundTruth(
            expected_intent="OUTFIT_ADVICE",
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
        intent="OUTFIT_ADVICE",
        tools_used=["suggest_outfit"],
        clarification_asked=False,
        latency_ms=100.0,
        outfits=[valid_outfit, bad_skirt_for_male],
    )
    assert res_ok.outfit_returned is True
    assert res_ok.outfit_rule_violation_rate == 0.5
    assert not res_ok.grounding_ok
    assert not res_ok.passed
