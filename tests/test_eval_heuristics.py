from customer_shop_assistant.eval.heuristics import (
    ClarificationEngine,
    classify_intent,
    extract_measurements_from_text,
)
from customer_shop_assistant.schemas.chat import (
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget


def test_classify_intents() -> None:
    assert classify_intent("Cao 1m72 nặng 65kg mặc size gì?") == "SIZE_ADVICE"
    assert classify_intent("Chính sách đổi trả hàng như thế nào?") == "POLICY_INQUIRY"
    assert classify_intent("Mã voucher SUMMER50 giảm bao nhiêu?") == "CATALOG_SEARCH"
    assert classify_intent("Tư vấn giúp mình đồ đi tiệc cho nữ") == "PRODUCT_ADVICE"
    assert classify_intent("Kiểm tra tình trạng đơn hàng #1234") == "OUT_OF_SCOPE"
    assert classify_intent("Xin chào shop") == "GENERAL_CHAT"


def test_extract_measurements_from_text() -> None:
    extracted = extract_measurements_from_text(
        "Mình là nam cao 1m75 nặng 68kg cần tìm áo polo form rộng"
    )
    assert extracted["height_cm"] == 175.0
    assert extracted["weight_kg"] == 68.0
    assert extracted["gender"] == GenderTarget.MALE
    assert extracted["fit_preference"] == "LOOSE"


def test_clarification_engine_extract_entities() -> None:
    engine = ClarificationEngine()
    extracted = engine.extract_entities(
        "Mình là nam cao 1m75 nặng 68kg cần tìm áo polo form rộng"
    )
    assert extracted.height_cm == 175.0
    assert extracted.weight_kg == 68.0
    assert extracted.gender == GenderTarget.MALE
    assert extracted.fit_preference == "LOOSE"
    assert extracted.category == "tops"


def test_immediate_answer_never_clarifies() -> None:
    engine = ClarificationEngine()
    decision = engine.evaluate(
        query="Tư vấn size áo",
        intent="SIZE_ADVICE",
        mode=ClarificationMode.IMMEDIATE_ANSWER,
        profile=None,
    )
    assert not decision.needs_clarification
    assert decision.clarification_question is None


def test_minimal_clarification_blocks_when_measurements_missing() -> None:
    engine = ClarificationEngine()
    decision = engine.evaluate(
        query="Tư vấn size áo thun giúp mình",
        intent="SIZE_ADVICE",
        mode=ClarificationMode.MINIMAL_CLARIFICATION,
        profile=None,
    )
    assert decision.needs_clarification
    assert decision.clarification_question is not None
    assert "chiều cao" in decision.clarification_question


def test_minimal_clarification_passes_when_measurements_provided() -> None:
    engine = ClarificationEngine()
    decision = engine.evaluate(
        query="Mình cao 1m70 nặng 62kg tư vấn size áo khoác",
        intent="SIZE_ADVICE",
        mode=ClarificationMode.MINIMAL_CLARIFICATION,
        profile=None,
    )
    assert not decision.needs_clarification


def test_always_clarify_asks_fit_preference() -> None:
    engine = ClarificationEngine()
    decision = engine.evaluate(
        query="Mình cao 170cm nặng 60kg",
        intent="SIZE_ADVICE",
        mode=ClarificationMode.ALWAYS_CLARIFY,
        profile=UserProfile(height_cm=170.0, weight_kg=60.0),
    )
    assert decision.needs_clarification
    q = decision.clarification_question or ""
    assert "ôm" in q or "form" in q
