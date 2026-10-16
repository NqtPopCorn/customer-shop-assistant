from customer_shop_assistant.schemas.chat import (
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.clarification_engine import (
    ClarificationEngine,
)


def test_extract_measurements() -> None:
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
