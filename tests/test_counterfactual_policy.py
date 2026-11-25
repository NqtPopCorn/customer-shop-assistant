"""Tests for CounterfactualClarificationPolicy and Agent + Telemetry integration."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.chat import ChatRequest, ClarificationMode
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.clarification_policy import (
    evaluate_clarification_policy,
)
from customer_shop_assistant.services.counterfactual_policy import (
    CandidateValueBuilder,
    CounterfactualClarificationPolicy,
)
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)
from customer_shop_assistant.services.session_manager import SessionManager
from customer_shop_assistant.services.slot_tracker import (
    TurnIntent,
    extract_conversation_slots,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _make_policy() -> CounterfactualClarificationPolicy:
    return CounterfactualClarificationPolicy(
        kg_repo=InMemoryOutfitKGRepository(),
        snapshot_index=get_snapshot_catalog_index(get_settings()),
    )


def test_candidate_value_builder() -> None:
    builder = CandidateValueBuilder()
    repo = InMemoryOutfitKGRepository()
    assert builder.build_candidates(
        "target_gender", anchor_category_id=None, occasion_id=None, kg_repo=repo
    ) == ["MALE", "FEMALE"]
    occs = builder.build_candidates(
        "occasion", anchor_category_id="ao_so_mi", occasion_id=None, kg_repo=repo
    )
    assert len(occs) >= 2
    styles = builder.build_candidates(
        "style", anchor_category_id=None, occasion_id="office", kg_repo=repo
    )
    assert "smart_casual" in styles


def test_scenario_1_party_outfit_asks_gender_via_counterfactual() -> None:
    policy = _make_policy()
    slots, intent = extract_conversation_slots(["Gợi ý outfit đi đám cưới."])
    assert intent == TurnIntent.OUTFIT_ADVICE

    dec = policy.evaluate(intent, slots, clarification_turns=0)
    assert dec is not None
    assert dec.action == "ASK"
    assert dec.target_slots == ["target_gender"]
    assert dec.decision_source == "mode_c_counterfactual"
    assert "target_gender" in dec.impact_signals
    assert dec.impact_signals["target_gender"]["mean_top_k_jaccard"] < 0.45
    assert dec.evidence_rule_ids


def test_scenario_2_anchor_white_shirt_proceeds_with_alternatives() -> None:
    policy = _make_policy()
    slots, intent = extract_conversation_slots(["Phối áo sơ mi trắng với quần gì?"])
    assert intent == TurnIntent.OUTFIT_ADVICE

    dec = policy.evaluate(intent, slots, clarification_turns=0)
    assert dec is not None
    assert dec.action == "PROCEED"
    assert dec.decision_source == "mode_c_counterfactual"
    assert "COVERED_WITH_ALTERNATIVES" in dec.reason_codes
    assert dec.use_fallback is True
    assert dec.fallback_slots == ["target_gender"]
    assert dec.impact_signals["target_gender"]["can_cover_with_alternatives"] is True


def test_unfulfillable_hard_constraint_triggers_ask() -> None:
    policy = _make_policy()
    slots, intent = extract_conversation_slots(
        ["Gợi ý outfit đi đám cưới cho nam dưới 10k"]
    )
    assert intent == TurnIntent.OUTFIT_ADVICE
    dec = policy.evaluate(intent, slots, clarification_turns=0)
    assert dec is not None
    assert dec.action == "ASK"
    assert "UNFULFILLABLE_HARD_CONSTRAINT" in dec.reason_codes
    assert dec.target_slots == ["budget"]


@pytest.mark.anyio
async def test_agent_integration_logs_counterfactual_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.services import telemetry_logger

    logged: list[dict[str, Any]] = []

    def _capture_log(log_path: Path, line: str) -> None:
        del log_path
        logged.append(json.loads(line))

    monkeypatch.setattr(telemetry_logger, "_write_log_line", _capture_log)

    cfg = Settings(_env_file=None, COUNTERFACTUAL_POLICY_ENABLED=True)
    agent = ShopAssistantAgent(
        chat_model=_FakeModel(
            messages=iter([AIMessage(content="Chọn giúp mình nhé.")])
        ),
        settings=cfg,
        session_manager=SessionManager(),
    )
    agent.structured_model = AsyncMock()

    res = await agent.chat(
        ChatRequest(
            message="Gợi ý outfit đi đám cưới",
            session_id="cf-agent-s1",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res.clarification_asked is True
    assert res.clarification_form is not None
    assert [q.id for q in res.clarification_form.questions] == ["target_gender"]
    agent.structured_model.ainvoke.assert_not_awaited()
    assert res.debug_meta is not None
    assert res.debug_meta.decision_source == "mode_c_counterfactual"
    assert res.debug_meta.evidence_rule_ids

    assert len(logged) == 1
    assert logged[0]["decision_source"] == "mode_c_counterfactual"
    assert "impact_signals" in logged[0]
    assert "evidence_rule_ids" in logged[0]
    assert "reason_codes" in logged[0]


def test_hard_answer_guard_triggers_unfulfillable_check_and_oos_proceeds() -> None:
    policy = _make_policy()

    # Complete required slots + impossible <100k budget still checks hard constraints
    slots_unfulf, intent_unfulf = extract_conversation_slots(
        ["Phối áo sơ mi với quần tây cho nam đi làm giá dưới 100k."]
    )
    dec_unfulf = evaluate_clarification_policy(
        ClarificationMode.CONDITIONAL_CLARIFICATION,
        intent_unfulf,
        slots_unfulf,
        clarification_turns=0,
        gray_zone_evaluator=policy,
    )
    assert dec_unfulf.action == "ASK"
    assert dec_unfulf.target_slots == ["budget"]
    assert "UNFULFILLABLE_HARD_CONSTRAINT" in dec_unfulf.reason_codes

    # Non-blueprint anchor categories in slots.category proceed directly
    for q in ("Phối váy liền với gì cho đẹp?", "Phối túi xách với trang phục gì hợp?"):
        slots_oos, intent_oos = extract_conversation_slots([q])
        dec_oos = evaluate_clarification_policy(
            ClarificationMode.CONDITIONAL_CLARIFICATION,
            intent_oos,
            slots_oos,
            clarification_turns=0,
            gray_zone_evaluator=policy,
        )
        assert dec_oos.action == "PROCEED"
        assert "COUNTERFACTUAL_OUT_OF_SCOPE_PROCEED" in dec_oos.reason_codes


def test_anchor_polo_navy_proceeds_with_alternatives() -> None:
    policy = _make_policy()
    slots, intent = extract_conversation_slots(
        ["Áo polo xanh navy phối với quần nào thì hợp?"]
    )
    dec = evaluate_clarification_policy(
        ClarificationMode.CONDITIONAL_CLARIFICATION,
        intent,
        slots,
        clarification_turns=0,
        gray_zone_evaluator=policy,
    )
    assert dec.action == "PROCEED"
    assert "COVERED_WITH_ALTERNATIVES" in dec.reason_codes
