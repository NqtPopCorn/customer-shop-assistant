"""Tests for Clarification Strategy Specification v1.1 (FR-01..FR-13 & Scenarios)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from customer_shop_assistant.schemas.chat import (
    AgentReply,
    ChatRecommendedProduct,
    ChatRequest,
    ClarificationForm,
    ClarificationMode,
    ClarificationQuestion,
    UserProfile,
)
from customer_shop_assistant.schemas.clarification import ClarificationDecision
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.clarification_policy import (
    evaluate_clarification_policy,
)
from customer_shop_assistant.services.session_manager import SessionManager
from customer_shop_assistant.services.slot_tracker import (
    ConversationSlots,
    TurnIntent,
    extract_conversation_slots_with_lifecycle,
    format_slots_for_prompt,
    get_missing_required_slots,
)
from customer_shop_assistant.services.telemetry_logger import log_chat_turn
from customer_shop_assistant.tools.catalog_tool import record_searched_products


class _FakeModel(GenericFakeChatModel):
    last_bound_tools: list[Any] | None = None

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.last_bound_tools = list(tools)
        return self


def test_slot_precedence_and_eight_states() -> None:
    # Current Explicit (PROVIDED) > History (RESOLVED_FROM_CONTEXT) > Profile > Inferred
    profile = UserProfile(gender=GenderTarget.MALE, height_cm=175.0, weight_kg=70.0)
    history = [
        HumanMessage(content="Tìm áo sơ mi đi làm"),
        HumanMessage(content="Tìm đầm dự tiệc cho bạn gái"),
    ]
    slots, intent, switched = extract_conversation_slots_with_lifecycle(
        history, profile
    )
    assert intent == TurnIntent.PRODUCT_SEARCH
    assert switched is True  # office -> party triggers task-scoped reset
    assert slots.target_gender is not None
    assert slots.target_gender.value == "FEMALE"
    assert slots.target_gender.status == "PROVIDED"
    assert slots.occasion is not None
    assert slots.occasion.value == "party"
    assert slots.occasion.status == "PROVIDED"
    # Session-persistent measurements from profile remain intact
    assert slots.measurements is not None
    assert slots.measurements.status == "RESOLVED_FROM_CONTEXT"

    # Cross-turn / profile override MUST NOT raise CONFLICTING
    assert not slots.is_conflicting("target_gender")
    assert slots.status_of("measurements", intent) == "NOT_APPLICABLE"


def test_same_turn_conflict_detection() -> None:
    slots_g, _, _ = extract_conversation_slots_with_lifecycle(
        ["Tìm đầm nữ cho nam"], UserProfile()
    )
    assert slots_g.status_of("target_gender") == "CONFLICTING"
    assert "target_gender" in get_missing_required_slots(
        TurnIntent.PRODUCT_SEARCH, slots_g
    )

    slots_b, _, _ = extract_conversation_slots_with_lifecycle(
        ["Tìm áo sơ mi nam ngân sách dưới 200k nhưng chỉ lấy hàng trên 1 triệu"]
    )
    assert slots_b.status_of("budget") == "CONFLICTING"
    assert "budget" in slots_b.get_conflicting_slots()


def test_topic_switch_lifecycle_keeps_persistent_resets_task_scoped() -> None:
    messages = [
        HumanMessage(content="Tìm áo sơ mi nam đi làm dưới 500k size L"),
        AIMessage(content="Đây là các mẫu áo sơ mi..."),
        HumanMessage(content="Giờ phối giúp mình 1 bộ đi biển"),
    ]
    slots, intent, switched = extract_conversation_slots_with_lifecycle(messages)
    assert switched is True
    assert intent == TurnIntent.OUTFIT_ADVICE
    # Session-persistent kept
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("size") == "L"
    # Task-scoped reset & re-extracted from new turn
    assert slots.value_of("category") is None
    assert slots.value_of("budget") is None
    assert slots.value_of("occasion") == "casual"


def test_prompt_marks_inferred_gender_clearly() -> None:
    slots, intent, _ = extract_conversation_slots_with_lifecycle(["Tìm đầm dự tiệc"])
    summary = format_slots_for_prompt(slots, intent)
    assert "suy luận độ tin cậy cao" in summary


@pytest.mark.anyio
async def test_scenario_3_complete_required_slots_never_clarifies_in_mode_b() -> None:
    # Spec Section 8 Scenario 3: "Tìm áo sơ mi nam." -> Mode B MUST answer immediately
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"keyword": "áo sơ mi nam", "target_gender": "MALE"},
                "id": "call_s3",
            }
        ],
    )
    final_msg = AIMessage(content="Shop gợi ý [101] Áo Sơ Mi Nam Oxford.")
    model = _FakeModel(messages=iter([tool_call, final_msg]))
    agent = ShopAssistantAgent(chat_model=model, session_manager=SessionManager())

    async def _mock_search(*args: Any, **kwargs: Any) -> str:
        record_searched_products(
            [ChatRecommendedProduct(id=101, name="Áo Sơ Mi Nam Oxford", price=420000.0)]
        )
        return "- [101] Áo Sơ Mi Nam Oxford | 420,000đ"

    mock_search = AsyncMock()
    mock_search.name = "search_products"
    mock_search.ainvoke.side_effect = _mock_search
    agent.tools_by_name["search_products"] = mock_search

    res = await agent.chat(
        ChatRequest(
            message="Tìm áo sơ mi nam.",
            session_id="scen-3-mode-b",
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    assert res.clarification_asked is False
    assert res.clarification_form is None
    assert "search_products" in res.tools_used
    assert [p.id for p in res.products] == [101]
    assert res.debug_meta is not None
    assert res.debug_meta.decision_source == "mode_b_required_check"


@pytest.mark.anyio
async def test_scenario_4_size_advice_across_all_three_modes() -> None:
    # Mode A: Strict 0-clarification -> returns standard Men's Tops size chart
    model_a = _FakeModel(messages=iter([AIMessage(content="ignored")]))
    agent_a = ShopAssistantAgent(chat_model=model_a, session_manager=SessionManager())
    res_a = await agent_a.chat(
        ChatRequest(
            message="Áo sơ mi nam này mình mặc size gì?",
            session_id="scen-4-mode-a",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    assert res_a.clarification_asked is False
    assert res_a.clarification_form is None
    assert "Áo Nam" in res_a.reply
    assert "| Size |" in res_a.reply

    # Mode B: Missing measurements -> asks measurements only
    model_b = _FakeModel(
        messages=iter([AIMessage(content="Bạn cho mình xin chiều cao cân nặng nhé!")])
    )
    agent_b = ShopAssistantAgent(chat_model=model_b, session_manager=SessionManager())
    res_b = await agent_b.chat(
        ChatRequest(
            message="Áo sơ mi nam này mình mặc size gì?",
            session_id="scen-4-mode-b",
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    assert res_b.clarification_asked is True
    assert res_b.clarification_form is not None
    assert [q.id for q in res_b.clarification_form.questions] == ["measurements"]

    # Mode C: Hard Ask Guard -> asks measurements
    model_c = _FakeModel(
        messages=iter([AIMessage(content="Bạn cho mình xin chiều cao cân nặng nhé!")])
    )
    agent_c = ShopAssistantAgent(chat_model=model_c, session_manager=SessionManager())
    res_c = await agent_c.chat(
        ChatRequest(
            message="Áo sơ mi nam này mình mặc size gì?",
            session_id="scen-4-mode-c",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res_c.clarification_asked is True
    assert res_c.clarification_form is not None
    assert [q.id for q in res_c.clarification_form.questions] == ["measurements"]
    assert res_c.debug_meta is not None
    assert res_c.debug_meta.decision_source == "mode_c_hard_ask"


@pytest.mark.anyio
async def test_mode_b_two_turn_cap_and_declined_fallback() -> None:
    session_mgr = SessionManager()
    session_id = "mode-b-2turn-cap"

    # Turn 1: "Tìm đồ cho mình" -> missing target_gender & category -> asks both
    model_t1 = _FakeModel(messages=iter([AIMessage(content="Bạn chọn giúp mình nhé.")]))
    agent_t1 = ShopAssistantAgent(chat_model=model_t1, session_manager=session_mgr)
    res1 = await agent_t1.chat(
        ChatRequest(
            message="Tìm đồ cho mình",
            session_id=session_id,
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    assert res1.clarification_asked is True
    assert res1.clarification_form is not None
    assert "target_gender" in [q.id for q in res1.clarification_form.questions]
    assert "category" in [q.id for q in res1.clarification_form.questions]

    # Turn 2: User answers only gender ("Nam"), omitting category -> asks category
    model_t2 = _FakeModel(
        messages=iter([AIMessage(content="Bạn chọn nhóm trang phục giúp mình nhé.")])
    )
    agent_t2 = ShopAssistantAgent(chat_model=model_t2, session_manager=session_mgr)
    res2 = await agent_t2.chat(
        ChatRequest(
            message="Nam",
            session_id=session_id,
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    assert res2.clarification_asked is True
    assert res2.clarification_form is not None
    assert [q.id for q in res2.clarification_form.questions] == ["category"]

    # Turn 3: User declines ("sao cũng được") AND turn cap 2 is reached -> Fallback
    model_t3 = _FakeModel(messages=iter([AIMessage(content="Gợi ý cho bạn.")]))
    agent_t3 = ShopAssistantAgent(chat_model=model_t3, session_manager=session_mgr)

    async def _mock_search(*args: Any, **kwargs: Any) -> str:
        record_searched_products(
            [ChatRecommendedProduct(id=205, name="Áo Polo Nam", price=350000.0)]
        )
        return "- [205] Áo Polo Nam | 350,000đ"

    mock_search = AsyncMock()
    mock_search.name = "search_products"
    mock_search.ainvoke.side_effect = _mock_search
    agent_t3.tools_by_name["search_products"] = mock_search

    res3 = await agent_t3.chat(
        ChatRequest(
            message="Sao cũng được",
            session_id=session_id,
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    assert res3.clarification_asked is False
    assert res3.clarification_form is None
    assert "search_products" in res3.tools_used
    assert [p.id for p in res3.products] == [205]
    assert res3.debug_meta is not None
    assert res3.debug_meta.decision_source == "fallback"


@pytest.mark.anyio
async def test_mode_c_gray_zone_hook_and_scenario_2_anchor_proceed() -> None:
    # Scenario 2: "Phối áo sơ mi trắng với quần gì?" in Mode C -> Gray Zone
    # Verify GrayZoneEvaluator hook can steer Mode C without changing Agent code
    class _StubGrayZoneEvaluator:
        def evaluate(
            self,
            intent: TurnIntent,
            slots: ConversationSlots,
            clarification_turns: int,
        ) -> ClarificationDecision | None:
            if (
                intent == TurnIntent.OUTFIT_ADVICE
                and slots.has("top_category")
                and not slots.has("target_gender")
            ):
                return ClarificationDecision(
                    action="PROCEED",
                    decision_source="mode_c_hard_answer",
                    decision_reason=(
                        "Anchor top_category=shirt present; cover with multi-gender."
                    ),
                    use_fallback=True,
                    fallback_slots=["target_gender"],
                )
            return None

    model = _FakeModel(messages=iter([AIMessage(content="Gợi ý phối đồ cho bạn.")]))
    agent = ShopAssistantAgent(
        chat_model=model,
        session_manager=SessionManager(),
        gray_zone_evaluator=_StubGrayZoneEvaluator(),
    )
    res = await agent.chat(
        ChatRequest(
            message="Phối áo sơ mi trắng với quần gì?",
            session_id="scen-2-hook",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res.clarification_asked is False
    assert res.debug_meta is not None
    assert res.debug_meta.decision_source == "mode_c_hard_answer"

    # Also verify default Mode C LLM gray-zone captures decision_reason when CF off
    from customer_shop_assistant.config import Settings

    model_llm = _FakeModel(
        messages=iter([AIMessage(content="Bạn chọn giới tính giúp mình nhé.")])
    )
    agent_llm = ShopAssistantAgent(
        chat_model=model_llm,
        settings=Settings(_env_file=None, COUNTERFACTUAL_POLICY_ENABLED=False),
        session_manager=SessionManager(),
    )
    agent_llm.structured_model = AsyncMock()
    agent_llm.structured_model.ainvoke.return_value = AgentReply(
        reply="Bạn chọn đối tượng mặc để mình phối chuẩn nhất nhé.",
        needs_clarification=True,
        decision_reason="Party outfit differs greatly between Male and Female.",
        clarification_form=ClarificationForm(
            questions=[
                ClarificationQuestion(
                    id="target_gender",
                    question="1. Bạn tìm đồ dự tiệc cho Nam hay Nữ?",
                    type="single",
                    options=["Nam", "Nữ"],
                )
            ]
        ),
    )
    res_s1 = await agent_llm.chat(
        ChatRequest(
            message="Gợi ý outfit đi đám cưới",
            session_id="scen-1-mode-c-llm",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res_s1.clarification_asked is True
    assert res_s1.debug_meta is not None
    assert res_s1.debug_meta.decision_source == "mode_c_llm"
    assert "Male and Female" in (res_s1.debug_meta.decision_reason or "")


def test_non_shopping_strict_bypass_in_all_modes() -> None:
    slots, intent, _ = extract_conversation_slots_with_lifecycle(
        ["Phí ship bao nhiêu?"]
    )
    assert intent == TurnIntent.NON_SHOPPING
    for mode in (
        ClarificationMode.IMMEDIATE_ANSWER,
        ClarificationMode.ALWAYS_CLARIFY,
        ClarificationMode.CONDITIONAL_CLARIFICATION,
    ):
        dec = evaluate_clarification_policy(mode, intent, slots, clarification_turns=0)
        assert dec.action == "PROCEED"
        assert dec.decision_source == "non_shopping_bypass"


@pytest.mark.anyio
async def test_telemetry_logs_spec_v11_contract(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from customer_shop_assistant.services import telemetry_logger

    written: list[dict[str, Any]] = []

    def _capture_write(log_path: Path, line: str) -> None:
        del log_path
        written.append(json.loads(line))

    monkeypatch.setattr(telemetry_logger, "_write_log_line", _capture_write)

    await log_chat_turn(
        session_id="tel-1",
        turn_id=1,
        user_message="Tìm áo sơ mi nam",
        assistant_message="Gợi ý [101]",
        latency_ms=123.4,
        token_usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        system_prompt_version="v2.1",
        clarification_mode="ALWAYS_CLARIFY",
        turn_intent="PRODUCT_SEARCH",
        clarification_asked=False,
        question_count=0,
        asked_slots=[],
        slot_states={"target_gender": {"value": "MALE", "state": "PROVIDED"}},
        decision_source="mode_b_required_check",
        decision_reason="All required slots are resolved.",
        tools_used=["search_products"],
    )

    assert len(written) == 1
    rec = written[0]
    for key in (
        "timestamp",
        "session_id",
        "turn_id",
        "provider",
        "model",
        "temperature",
        "system_prompt_version",
        "clarification_mode",
        "turn_intent",
        "clarification_asked",
        "question_count",
        "asked_slots",
        "slot_states",
        "decision_source",
        "decision_reason",
        "tools_used",
        "latency_ms",
        "token_usage",
    ):
        assert key in rec, f"Missing telemetry field: {key}"
    assert rec["decision_source"] == "mode_b_required_check"
    assert rec["token_usage"]["total_tokens"] == 150
