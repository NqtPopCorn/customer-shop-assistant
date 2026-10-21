from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.schemas.chat import (
    AgentReply,
    ChatRequest,
    ClarificationMode,
)
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.session_manager import (
    ClarifyPhase,
    SessionManager,
)


class InspectableFakeChatModel(GenericFakeChatModel):
    last_bound_tools: list[Any] | None = None

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        self.last_bound_tools = list(tools)
        return self


@pytest.mark.anyio
async def test_mode_b_turn1_gating_and_turn2_unlock() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-mode-b"

    # Turn 1: LLM asks clarification
    t1_reply = AIMessage(content="Bạn đang tìm trang phục cho dịp nào?")
    t1_model = InspectableFakeChatModel(messages=iter([t1_reply]))
    agent_t1 = ShopAssistantAgent(chat_model=t1_model, session_manager=session_mgr)
    agent_t1.structured_model = AsyncMock()
    agent_t1.structured_model.ainvoke.return_value = AgentReply(
        reply="Bạn đang tìm trang phục cho dịp nào?",
        needs_clarification=True,
        product_ids=[],
        quick_replies=["Đi làm", "Đi chơi"],
    )

    req1 = ChatRequest(
        message="Tìm cho mình áo sơ mi đẹp",
        session_id=session_id,
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    res1 = await agent_t1.chat(req1)

    assert res1.clarification_asked is True
    # Verify search_products was NOT bound in turn 1
    bound_tool_names_t1 = [
        getattr(t, "name", str(t)) for t in (t1_model.last_bound_tools or [])
    ]
    assert "search_products" not in bound_tool_names_t1
    assert "recommend_size" in bound_tool_names_t1

    # Check session state advanced to OPEN after turn 1
    sess = session_mgr.get_session(session_id)
    assert sess.clarification_mode == ClarificationMode.ALWAYS_CLARIFY.value
    assert sess.clarify_phase == ClarifyPhase.OPEN
    assert sess.gated_turns == 1

    # Turn 2: LLM calls search_products now that phase is OPEN
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"query": "áo sơ mi công sở"},
                "id": "call_search_1",
            }
        ],
    )
    t2_reply = AIMessage(content="Shop có mẫu sơ mi lụa rất hợp đi làm.")
    t2_model = InspectableFakeChatModel(messages=iter([tool_call_msg, t2_reply]))
    agent_t2 = ShopAssistantAgent(chat_model=t2_model, session_manager=session_mgr)
    agent_t2.structured_model = AsyncMock()
    agent_t2.structured_model.ainvoke.return_value = AgentReply(
        reply="Shop có mẫu sơ mi lụa rất hợp đi làm.",
        needs_clarification=False,
        product_ids=["prod_101"],
        quick_replies=["Xem chi tiết", "Chọn size"],
    )

    req2 = ChatRequest(
        message="Mình cần mặc đi làm văn phòng",
        session_id=session_id,
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    res2 = await agent_t2.chat(req2)

    # In turn 2, search_products must be bound
    bound_tool_names_t2 = [
        getattr(t, "name", str(t)) for t in (t2_model.last_bound_tools or [])
    ]
    assert "search_products" in bound_tool_names_t2
    assert "search_products" in res2.tools_used
    assert not res2.clarification_asked


@pytest.mark.anyio
async def test_tool_execution_guard_blocks_hallucinated_call() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-guard"

    # Turn 1: LLM hallucinates calling search_products despite it not being bound
    hallucinated_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"query": "áo thun"},
                "id": "call_search_hallucinated",
            }
        ],
    )
    fallback_reply = AIMessage(content="Bạn muốn tìm áo thun dáng ôm hay rộng rãi?")
    fake_model = InspectableFakeChatModel(
        messages=iter([hallucinated_tool_call, fallback_reply])
    )
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=session_mgr)
    agent.structured_model = AsyncMock()
    agent.structured_model.ainvoke.return_value = AgentReply(
        reply="Bạn muốn tìm áo thun dáng ôm hay rộng rãi?",
        needs_clarification=True,
        product_ids=[],
        quick_replies=["Dáng ôm", "Rộng rãi"],
    )

    req = ChatRequest(
        message="Tìm áo thun cho mình",
        session_id=session_id,
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    res = await agent.chat(req)

    # The tool call was guarded; search_products was NOT executed
    assert "search_products" not in res.tools_used
    assert res.clarification_asked is True


@pytest.mark.anyio
async def test_mode_a_immediate_answer_binds_search() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-mode-a"

    t1_reply = AIMessage(content="Shop có nhiều mẫu áo khoác gió đẹp.")
    fake_model = InspectableFakeChatModel(messages=iter([t1_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=session_mgr)

    req = ChatRequest(
        message="Tìm áo khoác gió",
        session_id=session_id,
        clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
    )
    res = await agent.chat(req)

    bound_tools = [
        getattr(t, "name", str(t)) for t in (fake_model.last_bound_tools or [])
    ]
    assert "search_products" in bound_tools
    sess = session_mgr.get_session(session_id)
    assert sess is not None
    assert sess.clarification_mode == ClarificationMode.IMMEDIATE_ANSWER.value
    assert not res.clarification_asked


@pytest.mark.anyio
async def test_mode_switch_resets_state_machine() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-switch"

    # Start with Mode A
    sess = session_mgr.get_or_create_session(session_id)
    sess.clarification_mode = ClarificationMode.IMMEDIATE_ANSWER.value
    sess.clarify_phase = ClarifyPhase.OPEN
    sess.gated_turns = 1

    # User switches to Mode B in next request
    t1_reply = AIMessage(content="Bạn cần tư vấn phong cách nào?")
    fake_model = InspectableFakeChatModel(messages=iter([t1_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=session_mgr)
    agent.structured_model = AsyncMock()
    agent.structured_model.ainvoke.return_value = AgentReply(
        reply="Bạn cần tư vấn phong cách nào?",
        needs_clarification=True,
        product_ids=[],
        quick_replies=[],
    )

    req = ChatRequest(
        message="Tư vấn cho mình",
        session_id=session_id,
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    await agent.chat(req)

    # Since it switched to Mode B, turn 1 must have run in INTERVIEW mode
    # (and search_products was NOT bound)
    bound_tools = [
        getattr(t, "name", str(t)) for t in (fake_model.last_bound_tools or [])
    ]
    assert "search_products" not in bound_tools
    # Session now reflects Mode B with 1 gated turn completed, advancing to OPEN
    assert sess.clarification_mode == ClarificationMode.ALWAYS_CLARIFY.value
    assert sess.clarify_phase == ClarifyPhase.OPEN
    assert sess.gated_turns == 1
