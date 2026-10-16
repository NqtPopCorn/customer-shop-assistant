from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.agent_service import (
    ShopAssistantAgent,
    classify_intent,
)


class MockChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_classify_intents() -> None:
    assert classify_intent("Cao 1m72 nặng 65kg mặc size gì?") == "SIZE_ADVICE"
    assert classify_intent("Chính sách đổi trả hàng như thế nào?") == "POLICY_INQUIRY"
    assert classify_intent("Mã voucher SUMMER50 giảm bao nhiêu?") == "CATALOG_SEARCH"
    assert classify_intent("Tư vấn giúp mình đồ đi tiệc cho nữ") == "PRODUCT_ADVICE"
    assert classify_intent("Xin chào shop") == "GENERAL_CHAT"


@pytest.mark.anyio
async def test_agent_clarification_gate_minimal_mode() -> None:
    fake_model = MockChatModel(messages=iter([]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    req = ChatRequest(
        message="Tư vấn size áo giúp mình",
        clarification_mode=ClarificationMode.MINIMAL_CLARIFICATION,
    )

    res = await agent.chat(req)
    assert res.clarification_asked
    assert "chiều cao" in res.reply
    assert "cân nặng" in res.reply


@pytest.mark.anyio
async def test_agent_immediate_answer_with_profile() -> None:
    mock_reply = AIMessage(
        content="Với chiều cao 1m72 và cân nặng 65kg, bạn nên chọn size L."
    )
    fake_model = MockChatModel(messages=iter([mock_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    req = ChatRequest(
        message="Gợi ý size áo thun nam",
        clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        user_profile=UserProfile(
            height_cm=172.0,
            weight_kg=65.0,
            gender=GenderTarget.MALE,
        ),
    )

    res = await agent.chat(req)
    assert not res.clarification_asked
    assert "size L" in res.reply


@pytest.mark.anyio
async def test_agent_chat_stream_clarification() -> None:
    fake_model = MockChatModel(messages=iter([]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    req = ChatRequest(
        message="Tư vấn size",
        clarification_mode=ClarificationMode.MINIMAL_CLARIFICATION,
    )

    chunks = []
    async for chunk in agent.chat_stream(req):
        chunks.append(chunk)

    types = [c["type"] for c in chunks]
    assert "intent" in types
    assert "clarification" in types
    assert "done" in types
