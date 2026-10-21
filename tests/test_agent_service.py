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
from customer_shop_assistant.services.agent_service import ShopAssistantAgent


class MockChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.mark.anyio
async def test_agent_clarification_response() -> None:
    mock_clarification = AIMessage(
        content=(
            "Để tư vấn size chuẩn xác nhất, bạn vui lòng cho mình "
            "biết chiều cao và cân nặng nhé!"
        )
    )
    fake_model = MockChatModel(messages=iter([mock_clarification]))
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
async def test_agent_tool_calling_loop() -> None:
    # First turn: call check_voucher tool; Second turn: summarize result
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_voucher",
                "args": {"code": "SUMMER50", "order_value": 600000.0},
                "id": "call_voucher_1",
            }
        ],
    )
    final_reply_msg = AIMessage(
        content="Mã SUMMER50 áp dụng thành công, bạn được giảm 50.000đ!"
    )
    fake_model = MockChatModel(messages=iter([tool_call_msg, final_reply_msg]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    req = ChatRequest(
        message="Kiểm tra mã SUMMER50 cho đơn 600k",
        clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
    )

    res = await agent.chat(req)
    assert "check_voucher" in res.tools_used
    assert "SUMMER50" in res.reply
    assert "giảm 50.000đ" in res.reply


@pytest.mark.anyio
async def test_agent_chat_stream() -> None:
    mock_reply = AIMessage(content="Chào bạn, shop có rất nhiều mẫu áo thun mới về!")
    fake_model = MockChatModel(messages=iter([mock_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    req = ChatRequest(
        message="Có mẫu áo thun nào mới không?",
        clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
    )

    chunks = []
    async for chunk in agent.chat_stream(req):
        chunks.append(chunk)

    types = [c["type"] for c in chunks]
    assert "token" in types
    assert "done" in types

    done_chunk = next(c for c in chunks if c["type"] == "done")
    assert "áo thun mới về" in done_chunk["reply"]


@pytest.mark.anyio
async def test_agent_structured_output_reply() -> None:
    from unittest.mock import AsyncMock

    from customer_shop_assistant.schemas.chat import AgentReply

    fake_model = MockChatModel(messages=iter([AIMessage(content="Draft reply")]))
    agent = ShopAssistantAgent(chat_model=fake_model)

    # Mock structured_model returning AgentReply
    mock_structured = AsyncMock()
    mock_structured.ainvoke.return_value = AgentReply(
        reply="Shop gợi ý 2 mẫu áo polo phù hợp dáng người bạn.",
        needs_clarification=False,
        product_ids=["101", "102"],
        quick_replies=["Xem thêm màu khác", "Tư vấn chất liệu"],
    )
    agent.structured_model = mock_structured

    req = ChatRequest(
        message="Gợi ý áo polo",
        clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
    )
    res = await agent.chat(req)

    assert res.reply == "Shop gợi ý 2 mẫu áo polo phù hợp dáng người bạn."
    assert not res.clarification_asked
    assert res.quick_replies == ["Xem thêm màu khác", "Tư vấn chất liệu"]
    mock_structured.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_agent_fallback_json_reply() -> None:
    json_payload = (
        '{"reply": "Bạn cần thêm thông tin gì?", '
        '"needs_clarification": true, '
        '"product_ids": [], '
        '"quick_replies": ["Tư vấn size", "Xem khuyến mãi"]}'
    )
    fake_model = MockChatModel(messages=iter([AIMessage(content=json_payload)]))
    agent = ShopAssistantAgent(chat_model=fake_model)
    # Ensure structured_model is None so it exercises fallback json parsing
    agent.structured_model = None

    req = ChatRequest(
        message="Alo shop ơi",
        clarification_mode=ClarificationMode.MINIMAL_CLARIFICATION,
    )
    res = await agent.chat(req)

    assert res.reply == "Bạn cần thêm thông tin gì?"
    assert res.clarification_asked is True
    assert res.quick_replies == ["Tư vấn size", "Xem khuyến mãi"]
