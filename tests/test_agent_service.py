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
    )
    agent.structured_model = mock_structured

    req = ChatRequest(
        message="Gợi ý áo polo",
        clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
    )
    res = await agent.chat(req)

    assert res.reply == "Shop gợi ý 2 mẫu áo polo phù hợp dáng người bạn."
    assert not res.clarification_asked
    mock_structured.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_agent_fallback_json_reply() -> None:
    json_payload = (
        '{"reply": "Bạn cần thêm thông tin gì?", '
        '"needs_clarification": true, '
        '"product_ids": []}'
    )
    fake_model = MockChatModel(messages=iter([AIMessage(content=json_payload)]))
    agent = ShopAssistantAgent(chat_model=fake_model)
    # Ensure structured_model is None so it exercises fallback json parsing
    agent.structured_model = None

    req = ChatRequest(
        message="Tìm áo thun đi chơi",
        clarification_mode=ClarificationMode.MINIMAL_CLARIFICATION,
    )
    res = await agent.chat(req)

    assert res.reply == "Bạn cần thêm thông tin gì?"
    assert res.clarification_asked is True


@pytest.mark.anyio
async def test_agent_clarification_form_and_sanitization() -> None:
    leaked_reply = (
        "Để mình chọn mẫu trang phục phù hợp nhất, bạn cho mình biết thêm nhé:\n"
        "1. Đi làm công sở\n"
        "2. Đi chơi dạo phố\n"
        "3. Ngân sách 300k - 600k"
    )
    fake_model = MockChatModel(messages=iter([AIMessage(content=leaked_reply)]))
    agent = ShopAssistantAgent(chat_model=fake_model)
    agent.structured_model = None

    req = ChatRequest(
        message="Tư vấn đồ đi chơi cho mình",
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    res = await agent.chat(req)

    assert res.clarification_asked is True
    # Verify inline numbered list was replaced by clean 1-sentence intro
    assert "1. Đi làm công sở" not in res.reply
    assert "đi chơi" in res.reply.lower()
    # Verify clarification_form was built with P0 target_gender first
    assert res.clarification_form is not None
    assert len(res.clarification_form.questions) >= 2
    assert res.clarification_form.questions[0].id == "target_gender"
    types = [q.type for q in res.clarification_form.questions]
    assert "single" in types


@pytest.mark.anyio
async def test_agent_clarification_aligns_reply_and_known_occasion() -> None:
    mismatched_llm_reply = (
        "Chào bạn! Để chọn mẫu trang phục phù hợp nhất cho buổi tiệc cuối tuần, "
        "mình cần một chút thông tin nhỏ nhé: - Chiều cao (cm) - Cân nặng (kg) "
        "- Số đo 3 vòng (dài, rộng, eo) Sau khi có số đo, shop sẽ tư vấn size "
        "chuẩn và gợi ý vài mẫu trang phục đẹp mắt cho bạn!"
    )
    fake_model = MockChatModel(messages=iter([AIMessage(content=mismatched_llm_reply)]))
    agent = ShopAssistantAgent(chat_model=fake_model)
    agent.structured_model = None

    req = ChatRequest(
        message="Gợi ý đồ đi tiệc cuối tuần 🥂",
        clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
    )
    res = await agent.chat(req)

    assert res.clarification_asked is True
    # Conflicting 3-measurements demand must be removed from intro reply
    assert "Số đo 3 vòng" not in res.reply
    assert "Chiều cao (cm)" not in res.reply
    assert "buổi tiệc" in res.reply

    # Form must NOT re-ask "dịp nào?" since user already specified "tiệc cuối tuần"
    assert res.clarification_form is not None
    q_titles = [q.question.lower() for q in res.clarification_form.questions]
    assert not any("dịp nào" in qt for qt in q_titles)
    assert any("dự tiệc" in qt for qt in q_titles)


@pytest.mark.anyio
async def test_agent_skips_structured_output_and_preserves_markdown_newlines() -> None:
    from unittest.mock import AsyncMock

    markdown_reply = (
        "Gợi ý phối đồ công sở lịch lãm:\n"
        "- Áo sơ mi Oxford phối quần âu\n"
        "- Áo polo tối giản kết hợp quần kaki"
    )
    fake_model_imm = MockChatModel(messages=iter([AIMessage(content=markdown_reply)]))
    agent_imm = ShopAssistantAgent(chat_model=fake_model_imm)
    agent_imm.structured_model = AsyncMock()

    res_imm = await agent_imm.chat(
        ChatRequest(
            message="Tìm áo khoác gió nam",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    assert "\n- Áo sơ mi Oxford" in res_imm.reply
    agent_imm.structured_model.ainvoke.assert_not_awaited()

    non_shopping_reply = "Cảm ơn bạn đã ghé shop!\nRất vui được hỗ trợ bạn."
    fake_model_ns = MockChatModel(
        messages=iter([AIMessage(content=non_shopping_reply)])
    )
    agent_ns = ShopAssistantAgent(chat_model=fake_model_ns)
    agent_ns.structured_model = AsyncMock()

    res_ns = await agent_ns.chat(
        ChatRequest(
            message="Cảm ơn shop nhiều nhé",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert "\nRất vui" in res_ns.reply
    agent_ns.structured_model.ainvoke.assert_not_awaited()


def test_ollama_reasoning_setting_and_factory_propagation() -> None:
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    from customer_shop_assistant.config import Settings
    from customer_shop_assistant.llm.factory import get_chat_model

    s_default = Settings(_env_file=None)
    assert s_default.OLLAMA_REASONING is False

    s_true = Settings(_env_file=None, OLLAMA_REASONING=True)
    assert s_true.OLLAMA_REASONING is True

    s_empty = Settings(_env_file=None, OLLAMA_REASONING="")
    assert s_empty.OLLAMA_REASONING is None

    s_none_str = Settings(_env_file=None, OLLAMA_REASONING="none")
    assert s_none_str.OLLAMA_REASONING is None

    ollama_false = get_chat_model(
        Settings(_env_file=None, LLM_PROVIDER="ollama", OLLAMA_REASONING=False)
    )
    assert isinstance(ollama_false, ChatOllama)
    assert ollama_false.reasoning is False

    ollama_default = get_chat_model(
        Settings(_env_file=None, LLM_PROVIDER="ollama", OLLAMA_REASONING="")
    )
    assert isinstance(ollama_default, ChatOllama)
    assert ollama_default.reasoning is None

    lmstudio_model = get_chat_model(
        Settings(_env_file=None, LLM_PROVIDER="lmstudio", OLLAMA_REASONING=False)
    )
    assert isinstance(lmstudio_model, ChatOpenAI)


@pytest.mark.anyio
async def test_agent_extracts_bracketed_id_prefix_and_caps_products_at_five() -> None:
    from unittest.mock import AsyncMock

    from customer_shop_assistant.schemas.chat import ChatRecommendedProduct
    from customer_shop_assistant.tools.catalog_tool import record_searched_products

    fake_reply = AIMessage(
        content=(
            "1. `[ID] 485` - Áo sơ mi Broadway\n"
            "2. `[ID] 473` - Áo sơ mi Holly\n"
            "3. `[ID] 425` - Áo sơ mi Boris\n"
            "4. `[ID] 60` - Áo sơ mi Bd\n"
            "5. `[ID] 437` - Áo sơ mi Ca\n"
            "6. `[ID] 476` - Áo sơ mi Darling\n"
            "7. `[ID] 431` - Áo sơ mi Td\n"
        )
    )
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"keyword": "quần âu", "limit": 10},
                "id": "call_cap_1",
            }
        ],
    )
    agent = ShopAssistantAgent(
        chat_model=MockChatModel(messages=iter([tool_call_msg, fake_reply]))
    )

    mock_prods = [
        ChatRecommendedProduct(id=pid, name=f"Mẫu {pid}", price=350000.0)
        for pid in (485, 473, 425, 60, 437, 476, 431)
    ]

    async def _mock_search(*args: Any, **kwargs: Any) -> str:
        record_searched_products(mock_prods)
        return "\n".join(f"- [{p.id}] {p.name}" for p in mock_prods)

    mock_search = AsyncMock()
    mock_search.name = "search_products"
    mock_search.ainvoke.side_effect = _mock_search
    agent.tools_by_name["search_products"] = mock_search

    res = await agent.chat(
        ChatRequest(
            message="gợi ý vài loại quần phù hợp",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    assert len(res.products) == 5
    assert [p.id for p in res.products] == [485, 473, 425, 60, 437]
