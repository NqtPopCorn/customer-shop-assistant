from typing import Any
from unittest.mock import AsyncMock

import httpx
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


@pytest.mark.anyio
async def test_broad_open_inquiry_does_not_echo_welcome_and_avoids_duplicate() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-handsome-men"

    # Simulate model generating questions with markdown table
    raw_ai_text = (
        "Chào bạn! 😊\n"
        "Tôi hiểu bạn đang tìm kiếm phong cách thời trang đẹp trai. "
        "Tuy nhiên để tư vấn chính xác nhất mình cần thêm một số thông tin nhé:\n"
        "| Câu hỏi | Lựa chọn |\n"
        "|---------|----------|\n"
        "| Loại đồ | Sơ mi hoặc polo |\n"
    )
    fake_model = InspectableFakeChatModel(
        messages=iter([AIMessage(content=raw_ai_text)])
    )
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=session_mgr)

    from customer_shop_assistant.schemas.chat import ChatMessage

    req = ChatRequest(
        message="tôi muốn thật đẹp trai",
        session_id=session_id,
        clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        history=[
            ChatMessage(
                role="assistant",
                content=(
                    "Xin chào! 👋 Mình là Stylist AI của Fashion Shop. "
                    "Mình có thể giúp bạn tìm kiếm trang phục, gợi ý phối đồ "
                    "theo dịp hoặc tư vấn chọn size chuẩn xác. Bạn đang quan "
                    "tâm đến phong cách nào hôm nay?"
                ),
            ),
            ChatMessage(role="user", content="tôi muốn thật đẹp trai"),
        ],
    )

    res = await agent.chat(req)

    # 1. No duplicate human message in session memory
    human_msgs = [
        m for m in session_mgr.get_session(session_id).messages if m.type == "human"
    ]
    assert len(human_msgs) == 1
    assert human_msgs[0].content == "tôi muốn thật đẹp trai"

    # 2. It asked clarification and built a form
    assert res.clarification_asked is True
    assert res.clarification_form is not None

    # 3. Form asks for occasion first
    q1 = next(q for q in res.clarification_form.questions if q.id == "occasion")
    assert any("Đi làm công sở" in opt for opt in q1.options)
    assert any("Dự tiệc" in opt for opt in q1.options)

    # 4. Reply does NOT echo the welcome greeting
    assert "bạn đang quan tâm đến phong cách nào hôm nay" not in res.reply.lower()
    assert "|" not in res.reply  # Cleaned markdown table


@pytest.mark.anyio
async def test_mode_b_does_not_gate_non_shopping_and_preserves_interview() -> None:
    session_mgr = SessionManager()
    session_id = "test-session-non-shopping-mode-b"

    # Turn 1: Policy question must NOT be gated or transition phase to OPEN
    t1_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_policy",
                "args": {"query": "phí ship"},
                "id": "call_policy_1",
            }
        ],
    )
    t1_reply = AIMessage(
        content="Shop miễn phí ship cho đơn từ 500k [shipping.md: Phí vận chuyển]."
    )
    t1_model = InspectableFakeChatModel(messages=iter([t1_tool_call, t1_reply]))
    agent_t1 = ShopAssistantAgent(chat_model=t1_model, session_manager=session_mgr)
    agent_t1.structured_model = None

    res1 = await agent_t1.chat(
        ChatRequest(
            message="Phí ship của shop bao nhiêu?",
            session_id=session_id,
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    bound_t1 = [getattr(t, "name", str(t)) for t in (t1_model.last_bound_tools or [])]
    assert "search_products" in bound_t1
    assert res1.clarification_asked is False
    sess = session_mgr.get_session(session_id)
    assert sess is not None
    assert sess.clarify_phase == ClarifyPhase.INTERVIEW
    assert sess.gated_turns == 0

    # Turn 2: Shopping inquiry in same session STILL gets interviewed & asks gender
    t2_reply = AIMessage(content="Chào bạn, bạn chọn thêm thông tin bên dưới nhé!")
    t2_model = InspectableFakeChatModel(messages=iter([t2_reply]))
    agent_t2 = ShopAssistantAgent(chat_model=t2_model, session_manager=session_mgr)
    agent_t2.structured_model = None

    res2 = await agent_t2.chat(
        ChatRequest(
            message="Tìm quần đi làm",
            session_id=session_id,
            clarification_mode=ClarificationMode.ALWAYS_CLARIFY,
        )
    )
    bound_t2 = [getattr(t, "name", str(t)) for t in (t2_model.last_bound_tools or [])]
    assert "search_products" not in bound_t2
    assert res2.clarification_asked is True
    assert res2.clarification_form is not None
    assert res2.clarification_form.questions[0].id == "target_gender"
    assert sess.clarify_phase == ClarifyPhase.OPEN
    assert sess.gated_turns == 1


@pytest.mark.anyio
async def test_conditional_clarification_llm_autonomous_search_vs_ask() -> None:
    session_mgr = SessionManager()

    # Case A: Sufficient P0 -> LLM searches directly without clarification
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {
                    "keyword": "áo sơ mi công sở",
                    "target_gender": "MALE",
                    "max_price": 500000,
                },
                "id": "call_search_cond",
            }
        ],
    )
    search_reply = AIMessage(content="Shop gợi ý cho bạn mẫu sơ mi Oxford nam.")
    model_a = InspectableFakeChatModel(messages=iter([tool_call_msg, search_reply]))
    agent_a = ShopAssistantAgent(chat_model=model_a, session_manager=session_mgr)
    agent_a.structured_model = AsyncMock()
    agent_a.structured_model.ainvoke.return_value = AgentReply(
        reply="Shop gợi ý cho bạn mẫu sơ mi Oxford nam.",
        needs_clarification=False,
        product_ids=["1"],
    )

    res_a = await agent_a.chat(
        ChatRequest(
            message="Tìm áo sơ mi nam đi làm dưới 500k",
            session_id="cond-search-direct",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res_a.clarification_asked is False
    assert "search_products" in res_a.tools_used

    # Case B: Missing critical P0 (gender) -> LLM asks and keeps custom valid form
    from customer_shop_assistant.schemas.chat import (
        ClarificationForm,
        ClarificationQuestion,
    )

    custom_form = ClarificationForm(
        title="Chọn nhanh đối tượng",
        questions=[
            ClarificationQuestion(
                id="target_gender",
                question="1. Bạn tìm quần đi làm cho Nam hay Nữ?",
                type="single",
                options=["Nam", "Nữ"],
            )
        ],
    )
    model_b = InspectableFakeChatModel(
        messages=iter([AIMessage(content="Bạn chọn đối tượng giúp mình nhé.")])
    )
    agent_b = ShopAssistantAgent(chat_model=model_b, session_manager=session_mgr)
    agent_b.structured_model = AsyncMock()
    agent_b.structured_model.ainvoke.return_value = AgentReply(
        reply="Bạn chọn đối tượng giúp mình nhé.",
        needs_clarification=True,
        clarification_form=custom_form,
        product_ids=[],
    )

    res_b = await agent_b.chat(
        ChatRequest(
            message="Tìm quần đi làm",
            session_id="cond-ask-gender",
            clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
        )
    )
    assert res_b.clarification_asked is True
    assert res_b.clarification_form == custom_form


@pytest.mark.anyio
async def test_mode_a_never_clarifies_and_fallbacks_to_search_if_llm_asks() -> None:
    from customer_shop_assistant.schemas.chat import ChatRecommendedProduct
    from customer_shop_assistant.tools.catalog_tool import record_searched_products

    captured_prompts: list[str] = []

    class PromptCapturingFakeChatModel(InspectableFakeChatModel):
        async def ainvoke(
            self, input: Any, config: Any = None, **kwargs: Any
        ) -> AIMessage:
            if isinstance(input, list) and input:
                captured_prompts.append(str(input[0].content))
            return await super().ainvoke(input, config=config, **kwargs)

    # Tiny local model ignores prompt and asks a clarifying question without tool call
    bad_clarify_reply = AIMessage(
        content=(
            "Chào bạn! Để tư vấn chính xác nhất, bạn đang tìm trang phục cho "
            "đối tượng nào? (Nam/Nữ)"
        )
    )
    fake_model = PromptCapturingFakeChatModel(messages=iter([bad_clarify_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=SessionManager())
    agent.structured_model = None

    sample_product = ChatRecommendedProduct(
        id=57,
        name="Đầm Dạ Hội Ren Hoa",
        price=490000.0,
        category="Đầm dạ hội",
    )

    async def _mock_search_invoke(*args: Any, **kwargs: Any) -> str:
        record_searched_products([sample_product])
        return "- [57] Đầm Dạ Hội Ren Hoa | Giá từ: 490,000đ | Size: S, M, L"

    mock_search = AsyncMock()
    mock_search.name = "search_products"
    mock_search.ainvoke.side_effect = _mock_search_invoke
    agent.tools_by_name["search_products"] = mock_search

    res = await agent.chat(
        ChatRequest(
            message="Gợi ý đồ đi tiệc cuối tuần 🥂",
            session_id="mode-a-no-clarify",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )

    assert captured_prompts
    assert "Còn thiếu (theo thứ tự ưu tiên)" not in captured_prompts[0]
    assert res.clarification_asked is False
    assert res.clarification_form is None
    assert "search_products" in res.tools_used
    assert len(res.products) >= 1
    assert res.products[0].id == 57
    assert "Đầm Dạ Hội Ren Hoa" in res.reply
    mock_search.ainvoke.assert_awaited_once()


@pytest.mark.anyio
async def test_mode_a_overrides_hallucinated_female_args_and_runs_balanced_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import ToolMessage

    captured_calls: list[dict[str, Any]] = []
    captured_tool_messages: list[str] = []

    def _item(pid: int, name: str, gender: str, cat: str) -> dict[str, Any]:
        return {
            "id": pid,
            "name": name,
            "slug": f"prod-{pid}",
            "category": {"id": pid, "name": cat},
            "targetGender": gender,
            "variants": [
                {
                    "id": pid * 10,
                    "sku": f"SKU-{pid}",
                    "size": "M",
                    "listPrice": 600000,
                    "sellingPrice": 550000,
                    "finalPrice": 550000,
                    "stock": 4,
                }
            ],
        }

    items_by_gender = {
        "FEMALE": [
            _item(101, "Đầm Dạ Hội Nữ Esme", "FEMALE", "Đầm dạ hội"),
            _item(102, "Váy Liền Nữ Dora", "FEMALE", "Váy liền"),
        ],
        "MALE": [
            _item(201, "Áo Sơ Mi Nam Oxford", "MALE", "Áo sơ mi"),
            _item(202, "Quần Âu Nam Slim", "MALE", "Quần âu"),
        ],
        "UNISEX": [
            _item(301, "Áo Khoác Blazer Unisex", "UNISEX", "Áo khoác"),
        ],
    }

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        call_params = dict(params or {})
        captured_calls.append(call_params)
        gender = str(call_params.get("targetGender", ""))
        limit = int(call_params.get("limit", 5))
        return httpx.Response(
            status_code=200,
            json={"items": items_by_gender.get(gender, [])[:limit]},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    class MessageCapturingFakeModel(InspectableFakeChatModel):
        async def ainvoke(
            self, input: Any, config: Any = None, **kwargs: Any
        ) -> AIMessage:
            if isinstance(input, list):
                for m in input:
                    if isinstance(m, ToolMessage):
                        captured_tool_messages.append(str(m.content))
            return await super().ainvoke(input, config=config, **kwargs)

    hallucinated_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {
                    "keyword": "đầm dạ hội sang trọng",
                    "category": "Đầm dạ hội",
                    "target_gender": "female",
                },
                "id": "call_mode_a_biased",
            }
        ],
    )
    final_reply = AIMessage(
        content=(
            "Vì bạn chưa nêu giới tính, mình gợi ý Nữ [101], [102], "
            "Nam [201], [202] và Unisex [301]."
        )
    )
    fake_model = MessageCapturingFakeModel(
        messages=iter([hallucinated_tool_call, final_reply])
    )
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=SessionManager())
    agent.structured_model = None

    res = await agent.chat(
        ChatRequest(
            message="Gợi ý đồ đi tiệc cuối tuần 🥂",
            session_id="mode-a-balanced-party",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )

    assert len(captured_calls) == 3
    by_gender = {c["targetGender"]: c for c in captured_calls}
    assert set(by_gender) == {"FEMALE", "MALE", "UNISEX"}
    for call in captured_calls:
        assert "category" not in call
    assert "đầm" not in str(by_gender["MALE"]["search"]).lower()
    assert "đầm" not in str(by_gender["UNISEX"]["search"]).lower()

    assert captured_tool_messages
    tool_out = captured_tool_messages[-1]
    assert "=== GỢI Ý CHO NỮ ===" in tool_out
    assert "=== GỢI Ý CHO NAM ===" in tool_out
    assert "=== GỢI Ý UNISEX ===" in tool_out

    assert res.clarification_asked is False
    assert [p.id for p in res.products] == [101, 102, 201, 202, 301]


@pytest.mark.anyio
async def test_mode_a_respects_explicit_gender_without_balanced_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        captured_calls.append(dict(params or {}))
        return httpx.Response(
            status_code=200,
            json={
                "items": [
                    {
                        "id": 201,
                        "name": "Áo Sơ Mi Nam Công Sở",
                        "slug": "ao-so-mi-nam",
                        "targetGender": "MALE",
                        "variants": [
                            {
                                "id": 2010,
                                "sku": "SM-M",
                                "size": "M",
                                "listPrice": 480000,
                                "sellingPrice": 420000,
                                "finalPrice": 420000,
                                "stock": 6,
                            }
                        ],
                    }
                ]
            },
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {
                    "keyword": "áo sơ mi công sở",
                    "target_gender": "MALE",
                },
                "id": "call_mode_a_male",
            }
        ],
    )
    final_reply = AIMessage(content="Gợi ý cho bạn mẫu [201] Áo Sơ Mi Nam Công Sở.")
    fake_model = InspectableFakeChatModel(messages=iter([tool_call, final_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=SessionManager())
    agent.structured_model = None

    res = await agent.chat(
        ChatRequest(
            message="Tìm áo sơ mi nam công sở",
            session_id="mode-a-explicit-male",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )

    assert len(captured_calls) == 1
    assert captured_calls[0].get("targetGender") == "MALE"
    assert [p.id for p in res.products] == [201]


@pytest.mark.anyio
async def test_mode_a_respects_female_inferred_garment_like_dress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        captured_calls.append(dict(params or {}))
        return httpx.Response(
            status_code=200,
            json={
                "items": [
                    {
                        "id": 105,
                        "name": "Đầm Dạ Hội Ren",
                        "slug": "dam-da-hoi-ren",
                        "targetGender": "FEMALE",
                        "variants": [
                            {
                                "id": 1050,
                                "sku": "DD-S",
                                "size": "S",
                                "listPrice": 850000,
                                "sellingPrice": 780000,
                                "finalPrice": 780000,
                                "stock": 3,
                            }
                        ],
                    }
                ]
            },
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {
                    "keyword": "đầm dự tiệc",
                    "target_gender": "FEMALE",
                },
                "id": "call_mode_a_dress",
            }
        ],
    )
    final_reply = AIMessage(content="Shop gợi ý mẫu [105] Đầm Dạ Hội Ren.")
    fake_model = InspectableFakeChatModel(messages=iter([tool_call, final_reply]))
    agent = ShopAssistantAgent(chat_model=fake_model, session_manager=SessionManager())
    agent.structured_model = None

    res = await agent.chat(
        ChatRequest(
            message="Tìm đầm đi tiệc",
            session_id="mode-a-inferred-dress",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )

    assert len(captured_calls) == 1
    assert captured_calls[0].get("targetGender") == "FEMALE"
    assert "đầm" in str(captured_calls[0].get("search", "")).lower()
    assert [p.id for p in res.products] == [105]
