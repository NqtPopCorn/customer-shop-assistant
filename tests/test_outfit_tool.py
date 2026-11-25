"""Unit and integration tests for suggest_outfit tool and OUTFIT_KG_ENABLED toggle."""

from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import Settings
from customer_shop_assistant.schemas.chat import ChatRequest, ClarificationMode
from customer_shop_assistant.schemas.shop import (
    CategorySchema,
    GenderTarget,
    ProductSchema,
    ProductVariantSchema,
)
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.tools.catalog_tool import (
    get_latest_outfits,
    product_turn_context,
)
from customer_shop_assistant.tools.outfit_tool import suggest_outfit


def _make_product(
    pid: int,
    name: str,
    category: str,
    gender: GenderTarget,
    variants: list[tuple[str, int, float]],
    slug: str | None = None,
) -> ProductSchema:
    return ProductSchema(
        id=pid,
        name=name,
        slug=slug or f"slug-{pid}",
        targetGender=gender,
        category=CategorySchema(id=pid, name=category),
        variants=[
            ProductVariantSchema(
                id=pid * 10 + idx,
                sku=f"SKU-{pid}-{idx}",
                size="M",
                color=col,
                listPrice=price,
                sellingPrice=price,
                finalPrice=price,
                stock=stock,
            )
            for idx, (col, stock, price) in enumerate(variants)
        ],
    )


def _sample_products() -> list[ProductSchema]:
    return [
        _make_product(
            1,
            "Áo Polo Pique Nam",
            "Áo polo",
            GenderTarget.MALE,
            [("Xanh navy", 5, 350000.0), ("Đỏ tươi", 3, 350000.0)],
        ),
        _make_product(
            2,
            "Quần Kaki Chinos Nam",
            "Quần kaki",
            GenderTarget.MALE,
            [("Be sữa", 0, 450000.0), ("Xanh lá", 4, 450000.0)],
        ),
        _make_product(
            3,
            "Quần Jeans Slim Nam",
            "Quần jeans",
            GenderTarget.MALE,
            [("Đen", 6, 490000.0)],
        ),
        _make_product(
            4,
            "Áo Khoác Bomber Nam",
            "Áo khoác",
            GenderTarget.MALE,
            [("Đen", 2, 650000.0)],
        ),
        _make_product(
            5,
            "Áo Sơ Mi Nữ Lụa",
            "Áo sơ mi",
            GenderTarget.FEMALE,
            [("Trắng", 5, 420000.0)],
        ),
        _make_product(
            6,
            "Chân Váy Xếp Ly Nữ",
            "Chân váy",
            GenderTarget.FEMALE,
            [("Đen", 3, 380000.0)],
        ),
        _make_product(
            7,
            "Váy Liền Dự Tiệc",
            "Váy liền",
            GenderTarget.FEMALE,
            [("Đỏ đô", 2, 790000.0)],
        ),
    ]


def test_agent_respects_outfit_kg_enabled_flag() -> None:
    class FakeModel(GenericFakeChatModel):
        last_tools: list[Any] | None = None

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            self.last_tools = list(tools)
            return self

    m_on = FakeModel(messages=iter([AIMessage(content="ok")]))
    agent_on = ShopAssistantAgent(
        chat_model=m_on, settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True)
    )
    assert "suggest_outfit" in agent_on.tools_by_name

    m_off = FakeModel(messages=iter([AIMessage(content="ok")]))
    agent_off = ShopAssistantAgent(
        chat_model=m_off, settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=False)
    )
    assert "suggest_outfit" not in agent_off.tools_by_name


@pytest.mark.anyio
async def test_agent_chat_populates_outfits_and_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": [p.model_dump(by_alias=True) for p in _sample_products()],
        "meta": {"total": 7, "page": 1, "limit": 2000, "totalPages": 1},
    }

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "suggest_outfit",
                "args": {"product_id": 1, "color": "Xanh navy", "occasion": "đi làm"},
                "id": "call_outfit_1",
            }
        ],
    )
    final_msg = AIMessage(
        content=(
            "Gợi ý phối đồ cho [1] Áo Polo Pique Nam (Xanh navy): "
            "phối cùng [2] Quần Kaki Chinos Nam màu Xanh lá."
        )
    )
    agent = ShopAssistantAgent(
        chat_model=FakeToolModel(messages=iter([tool_call_msg, final_msg])),
        settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True),
    )

    res = await agent.chat(
        ChatRequest(
            message="Áo polo xanh navy mã 1 phối quần gì đi làm?",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )

    assert "suggest_outfit" in res.tools_used
    assert len(res.outfits) >= 1
    assert res.outfits[0].top.product_id == 1
    assert res.outfits[0].bottom.product_id == 2
    assert any(p.id == 2 for p in res.products)
    assert "Quần Kaki Chinos Nam" in res.reply


@pytest.mark.anyio
async def test_suggest_open_with_category_and_color_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": [p.model_dump(by_alias=True) for p in _sample_products()],
        "meta": {"total": 7, "page": 1, "limit": 2000, "totalPages": 1},
    }

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    with product_turn_context():
        text = await suggest_outfit.ainvoke(
            {
                "top_category": "Áo polo",
                "top_color": "Xanh navy",
                "bottom_category": "Quần kaki",
            }
        )
        outfits = get_latest_outfits()
        assert len(outfits) == 1
        assert outfits[0].top.product_id == 1
        assert outfits[0].top.color == "Xanh navy"
        assert outfits[0].bottom.product_id == 2
        assert outfits[0].bottom.in_stock is True
        assert outfits[0].bottom.color == "Xanh lá"
        assert "[1]" in text and "[2]" in text

        oos_text = await suggest_outfit.ainvoke({"top_category": "Váy liền"})
        assert "chưa hỗ trợ" in oos_text
        assert get_latest_outfits() == []


@pytest.mark.anyio
async def test_strict_outfit_intent_redirects_search_and_fallback_returns_combos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": [p.model_dump(by_alias=True) for p in _sample_products()],
        "meta": {"total": 7, "page": 1, "limit": 2000, "totalPages": 1},
    }

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    class InspectableToolModel(GenericFakeChatModel):
        last_bound_tools: list[Any] | None = None

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            self.last_bound_tools = list(tools)
            return self

    # 1) LLM tries to call search_products -> blocked from bound tools & redirected
    hallucinated_search = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_products",
                "args": {"keyword": "áo polo xanh navy", "target_gender": "MALE"},
                "id": "call_wrong_search_1",
            }
        ],
    )
    final_msg = AIMessage(
        content=(
            "Bộ 1: [1] Áo Polo Pique Nam (Xanh navy) "
            "+ [2] Quần Kaki Chinos Nam (Xanh lá)."
        )
    )
    m_redirect = InspectableToolModel(messages=iter([hallucinated_search, final_msg]))
    agent_redirect = ShopAssistantAgent(
        chat_model=m_redirect,
        settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True),
    )

    res1 = await agent_redirect.chat(
        ChatRequest(
            message="Áo polo màu xanh navy phối với quần kaki đi làm",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    bound_names = [
        getattr(t, "name", str(t)) for t in (m_redirect.last_bound_tools or [])
    ]
    assert "search_products" not in bound_names
    assert "suggest_outfit" in bound_names
    assert "suggest_outfit" in res1.tools_used
    assert "search_products" not in res1.tools_used
    assert len(res1.outfits) >= 1
    assert res1.outfits[0].top.product_id == 1
    assert res1.outfits[0].bottom.product_id == 2
    assert {p.id for p in res1.products} >= {1, 2}

    # 2) IMMEDIATE_ANSWER fallback when LLM asks instead of calling tool -> combo reply
    m_fallback = InspectableToolModel(
        messages=iter(
            [AIMessage(content="Bạn cho mình biết thêm màu sắc yêu thích nhé?")]
        )
    )
    agent_fallback = ShopAssistantAgent(
        chat_model=m_fallback,
        settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True),
    )
    res2 = await agent_fallback.chat(
        ChatRequest(
            message="Gợi ý set đồ áo polo và quần jeans cho nam đi chơi",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    assert res2.clarification_asked is False
    assert "suggest_outfit" in res2.tools_used
    assert len(res2.outfits) >= 1
    assert res2.outfits[0].top.product_id == 1
    assert res2.outfits[0].bottom.product_id == 3
    assert "[1]" in res2.reply and "[3]" in res2.reply
