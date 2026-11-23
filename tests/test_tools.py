import httpx
import pytest

from customer_shop_assistant.tools.catalog_tool import (
    get_product_detail,
    search_products,
)
from customer_shop_assistant.tools.policy_tool import search_policy
from customer_shop_assistant.tools.size_tool import recommend_size
from customer_shop_assistant.tools.voucher_tool import check_voucher


@pytest.mark.anyio
async def test_recommend_size_tool() -> None:
    result = await recommend_size.ainvoke(
        {
            "category": "tops",
            "height_cm": 172.0,
            "weight_kg": 64.0,
            "gender": "male",
            "fit_preference": "regular",
        }
    )
    assert "Size đề xuất: M" in result
    assert "chọn L" in result
    assert "Độ tin cậy:" in result
    assert "Giải thích:" in result


def test_search_policy_tool() -> None:
    result = search_policy.invoke({"query": "đổi hàng trong mấy ngày?"})
    assert "returns.md" in result
    assert "7 ngày" in result
    assert "Nguồn trích dẫn:" in result


@pytest.mark.anyio
async def test_search_products_tool_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_payload = {
        "items": [
            {
                "id": 1,
                "name": "Áo Thun Unisex Basic",
                "slug": "ao-thun-unisex-basic",
                "material": "100% Cotton",
                "targetGender": "UNISEX",
                "variants": [
                    {
                        "id": 10,
                        "sku": "AT-M-BLK",
                        "size": "M",
                        "color": "Đen",
                        "listPrice": 250000,
                        "sellingPrice": 199000,
                        "finalPrice": 199000,
                        "stock": 10,
                    }
                ],
            }
        ]
    }

    async def mock_get(*args, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=mock_payload,
            request=httpx.Request("GET", "http://test/products"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    result = await search_products.ainvoke({"keyword": "thun"})
    assert "[1]" in result
    assert "Áo Thun Unisex Basic" in result
    assert "199,000đ" in result
    assert "Size: M" in result


@pytest.mark.anyio
async def test_search_products_normalizes_english_category_to_semantic_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_params: dict[str, object] = {}

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        if params:
            captured_params.update(params)
        return httpx.Response(
            status_code=200,
            json={"items": []},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    await search_products.ainvoke(
        {"keyword": "đi tiệc", "category": "dresses", "target_gender": "female"}
    )
    assert captured_params.get("mode") == "semantic"
    assert "đầm váy" in str(captured_params.get("search", ""))
    assert "category" not in captured_params
    assert captured_params.get("targetGender") == "FEMALE"


@pytest.mark.anyio
async def test_search_products_preserves_valid_vietnamese_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_params: dict[str, object] = {}

    async def mock_get(self, url, params=None, **kwargs) -> httpx.Response:
        if params:
            captured_params.update(params)
        return httpx.Response(
            status_code=200,
            json={"items": []},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    await search_products.ainvoke({"keyword": "đầm dự tiệc", "category": "Đầm dạ hội"})
    assert captured_params.get("mode") == "semantic"
    assert captured_params.get("search") == "đầm dự tiệc"
    assert captured_params.get("category") == "Đầm dạ hội"


@pytest.mark.anyio
async def test_get_product_detail_tool_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_prod = {
        "id": 1,
        "name": "Áo Thun Unisex Basic",
        "slug": "ao-thun-unisex-basic",
        "material": "100% Cotton",
        "targetGender": "UNISEX",
        "description": "Áo thun cotton thoáng mát",
        "variants": [
            {
                "id": 10,
                "sku": "AT-M-BLK",
                "size": "M",
                "color": "Đen",
                "listPrice": 250000,
                "sellingPrice": 199000,
                "finalPrice": 199000,
                "stock": 10,
            }
        ],
    }

    async def mock_get(self, url, *args, **kwargs) -> httpx.Response:
        if "/reviews" in str(url):
            return httpx.Response(
                status_code=200,
                json={"items": [], "stats": {"total": 0, "averageRating": 0.0}},
                request=httpx.Request("GET", str(url)),
            )
        return httpx.Response(
            status_code=200,
            json=mock_prod,
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    result = await get_product_detail.ainvoke({"product_id": 1})
    assert "Áo Thun Unisex Basic" in result
    assert "100% Cotton" in result
    assert "Size: M" in result


@pytest.mark.anyio
async def test_voucher_tool_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_payload = {
        "code": "SUMMER50",
        "name": "Khuyến Mãi Hè",
        "discountType": "FIXED",
        "discountValue": 50000,
        "minOrderValue": 300000,
        "startDate": "2026-01-01T00:00:00.000Z",
        "endDate": "2026-12-31T23:59:59.000Z",
        "usageLimit": 100,
        "usedCount": 5,
    }

    async def mock_get(*args, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=mock_payload,
            request=httpx.Request("GET", "http://test/vouchers/check/SUMMER50"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    result = await check_voucher.ainvoke({"code": "SUMMER50", "order_value": 400000})
    assert "HỢP LỆ" in result
    assert "50,000đ" in result


@pytest.mark.anyio
async def test_concurrent_turns_have_isolated_product_contexts() -> None:
    import asyncio

    from customer_shop_assistant.schemas.chat import ChatRecommendedProduct
    from customer_shop_assistant.tools.catalog_tool import (
        get_cached_product,
        get_latest_searched_product_ids,
        product_turn_context,
        record_searched_products,
    )

    async def turn_a() -> tuple[list[int], ChatRecommendedProduct | None]:
        with product_turn_context():
            record_searched_products(
                [ChatRecommendedProduct(id=57, name="Đầm Dạ Hội", price=490000.0)]
            )
            await asyncio.sleep(0.05)
            # Even after turn_b runs concurrently, turn_a only sees its own product
            assert get_cached_product(101) is None
            return get_latest_searched_product_ids(), get_cached_product(57)

    async def turn_b() -> tuple[list[int], ChatRecommendedProduct | None]:
        with product_turn_context():
            await asyncio.sleep(0.01)
            record_searched_products(
                [ChatRecommendedProduct(id=101, name="Áo Sơ Mi Nam", price=350000.0)]
            )
            assert get_cached_product(57) is None
            return get_latest_searched_product_ids(), get_cached_product(101)

    (ids_a, prod_a), (ids_b, prod_b) = await asyncio.gather(turn_a(), turn_b())
    assert ids_a == [57]
    assert prod_a is not None and prod_a.name == "Đầm Dạ Hội"
    assert ids_b == [101]
    assert prod_b is not None and prod_b.name == "Áo Sơ Mi Nam"


@pytest.mark.anyio
async def test_empty_search_clears_searched_ids_and_context_resets_after_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.schemas.chat import ChatRecommendedProduct
    from customer_shop_assistant.tools.catalog_tool import (
        get_cached_product,
        get_latest_searched_product_ids,
        product_turn_context,
        record_searched_products,
    )

    async def mock_empty_get(self, url, params=None, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"items": []},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_empty_get)

    with product_turn_context():
        record_searched_products(
            [ChatRecommendedProduct(id=57, name="Đầm Dạ Hội", price=490000.0)]
        )
        assert get_latest_searched_product_ids() == [57]

        # Subsequent empty search must clear searched_ids in the same turn
        res = await search_products.ainvoke({"keyword": "không tồn tại"})
        assert "Không tìm thấy" in res
        assert get_latest_searched_product_ids() == []

    # After exiting context, nothing leaks globally
    assert get_latest_searched_product_ids() == []
    assert get_cached_product(57) is None
