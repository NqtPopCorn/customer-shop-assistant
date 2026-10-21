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
