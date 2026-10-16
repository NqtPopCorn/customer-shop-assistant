from typing import Any

import httpx
import pytest

from customer_shop_assistant.config import Settings
from customer_shop_assistant.schemas.shop import (
    GenderTarget,
    ProductFilterParams,
    ProductFit,
)
from customer_shop_assistant.services.shop_client import ShopClient


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        MAIN_BE_BASE_URL="http://test-server/api",
        MAIN_BE_TIMEOUT=2.0,
    )


@pytest.mark.anyio
async def test_get_products_with_filter(test_settings: Settings) -> None:
    mock_products = [
        {
            "id": 1,
            "name": "Áo Thun Cotton Compact",
            "slug": "ao-thun-cotton-compact",
            "targetGender": "MALE",
            "ratingAverage": 4.8,
            "reviewCount": 12,
            "variants": [
                {
                    "id": 101,
                    "sku": "TSHIRT-M-BLK",
                    "size": "M",
                    "color": "Đen",
                    "listPrice": 250000,
                    "sellingPrice": 220000,
                    "stock": 15,
                }
            ],
            "images": [
                {"id": 1, "url": "http://img.jpg", "altText": "front", "sortOrder": 0}
            ],
        },
        {
            "id": 2,
            "name": "Váy Suông Hoa Nhí",
            "slug": "vay-suong-hoa-nhi",
            "targetGender": "FEMALE",
            "ratingAverage": 5.0,
            "reviewCount": 8,
            "variants": [
                {
                    "id": 102,
                    "sku": "DRESS-S-WHT",
                    "size": "S",
                    "color": "Trắng",
                    "listPrice": 450000,
                    "sellingPrice": 400000,
                    "stock": 0,
                }
            ],
            "images": [],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/products"
        assert request.url.params.get("targetGender") == "MALE"
        return httpx.Response(200, json=mock_products)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = ShopClient(settings=test_settings, client=mock_http)
        filters = ProductFilterParams(
            target_gender=GenderTarget.MALE, in_stock_only=True
        )
        products = await client.get_products(filters)

        assert len(products) == 1
        assert products[0].id == 1
        assert products[0].name == "Áo Thun Cotton Compact"
        assert products[0].target_gender == GenderTarget.MALE
        assert products[0].variants[0].stock == 15


@pytest.mark.anyio
async def test_get_product_by_id_found_and_not_found(
    test_settings: Settings,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/products/1":
            return httpx.Response(
                200,
                json={
                    "id": 1,
                    "name": "Áo Polo Pique",
                    "slug": "ao-polo-pique",
                    "targetGender": "MALE",
                    "variants": [],
                    "images": [],
                },
            )
        return httpx.Response(404, json={"message": "Not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = ShopClient(settings=test_settings, client=mock_http)

        product = await client.get_product_by_id(1)
        assert product is not None
        assert product.name == "Áo Polo Pique"

        missing = await client.get_product_by_id(999)
        assert missing is None


@pytest.mark.anyio
async def test_check_voucher_success_and_failures(
    test_settings: Settings,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "SUMMER2026" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "valid": True,
                    "voucher": {
                        "id": 10,
                        "code": "SUMMER2026",
                        "name": "Voucher Hè 10%",
                        "discountType": "PERCENT",
                        "discountValue": 10,
                        "maxDiscountValue": 50000,
                        "minOrderAmount": 300000,
                    },
                },
            )
        return httpx.Response(404, json={"message": "Mã khuyến mãi không tồn tại"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = ShopClient(settings=test_settings, client=mock_http)

        # Case 1: Below minimum order amount
        res1 = await client.check_voucher("SUMMER2026", order_amount=200000)
        assert res1.is_valid is False
        assert "tối thiểu" in (res1.message or "")

        # Case 2: Above minimum order amount with cap
        res2 = await client.check_voucher("SUMMER2026", order_amount=600000)
        assert res2.is_valid is True
        # 10% of 600k = 60k, but maxDiscountValue is 50k
        assert res2.discount_amount == 50000

        # Case 3: Invalid code
        res3 = await client.check_voucher("EXPIREDCODE")
        assert res3.is_valid is False
        assert "không tồn tại" in (res3.message or "")


@pytest.mark.anyio
async def test_get_product_reviews_and_fit_recommendation(
    test_settings: Settings,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        reviews_data: dict[str, Any] = {
            "items": [
                {
                    "id": i,
                    "productId": 1,
                    "rating": 5,
                    "fit": "TIGHT",
                    "userHeight": 170,
                    "userWeight": 65,
                }
                for i in range(12)
            ],
            "stats": {
                "total": 12,
                "averageRating": 4.9,
                "fitDistribution": {"TIGHT": 10, "TRUE_TO_SIZE": 2, "LOOSE": 0},
            },
        }
        return httpx.Response(200, json=reviews_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        client = ShopClient(settings=test_settings, client=mock_http)
        summary = await client.get_product_reviews(1)

        assert summary.total_reviews == 12
        assert summary.average_rating == 4.9
        assert summary.dominant_fit == ProductFit.TIGHT
        assert summary.fit_recommendation is not None
        assert "tăng 1 size" in summary.fit_recommendation
