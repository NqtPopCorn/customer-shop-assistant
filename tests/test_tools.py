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

    async def mock_post(*args, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "items": mock_payload["items"],
                "meta": {"total": 1, "strategy": "hybrid_rrf"},
            },
            request=httpx.Request("POST", "http://test/products/search"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await search_products.ainvoke({"keyword": "thun"})
    assert "[1]" in result
    assert "Áo Thun Unisex Basic" in result
    assert "199,000đ" in result
    assert "Size: M" in result


@pytest.mark.anyio
async def test_search_products_normalizes_english_category_to_semantic_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        if isinstance(json, dict):
            captured_payload.update(json)
        return httpx.Response(
            status_code=200,
            json={"items": [], "meta": {"total": 0}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await search_products.ainvoke(
        {"keyword": "đi tiệc", "category": "dresses", "target_gender": "female"}
    )
    filters = captured_payload.get("filters") or {}
    assert isinstance(filters, dict)
    assert "đầm váy" in str(captured_payload.get("query", ""))
    assert "categories" not in filters
    assert filters.get("targetGender") == "FEMALE"


@pytest.mark.anyio
async def test_search_products_preserves_valid_vietnamese_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        if isinstance(json, dict):
            captured_payload.update(json)
        return httpx.Response(
            status_code=200,
            json={"items": [], "meta": {"total": 0}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    await search_products.ainvoke({"keyword": "đầm dự tiệc", "category": "Đầm dạ hội"})
    filters = captured_payload.get("filters") or {}
    assert isinstance(filters, dict)
    assert captured_payload.get("query") == "đầm dự tiệc"
    assert filters.get("categories") == ["Đầm dạ hội"]


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

    async def mock_empty_post(self, url, json=None, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"items": [], "meta": {"total": 0}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_empty_post)

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


@pytest.mark.anyio
async def test_search_products_balanced_gender_returns_2_female_2_male_1_unisex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.schemas.shop import GenderTarget
    from customer_shop_assistant.tools.catalog_tool import (
        balanced_gender_search_context,
        get_cached_product_gender,
        get_latest_searched_product_ids,
        product_turn_context,
    )

    captured_calls: list[dict[str, object]] = []

    def _item(pid: int, name: str, gender: str) -> dict[str, object]:
        return {
            "id": pid,
            "name": name,
            "slug": f"prod-{pid}",
            "material": "Cotton",
            "targetGender": gender,
            "variants": [
                {
                    "id": pid * 10,
                    "sku": f"SKU-{pid}",
                    "size": "M",
                    "color": "Đen",
                    "listPrice": 500000,
                    "sellingPrice": 450000,
                    "finalPrice": 450000,
                    "stock": 5,
                }
            ],
        }

    items_by_gender = {
        "FEMALE": [
            _item(101, "Đầm Dự Tiệc Nữ Esme", "FEMALE"),
            _item(102, "Váy Liền Dự Tiệc Nữ Dora", "FEMALE"),
            _item(103, "Chân Váy Xếp Ly Nữ", "FEMALE"),
        ],
        "MALE": [
            _item(201, "Áo Sơ Mi Nam Dự Tiệc Oxford", "MALE"),
            _item(202, "Quần Âu Nam Lịch Lãm", "MALE"),
        ],
        "UNISEX": [
            _item(301, "Áo Khoác Blazer Unisex Minimal", "UNISEX"),
        ],
    }

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        payload = dict(json or {})
        captured_calls.append(payload)
        limits = payload.get("genderLimits") or {}
        combined = [
            *items_by_gender["FEMALE"][: int(limits.get("FEMALE", 2))],
            *items_by_gender["MALE"][: int(limits.get("MALE", 2))],
            *items_by_gender["UNISEX"][: int(limits.get("UNISEX", 1))],
        ]
        return httpx.Response(
            status_code=200,
            json={"items": combined, "meta": {"total": len(combined)}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with product_turn_context(), balanced_gender_search_context(occasion="party"):
        result = await search_products.ainvoke(
            {"keyword": "trang phục dự tiệc lịch sự"}
        )
        searched_ids = get_latest_searched_product_ids()
        gender_map = {pid: get_cached_product_gender(pid) for pid in searched_ids}

    assert len(captured_calls) == 1
    payload = captured_calls[0]
    assert payload.get("diversifyByGender") is True
    assert payload.get("genderLimits") == {"FEMALE": 2, "MALE": 2, "UNISEX": 1}
    gender_queries = payload.get("genderQueries") or {}
    assert isinstance(gender_queries, dict)
    male_kw = str(gender_queries.get("MALE", "")).lower()
    assert "áo sơ mi" in male_kw or "quần âu" in male_kw

    assert "=== GỢI Ý CHO NỮ ===" in result
    assert "=== GỢI Ý CHO NAM ===" in result
    assert "=== GỢI Ý UNISEX ===" in result
    assert searched_ids == [101, 102, 201, 202, 301]
    assert gender_map == {
        101: GenderTarget.FEMALE,
        102: GenderTarget.FEMALE,
        201: GenderTarget.MALE,
        202: GenderTarget.MALE,
        301: GenderTarget.UNISEX,
    }


@pytest.mark.anyio
async def test_search_products_balanced_omits_empty_unisex_without_padding_female(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.tools.catalog_tool import (
        balanced_gender_search_context,
        get_latest_searched_product_ids,
        product_turn_context,
    )

    def _item(pid: int, name: str, gender: str) -> dict[str, object]:
        return {
            "id": pid,
            "name": name,
            "slug": f"prod-{pid}",
            "targetGender": gender,
            "variants": [
                {
                    "id": pid * 10,
                    "sku": f"SKU-{pid}",
                    "size": "M",
                    "listPrice": 450000,
                    "sellingPrice": 400000,
                    "finalPrice": 400000,
                    "stock": 5,
                }
            ],
        }

    items_by_gender = {
        "FEMALE": [
            _item(101, "Đầm Dạ Hội Nữ A", "FEMALE"),
            _item(102, "Đầm Dạ Hội Nữ B", "FEMALE"),
            _item(103, "Đầm Dạ Hội Nữ C", "FEMALE"),
        ],
        "MALE": [
            _item(201, "Áo Sơ Mi Nam A", "MALE"),
            _item(202, "Quần Âu Nam B", "MALE"),
        ],
        "UNISEX": [],
    }

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        combined = [
            *items_by_gender["FEMALE"],
            *items_by_gender["MALE"],
            *items_by_gender["UNISEX"],
        ]
        return httpx.Response(
            status_code=200,
            json={"items": combined, "meta": {"total": len(combined)}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with product_turn_context(), balanced_gender_search_context(occasion="party"):
        result = await search_products.ainvoke(
            {"keyword": "trang phục dự tiệc lịch sự"}
        )
        searched_ids = get_latest_searched_product_ids()

    assert "=== GỢI Ý CHO NỮ ===" in result
    assert "=== GỢI Ý CHO NAM ===" in result
    assert "=== GỢI Ý UNISEX ===" not in result
    assert searched_ids == [101, 102, 201, 202]
    assert 103 not in searched_ids


@pytest.mark.anyio
async def test_search_products_balanced_skipped_when_explicit_gender_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.tools.catalog_tool import (
        balanced_gender_search_context,
        product_turn_context,
    )

    captured_calls: list[dict[str, object]] = []

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        captured_calls.append(dict(json or {}))
        return httpx.Response(
            status_code=200,
            json={"items": [], "meta": {"total": 0}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with product_turn_context(), balanced_gender_search_context():
        await search_products.ainvoke(
            {"keyword": "áo sơ mi công sở", "target_gender": "MALE"}
        )

    assert len(captured_calls) == 1
    assert captured_calls[0].get("diversifyByGender") is False
    assert (captured_calls[0].get("filters") or {}).get("targetGender") == "MALE"

    captured_calls.clear()
    with product_turn_context():
        await search_products.ainvoke({"keyword": "áo sơ mi công sở"})

    assert len(captured_calls) == 1
    assert "targetGender" not in (captured_calls[0].get("filters") or {})


@pytest.mark.anyio
async def test_search_products_caps_limit_at_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customer_shop_assistant.tools.catalog_tool import (
        get_latest_searched_product_ids,
        product_turn_context,
    )

    captured_payload: dict[str, object] = {}
    items = [
        {
            "id": 100 + i,
            "name": f"Quần Âu Nam Mẫu {i}",
            "slug": f"quan-au-{i}",
            "targetGender": "MALE",
            "variants": [
                {
                    "id": 1000 + i,
                    "sku": f"QA-{i}",
                    "size": "31",
                    "listPrice": 400000,
                    "sellingPrice": 400000,
                    "finalPrice": 400000,
                    "stock": 5,
                }
            ],
        }
        for i in range(10)
    ]

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        if isinstance(json, dict):
            captured_payload.update(json)
        req_limit = int(captured_payload.get("limit", 5))
        return httpx.Response(
            status_code=200,
            json={"items": items[:req_limit], "meta": {"total": 10}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with product_turn_context():
        result = await search_products.ainvoke({"keyword": "quần âu", "limit": 10})
        ids = get_latest_searched_product_ids()

    assert captured_payload.get("limit") == 5
    assert len(ids) == 5
    assert result.count("Quần Âu Nam Mẫu") == 5


@pytest.mark.anyio
async def test_search_products_passes_color_size_sort_and_formats_relaxed_facets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}

    async def mock_post(self, url, json=None, **kwargs) -> httpx.Response:
        if isinstance(json, dict):
            captured_payload.update(json)
        return httpx.Response(
            status_code=200,
            json={
                "items": [
                    {
                        "id": 42,
                        "name": "Áo Sơ Mi Oxford Trơn",
                        "slug": "ao-so-mi-oxford",
                        "material": "100% Cotton Oxford",
                        "fit": "Regular Fit",
                        "pattern": "Trơn Basic",
                        "targetGender": "MALE",
                        "variants": [
                            {
                                "id": 420,
                                "sku": "SM-L-WHT",
                                "size": "L",
                                "color": "Trắng",
                                "listPrice": 450000,
                                "sellingPrice": 390000,
                                "finalPrice": 390000,
                                "stock": 8,
                            }
                        ],
                    }
                ],
                "meta": {
                    "total": 1,
                    "strategy": "hybrid_rrf",
                    "relaxedFilters": ["maxPrice"],
                    "facets": {
                        "priceRange": {"min": 350000, "max": 520000},
                        "availableColors": ["Trắng", "Xanh navy"],
                        "availableSizes": ["M", "L", "XL"],
                    },
                },
            },
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await search_products.ainvoke(
        {
            "keyword": "áo sơ mi đi làm",
            "color": "Trắng",
            "size": "L",
            "max_price": 300000,
            "sort": "price_asc",
        }
    )

    filters = captured_payload.get("filters") or {}
    assert isinstance(filters, dict)
    assert filters.get("colors") == ["Trắng"]
    assert filters.get("sizes") == ["L"]
    assert filters.get("maxPrice") == 300000.0
    assert captured_payload.get("sort") == "price_asc"
    assert "nới lỏng tiêu chí: giá tối đa" in result
    assert "Form: Regular Fit" in result
    assert "Họa tiết: Trơn Basic" in result
    assert "Thông tin kho hiện có" in result
