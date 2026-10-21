import logging
from typing import Any

import httpx

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.schemas.shop import (
    CategorySchema,
    ProductFilterParams,
    ProductFit,
    ProductReviewSummary,
    ProductSchema,
    ReviewItemSchema,
    VoucherCheckResult,
)

logger = logging.getLogger(__name__)


class ShopClient:
    """Async HTTP client communicating with Main BE (fashion-shop-be) REST API."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        base = self.settings.MAIN_BE_BASE_URL.rstrip("/")
        if not base.endswith("/api"):
            base = f"{base}/api"
        self.base_url = base
        self.timeout = self.settings.MAIN_BE_TIMEOUT
        self._external_client = client
        self._owned_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        if self._owned_client is None or self._owned_client.is_closed:
            self._owned_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        return self._owned_client

    async def close(self) -> None:
        """Close the owned HTTP client if open."""
        if self._owned_client is not None and not self._owned_client.is_closed:
            await self._owned_client.aclose()
            self._owned_client = None

    async def __aenter__(self) -> "ShopClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get_products(
        self,
        filters: ProductFilterParams | None = None,
    ) -> list[ProductSchema]:
        """Fetch list of products from catalog with optional filtering."""
        client = await self._get_client()
        params: dict[str, Any] = {}

        if filters:
            if filters.search:
                params["search"] = filters.search
            if filters.category_id is not None:
                params["categoryId"] = filters.category_id
            if filters.category:
                params["category"] = filters.category
            if filters.brand_id is not None:
                params["brandId"] = filters.brand_id
            if filters.target_gender:
                params["targetGender"] = filters.target_gender.value
            if filters.min_price is not None:
                params["minPrice"] = filters.min_price
            if filters.max_price is not None:
                params["maxPrice"] = filters.max_price
            if filters.mode:
                params["mode"] = filters.mode
            if filters.in_stock_only:
                params["inStock"] = "true"
            if filters.page is not None:
                params["page"] = filters.page
            if filters.limit is not None:
                params["limit"] = filters.limit

        url = f"{self.base_url}/products"
        try:
            logger.info("ShopClient -> GET %s (params=%s)", url, params)
            response = await client.get(url, params=params)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                items = data.get("items") or data.get("data") or []
            elif isinstance(data, list):
                items = data
            else:
                items = []

            return [ProductSchema.model_validate(item) for item in items]
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch products from %s: %s", url, exc)
            return []

    async def get_product_by_id(self, product_id: int) -> ProductSchema | None:
        """Retrieve single product details by integer ID."""
        client = await self._get_client()
        url = f"{self.base_url}/products/{product_id}"
        try:
            logger.info("ShopClient -> GET %s", url)
            response = await client.get(url)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return ProductSchema.model_validate(response.json())
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch product %s from %s: %s", product_id, url, exc)
            return None

    async def get_product_by_slug(self, slug: str) -> ProductSchema | None:
        """Retrieve single product details by unique slug."""
        client = await self._get_client()
        url = f"{self.base_url}/products/slug/{slug}"
        try:
            logger.info("ShopClient -> GET %s", url)
            response = await client.get(url)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return ProductSchema.model_validate(response.json())
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch product slug %s from %s: %s", slug, url, exc)
            return None

    async def check_voucher(
        self,
        code: str,
        order_amount: float = 0.0,
    ) -> VoucherCheckResult:
        """Check voucher validity and calculate discount amount."""
        client = await self._get_client()
        normalized_code = code.strip().upper()
        url = f"{self.base_url}/vouchers/check/{normalized_code}"

        try:
            logger.info("ShopClient -> GET %s (order_amount=%s)", url, order_amount)
            response = await client.get(url)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            if response.status_code in (400, 404):
                error_detail = response.json().get("message", "Voucher không hợp lệ")
                return VoucherCheckResult(
                    is_valid=False,
                    code=normalized_code,
                    message=error_detail,
                )
            response.raise_for_status()
            res_data = response.json()

            voucher_data = res_data.get("voucher", res_data)
            min_order = voucher_data.get("minOrderAmount")
            min_order_float = float(min_order) if min_order is not None else None

            # Check min order amount if provided
            if (
                min_order_float is not None
                and order_amount > 0
                and order_amount < min_order_float
            ):
                return VoucherCheckResult(
                    is_valid=False,
                    code=normalized_code,
                    name=voucher_data.get("name"),
                    min_order_amount=min_order_float,
                    message=(
                        f"Mã '{normalized_code}' yêu cầu đơn hàng tối thiểu "
                        f"từ {min_order_float:,.0f}đ."
                    ),
                )

            # Use discount parameters directly from backend
            disc_type = voucher_data.get("discountType")
            disc_val = float(voucher_data.get("discountValue", 0))

            return VoucherCheckResult(
                is_valid=True,
                code=normalized_code,
                name=voucher_data.get("name"),
                discount_type=disc_type,
                discount_amount=disc_val,
                min_order_amount=min_order_float,
                message=f"Áp dụng thành công mã giảm giá '{normalized_code}'.",
            )
        except httpx.HTTPError as exc:
            logger.error(
                "Error checking voucher %s at %s: %s", normalized_code, url, exc
            )
            return VoucherCheckResult(
                is_valid=False,
                code=normalized_code,
                message="Không thể kết nối đến hệ thống kiểm tra voucher.",
            )

    async def get_product_reviews(
        self,
        product_id: int,
    ) -> ProductReviewSummary:
        """Fetch customer reviews and return raw fit distribution counts."""
        client = await self._get_client()
        url = f"{self.base_url}/products/{product_id}/reviews"

        try:
            logger.info("ShopClient -> GET %s", url)
            response = await client.get(url, params={"limit": 20})
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            if response.status_code == 404:
                return ProductReviewSummary(product_id=product_id)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            stats = data.get("stats", {})

            review_items = [ReviewItemSchema.model_validate(r) for r in items]
            total_reviews = stats.get("total", len(review_items))
            avg_rating = float(stats.get("averageRating", 0.0))

            # Fit distribution counts
            fit_dist = stats.get("fitDistribution") or {}
            tight_count = fit_dist.get("TIGHT", 0)
            true_count = fit_dist.get("TRUE_TO_SIZE", 0)
            loose_count = fit_dist.get("LOOSE", 0)

            # If stats missing, compute from review items
            if not fit_dist and review_items:
                for r in review_items:
                    if r.fit == ProductFit.TIGHT:
                        tight_count += 1
                    elif r.fit == ProductFit.TRUE_TO_SIZE:
                        true_count += 1
                    elif r.fit == ProductFit.LOOSE:
                        loose_count += 1

            fit_counts = {
                ProductFit.TIGHT.value: tight_count,
                ProductFit.TRUE_TO_SIZE.value: true_count,
                ProductFit.LOOSE.value: loose_count,
            }

            return ProductReviewSummary(
                product_id=product_id,
                total_reviews=total_reviews,
                average_rating=avg_rating,
                fit_counts=fit_counts,
                recent_reviews=review_items,
            )
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch reviews for product %s: %s", product_id, exc)
            return ProductReviewSummary(product_id=product_id)

    async def get_categories(self) -> list[CategorySchema]:
        """Fetch categories tree from Main BE."""
        client = await self._get_client()
        url = f"{self.base_url}/categories"
        try:
            logger.info("ShopClient -> GET %s", url)
            response = await client.get(url)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            response.raise_for_status()
            data = response.json()
            items = data if isinstance(data, list) else data.get("items", [])
            return [CategorySchema.model_validate(c) for c in items]
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch categories from %s: %s", url, exc)
            return []

    async def get_user_preferences(
        self, user_id: int, token: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch customer profile preferences (measurements, styles) from BE."""
        client = await self._get_client()
        url = f"{self.base_url}/preferences/{user_id}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            logger.info("ShopClient -> GET %s", url)
            response = await client.get(url, headers=headers)
            logger.info("ShopClient <- GET %s [%d]", url, response.status_code)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch preferences for user %s: %s", user_id, exc)
            return None

    async def update_user_preferences(
        self,
        user_id: int,
        preferences: dict[str, Any],
        token: str | None = None,
    ) -> dict[str, Any] | None:
        """Update customer profile preferences in Main BE."""
        client = await self._get_client()
        url = f"{self.base_url}/preferences/{user_id}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            logger.info("ShopClient -> PUT %s payload=%s", url, preferences)
            response = await client.put(url, json=preferences, headers=headers)
            logger.info("ShopClient <- PUT %s [%d]", url, response.status_code)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.error("Failed to update preferences for user %s: %s", user_id, exc)
            return None
