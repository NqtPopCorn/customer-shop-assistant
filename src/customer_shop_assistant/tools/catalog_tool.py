from langchain_core.tools import tool

from customer_shop_assistant.schemas.chat import ChatRecommendedProduct
from customer_shop_assistant.schemas.shop import (
    GenderTarget,
    ProductFilterParams,
    ProductSchema,
)
from customer_shop_assistant.services.shop_client import ShopClient

# In-memory registry of recent structured products returned by tools
PRODUCT_REGISTRY: dict[int, ChatRecommendedProduct] = {}


def product_schema_to_recommended(p: ProductSchema) -> ChatRecommendedProduct:
    """Map live ProductSchema to frontend ChatRecommendedProduct."""
    prices = [
        v.final_price if v.final_price is not None else v.selling_price
        for v in p.variants
    ]
    min_price = min(prices) if prices else 0.0
    list_prices = [v.list_price for v in p.variants if v.list_price]
    max_list_price = max(list_prices) if list_prices else min_price
    discount_pct = (
        round(((max_list_price - min_price) / max_list_price) * 100)
        if max_list_price > min_price
        else None
    )
    img = (
        p.images[0].url
        if p.images
        else (p.variants[0].image_url if p.variants else None)
    )

    return ChatRecommendedProduct(
        id=p.id,
        name=p.name,
        slug=p.slug,
        price=min_price,
        original_price=max_list_price if max_list_price > min_price else None,
        discount_percent=discount_pct,
        image_url=img,
        category=p.category.name if p.category else None,
        brand=p.brand.name if p.brand else None,
    )


@tool
async def search_products(
    keyword: str | None = None,
    category_id: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    target_gender: str | None = None,
    limit: int = 5,
) -> str:
    """Tìm kiếm danh sách sản phẩm thời trang trong kho theo từ khóa, giá, giới tính."""
    gender_enum = None
    if target_gender:
        norm_gender = target_gender.strip().upper()
        if norm_gender in GenderTarget._value2member_map_:
            gender_enum = GenderTarget(norm_gender)

    params = ProductFilterParams(
        search=keyword,
        category_id=category_id,
        min_price=min_price,
        max_price=max_price,
        target_gender=gender_enum,
        limit=limit,
    )

    client = ShopClient()
    products = await client.get_products(params)
    if not products:
        return "Không tìm thấy sản phẩm nào phù hợp với tiêu chí tìm kiếm."

    # Cache structured products for frontend carousel
    for p in products:
        PRODUCT_REGISTRY[p.id] = product_schema_to_recommended(p)

    items_text: list[str] = []
    for p in products:
        in_stock_sizes = sorted({v.size for v in p.variants if v.size and v.stock > 0})
        size_str = ", ".join(in_stock_sizes) if in_stock_sizes else "Hết hàng"

        prices = [
            v.final_price if v.final_price is not None else v.selling_price
            for v in p.variants
        ]
        price_val = min(prices) if prices else 0.0
        price_str = f"{price_val:,.0f}đ" if price_val > 0 else "Liên hệ"

        material_str = f" | Chất liệu: {p.material}" if p.material else ""
        gender_str = f" | Phù hợp: {p.target_gender.value}" if p.target_gender else ""
        items_text.append(
            f"- [{p.id}] {p.name} | Giá từ: {price_str} | Size: {size_str}"
            f"{material_str}{gender_str}"
        )

    return "\n".join(items_text)


@tool
async def get_product_detail(product_id: int | str) -> str:
    """Xem thông tin chi tiết một sản phẩm: chất liệu, mô tả, tồn kho và đánh giá."""
    try:
        pid = int(product_id)
    except (ValueError, TypeError):
        return f"Mã sản phẩm không hợp lệ: {product_id}"

    client = ShopClient()
    product = await client.get_product_by_id(pid)
    if not product:
        return f"Không tìm thấy sản phẩm với mã ID: {pid}"

    PRODUCT_REGISTRY[product.id] = product_schema_to_recommended(product)

    reviews = await client.get_product_reviews(product.id)

    prices = [
        v.final_price if v.final_price is not None else v.selling_price
        for v in product.variants
    ]
    price_val = min(prices) if prices else 0.0

    lines: list[str] = [
        f"Sản phẩm: {product.name} (ID: {product.id})",
        f"Giá: {price_val:,.0f}đ",
        f"Chất liệu: {product.material or 'N/A'}",
        f"Phù hợp: {product.target_gender.value if product.target_gender else 'N/A'}",
        f"Mô tả: {product.description or 'Không có mô tả'}",
        "Biến thể & Tồn kho:",
    ]

    for v in product.variants:
        v_size = v.size or "F"
        v_color = v.color or "Tiêu chuẩn"
        lines.append(f"  - Size: {v_size} | Màu: {v_color} | Còn: {v.stock}")

    if reviews and reviews.total_reviews > 0:
        fit_txt = (
            reviews.dominant_fit.value if reviews.dominant_fit else "Chưa đủ dữ liệu"
        )
        lines.append(
            f"Đánh giá: {reviews.average_rating}/5⭐ ({reviews.total_reviews} lượt) "
            f"| Tín hiệu form dáng: {fit_txt}"
        )
        if reviews.fit_recommendation:
            lines.append(f"Ghi chú: {reviews.fit_recommendation}")
    else:
        lines.append("Đánh giá: Chưa có lượt đánh giá nào.")

    return "\n".join(lines)
