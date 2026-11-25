from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.chat import (
    ChatRecommendedProduct,
    OutfitSuggestion,
)
from customer_shop_assistant.schemas.shop import (
    AssistantSearchFacets,
    AssistantSearchFilters,
    AssistantSearchRequest,
    GenderTarget,
    ProductSchema,
)
from customer_shop_assistant.services.shop_client import ShopClient


@dataclass
class TurnProductContext:
    """Per-turn isolated product cache and searched IDs for a single request."""

    registry: dict[int, ChatRecommendedProduct] = field(default_factory=dict)
    searched_ids: list[int] = field(default_factory=list)
    gender_by_id: dict[int, GenderTarget] = field(default_factory=dict)
    outfits: list[OutfitSuggestion] = field(default_factory=list)


@dataclass(frozen=True)
class BalancedGenderSearchConfig:
    """Per-call config activating balanced FEMALE/MALE/UNISEX retrieval."""

    occasion: str | None = None
    female_limit: int = 2
    male_limit: int = 2
    unisex_limit: int = 1


_TURN_PRODUCT_CTX: ContextVar[TurnProductContext | None] = ContextVar(
    "turn_product_context", default=None
)

_BALANCED_SEARCH_CTX: ContextVar[BalancedGenderSearchConfig | None] = ContextVar(
    "balanced_gender_search_context", default=None
)


@contextmanager
def product_turn_context() -> Iterator[TurnProductContext]:
    """Create an isolated per-turn product context for chat and chat_stream."""
    ctx = TurnProductContext()
    token = _TURN_PRODUCT_CTX.set(ctx)
    try:
        yield ctx
    finally:
        try:
            _TURN_PRODUCT_CTX.reset(token)
        except ValueError:
            _TURN_PRODUCT_CTX.set(None)


@contextmanager
def balanced_gender_search_context(
    occasion: str | None = None,
    female_limit: int = 2,
    male_limit: int = 2,
    unisex_limit: int = 1,
) -> Iterator[BalancedGenderSearchConfig]:
    """Activate balanced multi-gender search inside a single search_products call."""
    cfg = BalancedGenderSearchConfig(
        occasion=occasion,
        female_limit=female_limit,
        male_limit=male_limit,
        unisex_limit=unisex_limit,
    )
    token = _BALANCED_SEARCH_CTX.set(cfg)
    try:
        yield cfg
    finally:
        try:
            _BALANCED_SEARCH_CTX.reset(token)
        except ValueError:
            _BALANCED_SEARCH_CTX.set(None)


def record_searched_products(
    products: list[ChatRecommendedProduct],
    gender_map: dict[int, GenderTarget] | None = None,
) -> None:
    """Record searched products into the active turn context."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return
    ctx.searched_ids.clear()
    for p in products:
        ctx.registry[p.id] = p
        ctx.searched_ids.append(p.id)
        if gender_map and p.id in gender_map:
            ctx.gender_by_id[p.id] = gender_map[p.id]


def record_product(product: ChatRecommendedProduct) -> None:
    """Cache a single resolved product in the active turn context."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return
    ctx.registry[product.id] = product


def get_cached_product(product_id: int) -> ChatRecommendedProduct | None:
    """Lookup a product cached during the active turn."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return None
    return ctx.registry.get(product_id)


def get_cached_product_gender(product_id: int) -> GenderTarget | None:
    """Lookup the target gender cached for a searched product in the active turn."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return None
    return ctx.gender_by_id.get(product_id)


def get_latest_searched_product_ids() -> list[int]:
    """Return a copy of product IDs found by search_products in the active turn."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return []
    return list(ctx.searched_ids)


def record_outfits(outfits: list[OutfitSuggestion]) -> None:
    """Record KG outfit suggestions into the active turn context."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return
    ctx.outfits[:] = list(outfits)


def get_latest_outfits() -> list[OutfitSuggestion]:
    """Return a copy of outfit suggestions recorded in the active turn."""
    ctx = _TURN_PRODUCT_CTX.get()
    if ctx is None:
        return []
    return list(ctx.outfits)


FEMALE_ONLY_CATEGORIES: frozenset[str] = frozenset(
    {
        "đầm dạ hội",
        "váy liền",
        "chân váy",
        "dress",
        "dresses",
        "skirt",
        "skirts",
        "party dress",
        "evening dress",
        "gown",
    }
)


VALID_CATALOG_CATEGORIES: dict[str, str] = {
    "áo thun": "Áo thun",
    "áo sơ mi": "Áo sơ mi",
    "áo polo": "Áo polo",
    "áo khoác": "Áo khoác",
    "quần âu": "Quần âu",
    "quần kaki": "Quần kaki",
    "quần jeans": "Quần jeans",
    "quần short": "Quần short",
    "váy liền": "Váy liền",
    "chân váy": "Chân váy",
    "đầm dạ hội": "Đầm dạ hội",
    "thắt lưng": "Thắt lưng",
    "túi xách": "Túi xách",
    "mũ nón": "Mũ nón",
}

_CATEGORY_TO_VI_KEYWORD: dict[str, str] = {
    "dress": "đầm váy",
    "dresses": "đầm váy",
    "party dress": "đầm dự tiệc",
    "evening dress": "đầm dạ hội",
    "gown": "đầm dạ hội",
    "skirt": "chân váy",
    "skirts": "chân váy",
    "shirt": "áo sơ mi",
    "shirts": "áo sơ mi",
    "t-shirt": "áo thun",
    "t-shirts": "áo thun",
    "tshirt": "áo thun",
    "tshirts": "áo thun",
    "tee": "áo thun",
    "polo": "áo polo",
    "jacket": "áo khoác",
    "jackets": "áo khoác",
    "blazer": "áo khoác blazer",
    "coat": "áo khoác",
    "tops": "áo",
    "top": "áo",
    "pants": "quần",
    "trousers": "quần âu",
    "jeans": "quần jeans",
    "khaki": "quần kaki",
    "kaki": "quần kaki",
    "shorts": "quần short",
    "short": "quần short",
    "bottoms": "quần",
    "accessories": "phụ kiện",
    "bag": "túi xách",
    "bags": "túi xách",
    "hat": "mũ nón",
    "hats": "mũ nón",
    "belt": "thắt lưng",
    "belts": "thắt lưng",
    "đầm": "đầm váy",
    "váy": "đầm váy",
    "áo": "áo",
    "quần": "quần",
}


class SearchProductsInput(BaseModel):
    """Input schema for semantic/hybrid catalog product search."""

    keyword: str = Field(
        ...,
        description=(
            "Mô tả ngắn bằng tiếng Việt về trang phục, dịp mặc hoặc phong cách để "
            "tìm kiếm ngữ nghĩa (BẮT BUỘC, ví dụ: 'trang phục dự tiệc lịch sự', "
            "'áo sơ mi công sở', 'quần jeans đi chơi')."
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            "Tên danh mục tiếng Việt nếu khách đã nêu rõ (ví dụ: 'Áo sơ mi', "
            "'Váy liền', 'Quần jeans'). LUÔN để null nếu khách chưa nêu rõ loại đồ, "
            "KHÔNG hỏi lại khách."
        ),
    )
    min_price: float | int | None = Field(
        default=None, description="Giá tối thiểu theo VNĐ (ví dụ 300000)."
    )
    max_price: float | int | None = Field(
        default=None, description="Giá tối đa theo VNĐ (ví dụ 500000)."
    )
    target_gender: str | None = Field(
        default=None,
        description=(
            "Giới tính mục tiêu: 'MALE', 'FEMALE', hoặc 'UNISEX'. "
            "LUÔN để null nếu khách chưa nêu rõ giới tính, KHÔNG hỏi lại khách."
        ),
    )
    color: str | None = Field(
        default=None,
        description=(
            "Màu sắc khách yêu cầu nếu có (ví dụ: 'Đen', 'Trắng', 'Xanh navy'). "
            "Để null nếu khách chưa chọn màu."
        ),
    )
    size: str | None = Field(
        default=None,
        description=(
            "Kích cỡ/size khách yêu cầu nếu có (ví dụ: 'S', 'M', 'L', '31'). "
            "Để null nếu khách chưa nêu size."
        ),
    )
    sort: str | None = Field(
        default=None,
        description=(
            "Sắp xếp kết quả: 'relevance' (mặc định), 'price_asc' (giá tăng), "
            "'price_desc' (giá giảm), hoặc 'rating_desc' (đánh giá cao)."
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        description="Số lượng sản phẩm tối đa trả về (tối đa 5, mặc định 5).",
    )


def _normalize_gender(target_gender: str | None) -> GenderTarget | None:
    if not target_gender:
        return None
    norm = target_gender.strip().upper()
    if norm in GenderTarget._value2member_map_:
        return GenderTarget(norm)
    if norm in ("NAM", "MEN", "MAN", "BOY"):
        return GenderTarget.MALE
    if norm in ("NỮ", "NU", "WOMEN", "WOMAN", "GIRL"):
        return GenderTarget.FEMALE
    if norm in ("UNISEX", "ALL"):
        return GenderTarget.UNISEX
    return None


def _normalize_search_terms(
    keyword: str | None, category: str | None
) -> tuple[str, str | None]:
    """Ensure semantic search always gets a Vietnamese keyword and valid category."""
    resolved_kw = (keyword or "").strip()
    resolved_cat: str | None = None

    if category and category.strip():
        cat_clean = category.strip()
        cat_lower = cat_clean.lower()
        if cat_lower in VALID_CATALOG_CATEGORIES:
            resolved_cat = VALID_CATALOG_CATEGORIES[cat_lower]
        else:
            mapped_kw = _CATEGORY_TO_VI_KEYWORD.get(cat_lower, cat_clean)
            if mapped_kw.lower() not in resolved_kw.lower():
                resolved_kw = f"{mapped_kw} {resolved_kw}".strip()

    if not resolved_kw:
        resolved_kw = resolved_cat or "trang phục thời trang"

    return resolved_kw, resolved_cat


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


def _normalize_sort(sort: str | None) -> str:
    if not sort:
        return "relevance"
    norm = sort.strip().lower()
    if norm in ("relevance", "price_asc", "price_desc", "rating_desc"):
        return norm
    return "relevance"


def _format_relaxed_note(relaxed_filters: list[str]) -> str:
    if not relaxed_filters:
        return ""
    label_map = {
        "sizes": "kích cỡ",
        "colors": "màu sắc",
        "maxPrice": "giá tối đa",
        "minPrice": "giá tối thiểu",
        "fit": "phom dáng",
        "pattern": "họa tiết",
        "occasion": "dịp mặc",
    }
    labels = [label_map.get(f, f) for f in relaxed_filters]
    return (
        f"[LƯU Ý: Không có sản phẩm khớp 100%, hệ thống đã nới lỏng tiêu chí: "
        f"{', '.join(labels)}]\n"
    )


def _format_facets_summary(facets: AssistantSearchFacets) -> str:
    parts: list[str] = []
    if facets.price_range.min is not None and facets.price_range.max is not None:
        parts.append(
            f"Giá: {facets.price_range.min:,.0f}đ–{facets.price_range.max:,.0f}đ"
        )
    if facets.available_colors:
        parts.append(f"Màu có sẵn: {', '.join(facets.available_colors[:6])}")
    if facets.available_sizes:
        parts.append(f"Size có sẵn: {', '.join(facets.available_sizes[:8])}")
    if not parts:
        return ""
    return f"[Thông tin kho hiện có — {' | '.join(parts)}]"


def _format_product_line(p: ProductSchema) -> str:
    """Format a single product line for LLM tool output."""
    in_stock_sizes = sorted({v.size for v in p.variants if v.size and v.stock > 0})
    size_str = ", ".join(in_stock_sizes) if in_stock_sizes else "Hết hàng"

    prices = [
        v.final_price if v.final_price is not None else v.selling_price
        for v in p.variants
    ]
    price_val = min(prices) if prices else 0.0
    price_str = f"{price_val:,.0f}đ" if price_val > 0 else "Liên hệ"

    material_str = f" | Chất liệu: {p.material}" if p.material else ""
    fit_str = f" | Form: {p.fit}" if p.fit else ""
    pattern_str = f" | Họa tiết: {p.pattern}" if p.pattern else ""
    gender_str = f" | Phù hợp: {p.target_gender.value}" if p.target_gender else ""
    return (
        f"- [{p.id}] {p.name} | Giá từ: {price_str} | Size: {size_str}"
        f"{material_str}{fit_str}{pattern_str}{gender_str}"
    )


def _build_balanced_branch_keywords(
    base_kw: str, resolved_cat: str | None, occasion: str | None
) -> dict[GenderTarget, str]:
    """Build branch-specific semantic keywords when target category is open."""
    if resolved_cat is None and occasion == "party":
        return {
            GenderTarget.FEMALE: f"{base_kw} đầm váy dự tiệc nữ",
            GenderTarget.MALE: f"{base_kw} áo sơ mi áo khoác quần âu nam dự tiệc",
            GenderTarget.UNISEX: f"{base_kw} áo sơ mi áo khoác unisex lịch sự",
        }
    return {
        GenderTarget.FEMALE: base_kw,
        GenderTarget.MALE: base_kw,
        GenderTarget.UNISEX: base_kw,
    }


async def _search_products_balanced(
    resolved_kw: str,
    resolved_cat: str | None,
    min_price: float | int | None,
    max_price: float | int | None,
    color: str | None,
    size: str | None,
    sort: str,
    cfg: BalancedGenderSearchConfig,
) -> str:
    """Fetch balanced FEMALE, MALE, and UNISEX recommendations via single BE call."""
    branch_keywords = _build_balanced_branch_keywords(
        resolved_kw, resolved_cat, cfg.occasion
    )
    branches: list[tuple[GenderTarget, int, str]] = [
        (GenderTarget.FEMALE, cfg.female_limit, "=== GỢI Ý CHO NỮ ==="),
        (GenderTarget.MALE, cfg.male_limit, "=== GỢI Ý CHO NAM ==="),
        (GenderTarget.UNISEX, cfg.unisex_limit, "=== GỢI Ý UNISEX ==="),
    ]
    min_p = float(min_price) if min_price is not None else None
    max_p = float(max_price) if max_price is not None else None

    request = AssistantSearchRequest(
        query=resolved_kw,
        filters=AssistantSearchFilters(
            categories=[resolved_cat] if resolved_cat else None,
            colors=[color.strip()] if color and color.strip() else None,
            sizes=[size.strip()] if size and size.strip() else None,
            min_price=min_p,
            max_price=max_p,
            in_stock_only=True,
        ),
        sort=sort,
        diversify_by_gender=True,
        gender_queries={g.value: kw for g, kw in branch_keywords.items()},
        gender_limits={g.value: lim for g, lim, _ in branches},
        allow_relaxation=True,
        limit=cfg.female_limit + cfg.male_limit + cfg.unisex_limit,
    )

    async with ShopClient() as client:
        resp = await client.search_products_assistant(request)

    seen_ids: set[int] = set()
    by_gender: dict[GenderTarget, list[ProductSchema]] = {
        GenderTarget.FEMALE: [],
        GenderTarget.MALE: [],
        GenderTarget.UNISEX: [],
    }
    for p in resp.items:
        if p.id in seen_ids:
            continue
        seen_ids.add(p.id)
        bucket_key = p.target_gender or GenderTarget.UNISEX
        by_gender.setdefault(bucket_key, []).append(p)

    bucket_products: list[tuple[GenderTarget, str, list[ProductSchema]]] = []
    all_selected: list[ProductSchema] = []
    gender_map: dict[int, GenderTarget] = {}

    for branch_gender, branch_limit, header in branches:
        chosen = by_gender.get(branch_gender, [])[:branch_limit]
        if chosen:
            bucket_products.append((branch_gender, header, chosen))
            for p in chosen:
                all_selected.append(p)
                gender_map[p.id] = p.target_gender or branch_gender

    if not all_selected:
        record_searched_products([])
        facets_txt = _format_facets_summary(resp.meta.facets)
        base_msg = "Không tìm thấy sản phẩm nào phù hợp với tiêu chí tìm kiếm."
        return f"{base_msg}\n{facets_txt}" if facets_txt else base_msg

    record_searched_products(
        [product_schema_to_recommended(p) for p in all_selected],
        gender_map=gender_map,
    )

    section_blocks = [
        "\n".join([header, *(_format_product_line(p) for p in chosen)])
        for _, header, chosen in bucket_products
    ]
    relaxed_note = _format_relaxed_note(resp.meta.relaxed_filters)
    return (
        f"{relaxed_note}"
        "[LƯU Ý: Khách chưa nêu giới tính nên hệ thống đã tìm cân bằng theo "
        "Nữ, Nam và Unisex]\n" + "\n\n".join(section_blocks)
    )


@tool(args_schema=SearchProductsInput)
async def search_products(
    keyword: str,
    category: str | None = None,
    min_price: float | int | None = None,
    max_price: float | int | None = None,
    target_gender: str | None = None,
    color: str | None = None,
    size: str | None = None,
    sort: str | None = None,
    limit: int = 5,
) -> str:
    """Tìm kiếm kết hợp ngữ nghĩa & bộ lọc sản phẩm theo mô tả, giá, màu, size."""
    capped_limit = max(1, min(5, int(limit or 5)))
    gender_enum = _normalize_gender(target_gender)
    resolved_kw, resolved_cat = _normalize_search_terms(keyword, category)
    normalized_sort = _normalize_sort(sort)

    balanced_cfg = _BALANCED_SEARCH_CTX.get()
    if balanced_cfg is not None and gender_enum is None:
        return await _search_products_balanced(
            resolved_kw=resolved_kw,
            resolved_cat=resolved_cat,
            min_price=min_price,
            max_price=max_price,
            color=color,
            size=size,
            sort=normalized_sort,
            cfg=balanced_cfg,
        )

    request = AssistantSearchRequest(
        query=resolved_kw,
        filters=AssistantSearchFilters(
            categories=[resolved_cat] if resolved_cat else None,
            target_gender=gender_enum,
            colors=[color.strip()] if color and color.strip() else None,
            sizes=[size.strip()] if size and size.strip() else None,
            min_price=float(min_price) if min_price is not None else None,
            max_price=float(max_price) if max_price is not None else None,
            in_stock_only=True,
        ),
        sort=normalized_sort,
        allow_relaxation=True,
        limit=capped_limit,
    )

    async with ShopClient() as client:
        resp = await client.search_products_assistant(request)

    products = resp.items[:capped_limit]
    if not products:
        record_searched_products([])
        facets_txt = _format_facets_summary(resp.meta.facets)
        base_msg = "Không tìm thấy sản phẩm nào phù hợp với tiêu chí tìm kiếm."
        return f"{base_msg}\n{facets_txt}" if facets_txt else base_msg

    # Cache structured products into the active per-turn context
    gender_map = {
        p.id: p.target_gender for p in products if p.target_gender is not None
    }
    record_searched_products(
        [product_schema_to_recommended(p) for p in products],
        gender_map=gender_map,
    )

    relaxed_note = _format_relaxed_note(resp.meta.relaxed_filters)
    body = "\n".join(_format_product_line(p) for p in products)
    if len(products) < 3:
        facets_txt = _format_facets_summary(resp.meta.facets)
        if facets_txt:
            return f"{relaxed_note}{body}\n{facets_txt}"
    return f"{relaxed_note}{body}"


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

    record_product(product_schema_to_recommended(product))

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
        *([f"Phom dáng: {product.fit}"] if product.fit else []),
        *([f"Họa tiết: {product.pattern}"] if product.pattern else []),
        *([f"Dịp mặc: {product.occasion}"] if product.occasion else []),
        *([f"Phom dáng: {product.fit}"] if product.fit else []),
        *([f"Họa tiết: {product.pattern}"] if product.pattern else []),
        *([f"Dịp mặc: {product.occasion}"] if product.occasion else []),
        (
            f"Giới tính phù hợp: "
            f"{product.target_gender.value if product.target_gender else 'N/A'}"
        ),
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
