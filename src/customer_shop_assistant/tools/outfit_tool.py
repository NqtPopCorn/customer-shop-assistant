"""LangChain tool for Knowledge-Graph-backed top + bottom outfit matching."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from langchain_core.tools import tool

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import OutfitItem
from customer_shop_assistant.schemas.shop import (
    GenderTarget,
    ProductFilterParams,
    ProductSchema,
)
from customer_shop_assistant.services.outfit_kg_repository import (
    Neo4jOutfitKGRepository,
    get_outfit_kg_repository,
)
from customer_shop_assistant.services.outfit_retrieval import (
    OutfitRetrievalAndReranker,
)
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)
from customer_shop_assistant.tools.catalog_tool import (
    record_outfits,
    record_product,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutfitDiversityConfig:
    """Per-turn cross-session exclusion & rotation settings for suggest_outfit."""

    exclude_product_ids: frozenset[int] = frozenset()
    exclude_pairs: frozenset[tuple[int, int]] = frozenset()
    rotation_seed: str | None = None


_OUTFIT_DIVERSITY_CTX: ContextVar[OutfitDiversityConfig | None] = ContextVar(
    "outfit_diversity_context", default=None
)


@contextmanager
def outfit_diversity_context(
    exclude_product_ids: set[int] | list[int] | None = None,
    exclude_pairs: set[tuple[int, int]] | list[tuple[int, int]] | None = None,
    rotation_seed: str | None = None,
) -> Iterator[OutfitDiversityConfig]:
    """Activate cross-turn outfit exclusion and rotation for the current turn."""
    cfg = OutfitDiversityConfig(
        exclude_product_ids=frozenset(int(x) for x in (exclude_product_ids or ())),
        exclude_pairs=frozenset((int(a), int(b)) for a, b in (exclude_pairs or ())),
        rotation_seed=rotation_seed,
    )
    token = _OUTFIT_DIVERSITY_CTX.set(cfg)
    try:
        yield cfg
    finally:
        try:
            _OUTFIT_DIVERSITY_CTX.reset(token)
        except ValueError:
            _OUTFIT_DIVERSITY_CTX.set(None)


def _parse_gender(gender: str | None) -> GenderTarget | None:
    if not gender:
        return None
    g_norm = gender.strip().upper()
    if g_norm in GenderTarget.__members__:
        return GenderTarget[g_norm]
    if g_norm in ("NAM", "MEN", "MAN", "BOY"):
        return GenderTarget.MALE
    if g_norm in ("NỮ", "NU", "WOMEN", "WOMAN", "GIRL"):
        return GenderTarget.FEMALE
    return None


def _format_item_line(label: str, item: OutfitItem) -> str:
    stock_note = "" if item.in_stock else " (màu này đang hết hàng)"
    return (
        f"  - {label}: [{item.product_id}] {item.name} "
        f"| Màu: {item.color} | Giá: {item.price:,.0f}đ{stock_note}"
    )


@tool
async def suggest_outfit(
    product_id: int | None = None,
    color: str | None = None,
    top_category: str | None = None,
    bottom_category: str | None = None,
    top_color: str | None = None,
    bottom_color: str | None = None,
    occasion: str | None = None,
    style: str | None = None,
    gender: str | None = None,
    user_id: int | None = None,
    favorite_colors: list[str] | None = None,
    disliked_colors: list[str] | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    limit: int = 3,
) -> str:
    """Gợi ý phối đồ hoàn chỉnh gồm Áo và Quần/Váy bằng Knowledge Graph thời trang.

    Gọi trực tiếp tool này cho mọi câu hỏi phối đồ hoặc tìm set/bộ/combo áo + quần/váy
    (KHÔNG cần gọi `search_products` trước khi chưa có product_id):
    - Nếu đã có `product_id` cụ thể, truyền `product_id` (và `color`,
      `top_category`/`bottom_category` nếu khách chỉ định món phối cùng).
    - Nếu chưa có `product_id` (ví dụ 'Áo polo xanh navy phối quần gì?', 'Set áo thun
      và quần jeans'), truyền `top_category`, `bottom_category`, `top_color`,
      `bottom_color`, `occasion`, `style`, `gender` để gợi ý các bộ combo hoàn chỉnh.
    """
    settings = get_settings()
    fav_set: set[str] = set(favorite_colors or [])
    disliked_set: set[str] = set(disliked_colors or [])
    eff_occasion = occasion

    live_products: list[ProductSchema] | None = None
    async with ShopClient(settings=settings) as client:
        if user_id is not None:
            try:
                prefs = await client.get_user_preferences(user_id)
                if isinstance(prefs, dict):
                    fav_set.update(str(c) for c in (prefs.get("favoriteColors") or []))
                    disliked_set.update(
                        str(c) for c in (prefs.get("dislikedColors") or [])
                    )
                    if not eff_occasion:
                        freq_occ = prefs.get("frequentOccasions") or []
                        if freq_occ:
                            eff_occasion = str(freq_occ[0])
            except Exception as exc:
                logger.warning(
                    "Could not fetch preferences for user %s: %s", user_id, exc
                )

        try:
            fetched = await client.get_products(ProductFilterParams(limit=2000))
            if fetched:
                live_products = fetched
        except Exception as exc:
            logger.debug(
                "Main BE products unavailable (%s); falling back to snapshot.",
                exc,
            )

    parsed_gender = _parse_gender(gender)

    kg_repo = get_outfit_kg_repository(settings)
    div_cfg = _OUTFIT_DIVERSITY_CTX.get()
    try:
        snap_idx = None if live_products else get_snapshot_catalog_index(settings)
        reranker = OutfitRetrievalAndReranker(kg_repo=kg_repo, snapshot_index=snap_idx)
        ret = reranker.retrieve_outfits(
            live_products=live_products,
            product_id=product_id,
            target_gender=parsed_gender,
            occasion=eff_occasion,
            style=style,
            top_category=top_category,
            bottom_category=bottom_category,
            color=color,
            top_color=top_color,
            bottom_color=bottom_color,
            favorite_colors=sorted(fav_set),
            disliked_colors=sorted(disliked_set),
            min_price=min_price,
            max_price=max_price,
            limit=limit,
            exclude_product_ids=(
                set(div_cfg.exclude_product_ids) if div_cfg else None
            ),
            exclude_pairs=set(div_cfg.exclude_pairs) if div_cfg else None,
            rotation_seed=div_cfg.rotation_seed if div_cfg else None,
        )
    finally:
        if isinstance(kg_repo, Neo4jOutfitKGRepository):
            kg_repo.close()

    if ret.error == "out_of_scope":
        record_outfits([])
        return (
            "Sản phẩm này không nằm trong phạm vi phối áo + quần/váy "
            "(đầm, váy liền, phụ kiện chưa hỗ trợ)."
        )
    if ret.error == "not_found":
        record_outfits([])
        return f"Không tìm thấy sản phẩm mã [{product_id}] trong danh mục hiện tại."
    if not ret.outfits:
        record_outfits([])
        return (
            "Hiện chưa tìm thấy bộ phối đồ phù hợp với tiêu chí này "
            "trong danh mục của cửa hàng."
        )

    record_outfits(ret.outfits)
    for outfit in ret.outfits:
        for item in (outfit.top, outfit.bottom):
            rec_prod = ret.products_by_id.get(item.product_id)
            if rec_prod is not None:
                record_product(rec_prod)

    lines: list[str] = []
    for note in ret.notes:
        lines.append(f"Lưu ý: {note}")
    for idx, outfit in enumerate(ret.outfits, start=1):
        lines.append(f"Bộ {idx} (Điểm phù hợp: {outfit.score:.2f}):")
        lines.append(_format_item_line("Áo", outfit.top))
        lines.append(_format_item_line("Quần/Váy", outfit.bottom))
        if outfit.reasons:
            lines.append(f"  - Lý do: {'; '.join(outfit.reasons)}")
        if outfit.applied_rule_ids:
            lines.append(f"  - Quy tắc KG: {', '.join(outfit.applied_rule_ids)}")
    return "\n".join(lines)
