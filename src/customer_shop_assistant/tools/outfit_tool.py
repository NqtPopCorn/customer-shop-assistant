"""LangChain tool for Knowledge-Graph-backed top + bottom outfit matching."""

import logging

from langchain_core.tools import tool

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import OutfitItem
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.outfit_graph import (
    get_outfit_graph,
    resolve_outfit_category,
)
from customer_shop_assistant.services.outfit_rules import normalize_occasion
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.tools.catalog_tool import (
    product_schema_to_recommended,
    record_outfits,
    record_product,
)

logger = logging.getLogger(__name__)


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
    gender: str | None = None,
    user_id: int | None = None,
    favorite_colors: list[str] | None = None,
    disliked_colors: list[str] | None = None,
    limit: int = 3,
) -> str:
    """Gợi ý phối đồ hoàn chỉnh gồm Áo và Quần/Váy bằng Knowledge Graph thời trang.

    Gọi trực tiếp tool này cho mọi câu hỏi phối đồ hoặc tìm set/bộ/combo áo + quần/váy
    (KHÔNG cần gọi `search_products` trước khi chưa có product_id):
    - Nếu đã có `product_id` cụ thể, truyền `product_id` (và `color`,
      `top_category`/`bottom_category` nếu khách chỉ định món phối cùng).
    - Nếu chưa có `product_id` (ví dụ 'Áo polo xanh navy phối quần gì?', 'Set áo thun
      và quần jeans'), truyền `top_category`, `bottom_category`, `top_color`,
      `bottom_color`, `occasion`, `gender` để gợi ý các bộ combo hoàn chỉnh.
    """
    settings = get_settings()
    fav_set: set[str] = set(favorite_colors or [])
    disliked_set: set[str] = set(disliked_colors or [])
    eff_occasion = occasion

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

        graph = await get_outfit_graph(client=client)

    if not graph.nodes:
        record_outfits([])
        return (
            "Hiện chưa lấy được dữ liệu sản phẩm để phối đồ. "
            "Bạn vui lòng thử lại sau giây lát nhé."
        )

    occasion_range = normalize_occasion(eff_occasion, graph.rules)
    parsed_gender = _parse_gender(gender)

    if product_id is not None:
        anchor_node = graph.nodes.get(product_id)
        if anchor_node is not None and anchor_node.attrs.role == "BOTTOM":
            anchor_col = color or bottom_color
            target_cat = top_category
            target_col = top_color
        else:
            anchor_col = color or top_color
            target_cat = bottom_category
            target_col = bottom_color
        result = graph.suggest_for_anchor(
            anchor_id=product_id,
            colour=anchor_col,
            occasion_range=occasion_range,
            gender_hint=parsed_gender,
            favorite=fav_set,
            disliked=disliked_set,
            limit=limit,
            target_category=target_cat,
            target_color=target_col,
        )
    else:
        eff_top_color = top_color
        eff_bottom_color = bottom_color
        if color and not eff_top_color and not eff_bottom_color:
            _, t_role = resolve_outfit_category(top_category, graph.rules)
            _, b_role = resolve_outfit_category(bottom_category, graph.rules)
            if t_role in ("TOP", "JACKET") and b_role is None:
                eff_top_color = color
            elif b_role == "BOTTOM" and t_role is None:
                eff_bottom_color = color
            elif t_role == "BOTTOM" and b_role is None:
                eff_bottom_color = color
            elif b_role in ("TOP", "JACKET") and t_role is None:
                eff_top_color = color
            elif t_role is None and b_role is None:
                fav_set.add(color)
            else:
                eff_top_color = color

        result = graph.suggest_open(
            occasion_range=occasion_range,
            gender=parsed_gender,
            favorite=fav_set,
            disliked=disliked_set,
            limit=limit,
            top_category=top_category,
            bottom_category=bottom_category,
            top_color=eff_top_color,
            bottom_color=eff_bottom_color,
        )

    if result.error == "out_of_scope":
        record_outfits([])
        return (
            "Sản phẩm này không nằm trong phạm vi phối áo + quần/váy "
            "(đầm, váy liền, phụ kiện chưa hỗ trợ)."
        )

    if result.error == "not_found":
        record_outfits([])
        return f"Không tìm thấy sản phẩm mã [{product_id}] trong danh mục hiện tại."

    if not result.outfits:
        record_outfits([])
        return (
            "Hiện chưa tìm thấy bộ phối đồ phù hợp với tiêu chí này "
            "trong danh mục của cửa hàng."
        )

    record_outfits(result.outfits)
    for outfit in result.outfits:
        for item in (outfit.top, outfit.bottom):
            node = graph.nodes.get(item.product_id)
            if node is not None:
                record_product(product_schema_to_recommended(node.product))

    lines: list[str] = []
    for note in result.notes:
        lines.append(f"Lưu ý: {note}")

    for idx, outfit in enumerate(result.outfits, start=1):
        lines.append(f"Bộ {idx} (Điểm phù hợp: {outfit.score:.2f}):")
        lines.append(_format_item_line("Áo", outfit.top))
        lines.append(_format_item_line("Quần/Váy", outfit.bottom))
        if outfit.reasons:
            lines.append(f"  - Lý do: {'; '.join(outfit.reasons)}")

    return "\n".join(lines)
