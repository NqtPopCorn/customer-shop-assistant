from typing import Literal

from langchain_core.tools import tool

from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.size_recommender import SizeRecommender


@tool
async def recommend_size(
    category: Literal["tops", "bottoms", "dresses"] = "tops",
    height_cm: float | None = None,
    weight_kg: float | None = None,
    waist_cm: float | None = None,
    chest_cm: float | None = None,
    hip_cm: float | None = None,
    gender: str = "unisex",
    fit_preference: str = "regular",
    product_id: int | None = None,
) -> str:
    """Tư vấn size chuẩn xác cho quần áo dựa trên chiều cao, cân nặng,
    số đo 3 vòng và sở thích mặc ôm/rộng.
    """
    target_gender = GenderTarget.UNISEX
    if gender:
        g_norm = gender.strip().upper()
        if g_norm in GenderTarget.__members__:
            target_gender = GenderTarget[g_norm]
        elif g_norm in ("NAM", "MEN", "MAN"):
            target_gender = GenderTarget.MALE
        elif g_norm in ("NỮ", "NU", "WOMEN", "WOMAN"):
            target_gender = GenderTarget.FEMALE

    recommender = SizeRecommender()
    rec = await recommender.recommend_size(
        height_cm=height_cm,
        weight_kg=weight_kg,
        waist_cm=waist_cm,
        chest_cm=chest_cm,
        hip_cm=hip_cm,
        category=category,
        gender=target_gender,
        fit_preference=fit_preference,
        product_id=product_id,
    )

    if rec.status == "insufficient_data":
        missing_str = ", ".join(rec.missing_fields)
        return (
            f"Chưa đủ thông số để tư vấn size chuẩn xác. "
            f"Vui lòng hỏi lại khách hàng các thông tin còn thiếu: {missing_str}."
        )

    if rec.status == "no_size_chart":
        return f"Hiện tại cửa hàng chưa có bảng size chuẩn cho danh mục '{category}'."

    alt_str = (
        f" (hoặc chọn {rec.secondary_size} nếu muốn mặc rộng hơn)"
        if rec.secondary_size
        else ""
    )
    lines = [
        f"Size đề xuất: {rec.primary_size}{alt_str} [Độ tin cậy: {rec.confidence}]",
        f"Giải thích: {rec.reasoning}",
    ]
    if rec.review_insight:
        lines.append(f"Ghi chú form từ khách mua: {rec.review_insight}")

    return "\n".join(lines)
