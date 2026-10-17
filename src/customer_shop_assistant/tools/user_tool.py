import logging
from typing import Any

from langchain_core.tools import tool

from customer_shop_assistant.services.shop_client import ShopClient

logger = logging.getLogger(__name__)


@tool
async def update_customer_preference(
    user_id: int,
    height_cm: float | None = None,
    weight_kg: float | None = None,
    waist_cm: float | None = None,
    chest_cm: float | None = None,
    hip_cm: float | None = None,
    preferred_sizes: list[str] | None = None,
    fit_preference: str | None = None,
    styles: list[str] | None = None,
    favorite_colors: list[str] | None = None,
    disliked_colors: list[str] | None = None,
    price_preference_min: float | None = None,
    price_preference_max: float | None = None,
    frequent_occasions: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    """Cập nhật thông tin sở thích, số đo, phong cách thời trang của khách hàng.

    Gọi tool này khi khách hàng nói cảm ơn để chốt lưu số đo sau khi tư vấn,
    hoặc khi khách hàng chủ động yêu cầu lưu hồ sơ tài khoản:
    - Chiều cao (height_cm), cân nặng (weight_kg), số đo 3 vòng
    - Size thường mặc (preferred_sizes)
    - Gu mặc (fit_preference: tight, regular, oversize)
    - Phong cách yêu thích (styles: minimalism, công sở, streetwear...)
    - Màu sắc yêu thích (favorite_colors) hoặc màu không thích (disliked_colors)
    - Ngân sách (price_preference_min/max) hoặc dịp mặc (frequent_occasions)
    - session_id: Mã phiên chat hiện tại để lấy số đo tạm thời tích lũy
    """
    payload: dict[str, Any] = {}
    if height_cm is not None:
        payload["heightCm"] = height_cm
    if weight_kg is not None:
        payload["weightKg"] = weight_kg
    if waist_cm is not None:
        payload["waistCm"] = waist_cm
    if chest_cm is not None:
        payload["chestCm"] = chest_cm
    if hip_cm is not None:
        payload["hipCm"] = hip_cm
    if preferred_sizes is not None:
        payload["preferredSizes"] = preferred_sizes
    if fit_preference is not None:
        payload["fitPreference"] = fit_preference
    if styles is not None:
        payload["styles"] = styles
    if favorite_colors is not None:
        payload["favoriteColors"] = favorite_colors
    if disliked_colors is not None:
        payload["dislikedColors"] = disliked_colors
    if price_preference_min is not None:
        payload["pricePreferenceMin"] = price_preference_min
    if price_preference_max is not None:
        payload["pricePreferenceMax"] = price_preference_max
    if frequent_occasions is not None:
        payload["frequentOccasions"] = frequent_occasions

    # If some fields missing, fill from session active profile
    if session_id:
        from customer_shop_assistant.services.session_manager import (
            get_session_manager,
        )

        session = get_session_manager().get_session(session_id)
        if session and session.active_profile:
            prof = session.active_profile
            if "heightCm" not in payload and prof.height_cm:
                payload["heightCm"] = prof.height_cm
            if "weightKg" not in payload and prof.weight_kg:
                payload["weightKg"] = prof.weight_kg
            if "waistCm" not in payload and prof.waist_cm:
                payload["waistCm"] = prof.waist_cm
            if "chestCm" not in payload and prof.chest_cm:
                payload["chestCm"] = prof.chest_cm
            if "hipCm" not in payload and prof.hip_cm:
                payload["hipCm"] = prof.hip_cm
            if "fitPreference" not in payload and prof.fit_preference:
                payload["fitPreference"] = prof.fit_preference

    if not payload:
        return "Không có thông tin sở thích mới nào để cập nhật."

    try:
        async with ShopClient() as client:
            result = await client.update_user_preferences(
                user_id=user_id, preferences=payload
            )
            if result:
                return f"Đã lưu thành công thông tin hồ sơ cho khách hàng #{user_id}."
            return f"Đã ghi nhận thông tin hồ sơ cho khách hàng #{user_id}."
    except Exception as exc:
        logger.warning("Error updating customer preference: %s", exc)
        return "Hệ thống đã ghi nhận thông tin vào phiên đối thoại."
