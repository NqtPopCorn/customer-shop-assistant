from unittest.mock import AsyncMock, patch

import pytest

from customer_shop_assistant.tools.user_tool import update_customer_preference


@pytest.mark.anyio
async def test_update_customer_preference_empty() -> None:
    res = await update_customer_preference.ainvoke({"user_id": 1})
    assert "Không có thông tin sở thích mới" in res


@pytest.mark.anyio
async def test_update_customer_preference_success() -> None:
    with patch("customer_shop_assistant.tools.user_tool.ShopClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_instance.update_user_preferences.return_value = {
            "userId": 1,
            "heightCm": 175.0,
            "weightKg": 70.0,
        }
        mock_client_cls.return_value.__aenter__.return_value = mock_instance

        res = await update_customer_preference.ainvoke(
            {
                "user_id": 1,
                "height_cm": 175.0,
                "weight_kg": 70.0,
                "fit_preference": "regular",
            }
        )
        assert "Đã lưu thành công thông tin hồ sơ cho khách hàng #1" in res
        mock_instance.update_user_preferences.assert_called_once()
        call_kwargs = mock_instance.update_user_preferences.call_args.kwargs
        assert call_kwargs["user_id"] == 1
        assert call_kwargs["preferences"]["heightCm"] == 175.0
        assert call_kwargs["preferences"]["weightKg"] == 70.0


@pytest.mark.anyio
async def test_update_customer_preference_from_session() -> None:
    from customer_shop_assistant.schemas.chat import UserProfile
    from customer_shop_assistant.services.session_manager import (
        get_session_manager,
    )

    mgr = get_session_manager()
    session = mgr.get_or_create_session("sess_thank_you", user_id=99)
    session.active_profile = UserProfile(
        height_cm=168.0,
        weight_kg=58.0,
        fit_preference="LOOSE",
    )

    with patch("customer_shop_assistant.tools.user_tool.ShopClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_instance.update_user_preferences.return_value = {"userId": 99}
        mock_client_cls.return_value.__aenter__.return_value = mock_instance

        # Customer says thank you, tool called with only user_id and session_id
        res = await update_customer_preference.ainvoke(
            {"user_id": 99, "session_id": "sess_thank_you"}
        )
        assert "Đã lưu thành công thông tin hồ sơ cho khách hàng #99" in res
        call_kwargs = mock_instance.update_user_preferences.call_args.kwargs
        assert call_kwargs["user_id"] == 99
        assert call_kwargs["preferences"]["heightCm"] == 168.0
        assert call_kwargs["preferences"]["weightKg"] == 58.0
        assert call_kwargs["preferences"]["fitPreference"] == "LOOSE"
