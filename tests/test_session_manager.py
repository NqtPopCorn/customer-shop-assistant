import time

import pytest

from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.session_manager import (
    SessionManager,
    SessionState,
    get_session_manager,
)


@pytest.fixture
def session_mgr() -> SessionManager:
    return SessionManager(ttl_seconds=3600, default_max_turns=3)


def test_session_creation(session_mgr: SessionManager) -> None:
    session = session_mgr.get_or_create_session("sess_1", user_id=42)
    assert isinstance(session, SessionState)
    assert session.session_id == "sess_1"
    assert session.user_id == 42
    assert session.messages == []
    assert session.active_profile is None


def test_sliding_window_turns(session_mgr: SessionManager) -> None:
    # default_max_turns = 3, so max 6 messages
    for i in range(10):
        session_mgr.add_user_message("sess_window", f"User msg {i}")
        session_mgr.add_assistant_message("sess_window", f"AI msg {i}")

    messages = session_mgr.get_messages("sess_window")
    assert len(messages) == 6  # 3 turns * 2
    assert messages[0].content == "User msg 7"
    assert messages[-1].content == "AI msg 9"


def test_update_active_profile(session_mgr: SessionManager) -> None:
    session_mgr.update_active_profile(
        "sess_prof",
        {"height_cm": 172.0, "weight_kg": 68.0, "gender": "MALE"},
    )
    session = session_mgr.get_session("sess_prof")
    assert session is not None
    assert session.active_profile is not None
    assert session.active_profile.height_cm == 172.0
    assert session.active_profile.weight_kg == 68.0
    assert session.active_profile.gender == GenderTarget.MALE

    # Override in session
    session_mgr.update_active_profile(
        "sess_prof",
        {"weight_kg": 70.0, "fit_preference": "LOOSE"},
    )
    assert session.active_profile.height_cm == 172.0
    assert session.active_profile.weight_kg == 70.0
    assert session.active_profile.fit_preference == "LOOSE"


def test_gender_normalization_in_profile(session_mgr: SessionManager) -> None:
    session_mgr.update_active_profile(
        "sess_gender",
        {"gender": "nam"},
    )
    session = session_mgr.get_session("sess_gender")
    assert session is not None
    assert session.active_profile is not None
    assert session.active_profile.gender == GenderTarget.MALE

    session_mgr.update_active_profile(
        "sess_gender",
        {"gender": "nữ"},
    )
    assert session.active_profile.gender == GenderTarget.FEMALE


def test_cleanup_expired() -> None:
    short_mgr = SessionManager(ttl_seconds=1, default_max_turns=3)
    short_mgr.get_or_create_session("sess_old")
    # artificially age the session
    session = short_mgr.get_session("sess_old")
    assert session is not None
    session.last_accessed = time.time() - 10

    evicted = short_mgr.cleanup_expired()
    assert evicted == 1
    assert short_mgr.get_session("sess_old") is None


def test_singleton_getter() -> None:
    mgr1 = get_session_manager()
    mgr2 = get_session_manager()
    assert mgr1 is mgr2
