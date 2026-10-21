import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from customer_shop_assistant.schemas.chat import UserProfile

logger = logging.getLogger(__name__)


class ClarifyPhase(StrEnum):
    """Phases for multi-turn clarification state machine."""

    INTERVIEW = "INTERVIEW"
    OPEN = "OPEN"


@dataclass
class SessionState:
    """Working memory state for an active conversation session."""

    session_id: str
    user_id: int | None = None
    messages: list[BaseMessage] = field(default_factory=list)
    active_profile: UserProfile | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    # Clarification state machine tracking
    clarification_mode: str | None = None
    clarify_phase: ClarifyPhase = ClarifyPhase.INTERVIEW
    gated_turns: int = 0

    def reset_clarification(self, mode: str | None) -> None:
        """Reset clarification state machine when switching mode."""
        self.clarification_mode = mode
        self.clarify_phase = ClarifyPhase.INTERVIEW
        self.gated_turns = 0

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = time.time()

    def update_active_profile(
        self, profile_update: UserProfile | dict[str, Any]
    ) -> UserProfile:
        """Update active profile in session memory (session override)."""
        curr = self.active_profile.model_dump() if self.active_profile else {}

        if isinstance(profile_update, UserProfile):
            update_dict = profile_update.model_dump(exclude_unset=True)
        else:
            update_dict = {k: v for k, v in profile_update.items() if v is not None}

        curr.update(update_dict)
        self.active_profile = UserProfile(**curr)
        self.touch()
        return self.active_profile


class SessionManager:
    """In-memory Short-term Session Store with sliding window and TTL eviction."""

    def __init__(self, ttl_seconds: int = 3600, default_max_turns: int = 8) -> None:
        self.ttl_seconds = ttl_seconds
        self.default_max_turns = default_max_turns
        self._sessions: dict[str, SessionState] = {}

    def get_or_create_session(
        self, session_id: str, user_id: int | None = None
    ) -> SessionState:
        """Retrieve existing session or instantiate a new one."""
        self.cleanup_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(
                session_id=session_id, user_id=user_id
            )
        session = self._sessions[session_id]
        if user_id is not None and session.user_id is None:
            session.user_id = user_id
        session.touch()
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        """Retrieve existing session if present, updating touch timestamp."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def get_messages(
        self, session_id: str, max_turns: int | None = None
    ) -> list[BaseMessage]:
        """Return sliding window of recent messages for prompt assembly."""
        session = self.get_or_create_session(session_id)
        turns = max_turns or self.default_max_turns
        max_messages = turns * 2
        return session.messages[-max_messages:]

    def add_user_message(self, session_id: str, content: str) -> None:
        """Append user inquiry to session message history."""
        session = self.get_or_create_session(session_id)
        session.messages.append(HumanMessage(content=content))
        session.touch()

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
    ) -> None:
        """Append assistant reply to session message history."""
        session = self.get_or_create_session(session_id)
        session.messages.append(AIMessage(content=content))
        session.touch()

    def update_active_profile(
        self,
        session_id: str,
        profile_update: UserProfile | dict[str, Any],
    ) -> UserProfile:
        """Update active profile in session memory (session override)."""
        session = self.get_or_create_session(session_id)
        return session.update_active_profile(profile_update)

    def cleanup_expired(self) -> int:
        """Evict sessions that have exceeded TTL."""
        now = time.time()
        expired_ids = [
            sid
            for sid, state in self._sessions.items()
            if now - state.last_accessed > self.ttl_seconds
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        if expired_ids:
            logger.info("Evicted %d expired chat sessions", len(expired_ids))
        return len(expired_ids)

    def clear(self) -> None:
        """Clear all stored sessions (useful for tests)."""
        self._sessions.clear()


# Global module singleton for in-memory session tracking
_global_session_manager: SessionManager | None = None


def get_session_manager(
    ttl_seconds: int = 3600, default_max_turns: int = 8
) -> SessionManager:
    """Return application-wide SessionManager instance."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManager(
            ttl_seconds=ttl_seconds, default_max_turns=default_max_turns
        )
    return _global_session_manager
