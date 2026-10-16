from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.shop import GenderTarget


class ClarificationMode(StrEnum):
    """Clarification modes controlling how the assistant probes for missing info."""

    IMMEDIATE_ANSWER = "IMMEDIATE_ANSWER"
    MINIMAL_CLARIFICATION = "MINIMAL_CLARIFICATION"
    ALWAYS_CLARIFY = "ALWAYS_CLARIFY"


class UserProfile(BaseModel):
    """Customer body measurements and fit preferences."""

    height_cm: float | None = Field(
        default=None, ge=50, le=250, description="Height in cm"
    )
    weight_kg: float | None = Field(
        default=None, ge=20, le=200, description="Weight in kg"
    )
    waist_cm: float | None = Field(
        default=None, ge=30, le=150, description="Waist in cm"
    )
    chest_cm: float | None = Field(
        default=None, ge=30, le=160, description="Chest in cm"
    )
    hip_cm: float | None = Field(default=None, ge=30, le=160, description="Hip in cm")
    gender: GenderTarget | None = Field(default=None, description="Target gender")
    fit_preference: str | None = Field(
        default=None, description="Fit preference: tight, regular, loose/oversize"
    )


class ChatMessage(BaseModel):
    """Single turn message in chat history."""

    role: Literal["user", "assistant", "system"] = Field(
        default="user", description="Message author"
    )
    content: str = Field(..., min_length=1, description="Message text content")


class ChatRequest(BaseModel):
    """Customer chat input payload."""

    message: str = Field(..., min_length=1, description="Customer inquiry message")
    session_id: str | None = Field(
        default=None, description="Session identifier for multi-turn chat"
    )
    user_profile: UserProfile | None = Field(
        default=None, description="Known customer body profile"
    )
    clarification_mode: ClarificationMode = Field(
        default=ClarificationMode.MINIMAL_CLARIFICATION,
        description="Clarification strategy",
    )
    history: list[ChatMessage] = Field(
        default_factory=list, description="Prior conversation messages"
    )


class ChatResponse(BaseModel):
    """Customer shop assistant reply payload."""

    reply: str = Field(..., description="Assistant response text")
    session_id: str = Field(..., description="Session identifier")
    intent: str | None = Field(default=None, description="Detected customer intent")
    clarification_asked: bool = Field(
        default=False, description="Whether assistant asked a clarifying question"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Names of tools invoked during turn"
    )
