import logging
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from customer_shop_assistant.schemas.shop import GenderTarget

logger = logging.getLogger(__name__)


class ClarificationMode(StrEnum):
    """Clarification modes controlling how the assistant probes for missing info."""

    IMMEDIATE_ANSWER = "IMMEDIATE_ANSWER"
    ALWAYS_CLARIFY = "ALWAYS_CLARIFY"
    CONDITIONAL_CLARIFICATION = "CONDITIONAL_CLARIFICATION"
    MINIMAL_CLARIFICATION = "MINIMAL_CLARIFICATION"  # Deprecated alias


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

    @field_validator("gender", mode="before")
    @classmethod
    def parse_gender(cls, v: Any) -> GenderTarget | None:
        if v is None or isinstance(v, GenderTarget):
            return v
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in GenderTarget.__members__:
                return GenderTarget[v_clean]
            if v_clean in ("NAM", "MEN", "MAN", "BOY"):
                return GenderTarget.MALE
            if v_clean in ("NỮ", "NU", "WOMEN", "WOMAN", "GIRL"):
                return GenderTarget.FEMALE
            if v_clean in ("UNISEX", "ALL"):
                return GenderTarget.UNISEX
        return None


class AgentReply(BaseModel):
    """Structured output returned by the LLM at the final turn of conversation."""

    reply: str = Field(
        ...,
        description=(
            "Nội dung câu trả lời hoàn chỉnh, lịch sự bằng tiếng Việt "
            "gửi cho khách hàng."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True nếu cần khách hàng làm rõ hoặc cung cấp thêm số đo, dịp mặc, "
            "gu thời trang trước khi đưa ra kết luận."
        ),
    )
    product_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Danh sách ID các sản phẩm thực sự được gợi ý trong reply "
            "(ví dụ ['1', '2']). Trả về danh sách rỗng nếu không gợi ý sản phẩm."
        ),
    )
    quick_replies: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 gợi ý câu trả lời nhanh / câu hỏi tiếp theo phù hợp với ngữ cảnh "
            "hiện tại. Không bịa số đo hay cam kết ngoài kết quả tool."
        ),
    )


class ChatRecommendedProduct(BaseModel):
    """Structured product preview for chat carousel rendering."""

    id: int
    name: str
    slug: str | None = Field(default=None, description="Product URL slug")
    price: float = Field(..., description="Current selling price or discounted price")
    original_price: float | None = Field(
        default=None, description="Original list price before discount"
    )
    discount_percent: int | None = Field(
        default=None, description="Discount percentage"
    )
    image_url: str | None = Field(default=None, description="Primary product image URL")
    category: str | None = Field(default=None, description="Category name")
    brand: str | None = Field(default=None, description="Brand name")


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
        default=None,
        validation_alias=AliasChoices(
            "session_id", "conversation_id", "conversationId"
        ),
        description="Session identifier for multi-turn chat",
    )
    user_profile: UserProfile | None = Field(
        default=None, description="Known customer body profile"
    )
    clarification_mode: ClarificationMode | None = Field(
        default=None,
        description="Clarification strategy (defaults to server config if None)",
    )
    history: list[ChatMessage] = Field(
        default_factory=list, description="Prior conversation messages"
    )

    @field_validator("clarification_mode", mode="before")
    @classmethod
    def validate_clarification_mode(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in ("MINIMAL_CLARIFICATION", "MINIMAL"):
                logger.warning(
                    "ClarificationMode 'MINIMAL_CLARIFICATION' is deprecated; "
                    "mapping to 'CONDITIONAL_CLARIFICATION'."
                )
                return ClarificationMode.CONDITIONAL_CLARIFICATION
            if v_clean in ClarificationMode.__members__:
                return ClarificationMode[v_clean]
        return v


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
    products: list[ChatRecommendedProduct] = Field(
        default_factory=list, description="Recommended products matching inquiry"
    )
    quick_replies: list[str] = Field(
        default_factory=list, description="Contextual quick reply suggestions"
    )
    debug_meta: dict[str, Any] | None = Field(
        default=None, description="Debug turn telemetry (mode, phase, tools)"
    )
