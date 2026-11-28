import logging
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from customer_shop_assistant.schemas.outfit_kg import ColorScoreBreakdown
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


class ClarificationQuestion(BaseModel):
    """Single question item inside an inline clarification form card."""

    id: str = Field(
        ...,
        description=(
            "Unique question identifier (e.g. 'occasion', 'style', 'budget', 'fit')"
        ),
    )
    question: str = Field(
        ...,
        description=(
            "Numbered question title in Vietnamese (e.g. '1. Bạn tìm đồ cho dịp nào?')"
        ),
    )
    type: Literal["single", "multiple"] = Field(
        default="single",
        description=(
            "'single' for radio selection, 'multiple' for checkbox multi-select"
        ),
    )
    options: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 concise answer options written from the customer's perspective"
        ),
    )
    allow_custom: bool = Field(
        default=False,
        description=(
            "Whether to show an inline input for custom text or body measurements"
        ),
    )
    custom_placeholder: str | None = Field(
        default=None,
        description=(
            "Placeholder for custom input (e.g. 'Nhập chiều cao, cân nặng: 1m70, 62kg')"
        ),
    )


class ClarificationForm(BaseModel):
    """Interactive multi-question clarification card rendered inline in chat."""

    title: str = Field(
        default="Xác định thêm nhu cầu trang phục của bạn",
        description="Header title displayed above the clarification card",
    )
    submit_label: str = Field(
        default="Xác nhận và xem gợi ý",
        description=(
            "Label for the primary confirmation button at the bottom of the card"
        ),
    )
    questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        description="1-4 clarification questions for the customer to select",
    )


class AgentReply(BaseModel):
    """Structured output returned by the LLM at the final turn of conversation."""

    reply: str = Field(
        ...,
        description=(
            "Nội dung câu trả lời hoàn chỉnh, lịch sự bằng tiếng Việt "
            "gửi cho khách hàng. Nếu cần hỏi làm rõ, chỉ viết 1-2 câu dẫn ngắn gọn."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True nếu cần khách hàng làm rõ hoặc cung cấp thêm số đo, dịp mặc, "
            "gu thời trang trước khi đưa ra kết luận."
        ),
    )
    clarification_form: ClarificationForm | None = Field(
        default=None,
        description="Khung câu hỏi trắc nghiệm tương tác khi needs_clarification=True",
    )
    product_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Danh sách ID các sản phẩm thực sự được gợi ý trong reply "
            "(ví dụ ['1', '2']). Trả về danh sách rỗng nếu không gợi ý sản phẩm."
        ),
    )
    decision_reason: str | None = Field(
        default=None,
        description=(
            "Lý do ngắn gọn giải thích vì sao chọn hỏi làm rõ hay trả lời ngay "
            "trong chế độ CONDITIONAL_CLARIFICATION."
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
    variant_id: int | None = Field(default=None, description="Selected SKU variant ID")
    color: str | None = Field(default=None, description="Selected variant color label")
    size: str | None = Field(default=None, description="Selected variant size label")


class OutfitItem(BaseModel):
    """Single garment item (top or bottom) within a recommended outfit."""

    product_id: int = Field(..., description="Catalog product ID")
    name: str = Field(..., description="Product commercial name")
    slug: str | None = Field(default=None, description="Product URL slug")
    category: str | None = Field(default=None, description="Category name")
    color: str = Field(..., description="Selected variant colour")
    variant_id: int | None = Field(
        default=None, description="Representative variant ID for selected color"
    )
    size: str | None = Field(
        default=None, description="Representative variant size label"
    )
    in_stock: bool = Field(
        ..., description="Whether at least one variant of this colour has stock > 0"
    )
    image_url: str | None = Field(default=None, description="Item image URL")
    price: float = Field(..., ge=0.0, description="Lowest selling/final price")


class OutfitSuggestion(BaseModel):
    """Paired top + bottom outfit recommendation produced by the KG."""

    top: OutfitItem = Field(..., description="Recommended top item")
    bottom: OutfitItem = Field(..., description="Recommended bottom item")
    score: float = Field(..., ge=0.0, le=1.0, description="KG compatibility score")
    blueprint_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Category-level blueprint score"
    )
    color_score_breakdown: ColorScoreBreakdown | None = Field(
        default=None, description="Transparent Color-aware scoring debug breakdown"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable matching explanations"
    )
    applied_rule_ids: list[str] = Field(
        default_factory=list,
        description="KG rule IDs supporting this outfit recommendation",
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


class LangChainUsageMetadata(BaseModel):
    """Token usage metrics following LangChain's UsageMetadata specification."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, Any] | None = None
    output_token_details: dict[str, Any] | None = None


class LangChainToolCall(BaseModel):
    """Standard LangChain ToolCall structure."""

    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    type: Literal["tool_call"] = "tool_call"


class LangChainMessageDump(BaseModel):
    """Serialized representation of a LangChain BaseMessage for debug view."""

    type: str = Field(..., description="human | ai | system | tool")
    content: str | list[Any] = ""
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    usage_metadata: dict[str, Any] | None = None
    response_metadata: dict[str, Any] | None = None


class SystemPromptSectionDebug(BaseModel):
    """Single constituent section of a SystemMessage sent to the LLM."""

    id: str = Field(
        ..., description="Section identifier: base | profile | policy | slot_summary"
    )
    name: str = Field(
        ..., description="Formula token name (e.g. SYSTEM_PROMPT, PROMPT_A)"
    )
    included: bool = Field(default=True, description="Whether section was included")
    char_count: int = Field(default=0, ge=0, description="Character length of section")
    content: str = Field(default="", description="Plain-text content of section")


class SystemPromptDebug(BaseModel):
    """Structured composition breakdown of a SystemMessage for debugging."""

    version: str = Field(default="v1.0", description="System prompt version")
    formula: str = Field(
        ..., description="Composition formula (e.g. SYSTEM_PROMPT + PROMPT_A)"
    )
    total_chars: int = Field(default=0, ge=0, description="Total SystemMessage length")
    sections: list[SystemPromptSectionDebug] = Field(
        default_factory=list, description="Ordered constituent prompt sections"
    )


class LangChainRunTrace(BaseModel):
    """Trace record for a single LangChain execution run (chat model or tool)."""

    run_id: str
    parent_run_id: str | None = None
    name: str
    run_type: Literal["chat_model", "llm", "tool", "chain"]
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    system_prompt: SystemPromptDebug | None = None
    usage_metadata: LangChainUsageMetadata | dict[str, Any] | None = None
    latency_ms: int = 0
    error: str | None = None


class StateSnapshotDebug(BaseModel):
    """Debug snapshot of conversation and agent state machine."""

    clarify_phase: str
    next_phase: str | None = Field(
        default=None,
        description="Clarify phase transitioned to after this turn",
    )
    gated_turns: int
    clarification_turns: int = 0
    topic_switched: bool = False
    tools_available: list[str]
    gate_search: bool
    active_profile: dict[str, Any] | None = None
    detected_slots: dict[str, Any] = Field(default_factory=dict)
    context_messages_count: int = 0


class TurnDebugMeta(BaseModel):
    """Comprehensive observability payload attached to ChatResponse."""

    mode: str
    phase: str
    gated_turns: int
    clarification_turns: int = 0
    decision_source: str | None = None
    decision_reason: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    impact_signals: dict[str, Any] = Field(default_factory=dict)
    evidence_rule_ids: list[str] = Field(default_factory=list)
    asked_slots: list[str] = Field(default_factory=list)
    tools_available: list[str]
    latency_ms: int
    usage_metadata: LangChainUsageMetadata | dict[str, Any] | None = None
    runs: list[LangChainRunTrace] = Field(default_factory=list)
    state_snapshot: StateSnapshotDebug | None = None


class ChatResponse(BaseModel):
    """Customer shop assistant reply payload."""

    reply: str = Field(..., description="Assistant response text")
    session_id: str = Field(..., description="Session identifier")
    intent: str | None = Field(default=None, description="Detected customer intent")
    clarification_asked: bool = Field(
        default=False, description="Whether assistant asked a clarifying question"
    )
    clarification_form: ClarificationForm | None = Field(
        default=None, description="Structured interactive clarification card"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Names of tools invoked during turn"
    )
    products: list[ChatRecommendedProduct] = Field(
        default_factory=list, description="Recommended products matching inquiry"
    )
    outfits: list[OutfitSuggestion] = Field(
        default_factory=list,
        description="Structured top-bottom outfit recommendations from KG",
    )
    debug_meta: TurnDebugMeta | dict[str, Any] | None = Field(
        default=None, description="Debug turn telemetry (mode, phase, tools, traces)"
    )
