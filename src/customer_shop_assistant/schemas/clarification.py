"""Shared clarification strategy schemas and contracts (Spec v1.1 & v1.2 ready)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SlotStatus = Literal[
    "PROVIDED",
    "RESOLVED_FROM_CONTEXT",
    "INFERRED",
    "UNKNOWN",
    "AMBIGUOUS",
    "CONFLICTING",
    "DECLINED",
    "NOT_APPLICABLE",
]

SlotConfidence = Literal["high", "low"]

DecisionSource = Literal[
    "mode_a_immediate",
    "non_shopping_bypass",
    "mode_b_required_check",
    "mode_c_hard_ask",
    "mode_c_hard_answer",
    "mode_c_counterfactual",
    "mode_c_llm",
    "fallback",
]


class ClarificationDecision(BaseModel):
    """Policy decision returned by ClarificationPolicy before form/tool routing."""

    action: Literal["ASK", "PROCEED", "GRAY_ZONE"] = Field(
        ..., description="Routing action for the current turn"
    )
    target_slots: list[str] = Field(
        default_factory=list,
        description="Slots selected for clarification (<=3 in Mode B, <=2 in Mode C)",
    )
    decision_source: DecisionSource = Field(
        ..., description="Deterministic or LLM source that produced this decision"
    )
    decision_reason: str = Field(
        default="",
        description=(
            "Concise rationale explaining why clarification was asked or skipped"
        ),
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Structured reason codes for research auditability",
    )
    impact_signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional counterfactual impact signals (used by Spec v1.2)",
    )
    evidence_rule_ids: list[str] = Field(
        default_factory=list,
        description="Optional KG rule IDs supporting the decision (used by Spec v1.2)",
    )
    question_intent: str | None = Field(
        default=None,
        description="Optional high-level intent label for the clarification question",
    )
    use_fallback: bool = Field(
        default=False,
        description="Whether deterministic FR-11 fallback applies when proceeding",
    )
    fallback_slots: list[str] = Field(
        default_factory=list,
        description="Unresolved or declined slots requiring fallback handling",
    )
    max_questions: int = Field(
        default=0,
        ge=0,
        description="Maximum questions allowed in ClarificationForm for this turn",
    )
