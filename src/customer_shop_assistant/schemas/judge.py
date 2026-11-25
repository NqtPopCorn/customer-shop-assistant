"""Pydantic v2 schemas for Tier-2 human evaluation judge sheet."""

from typing import Literal

from pydantic import BaseModel, Field

JudgeRowStatus = Literal["unscored", "partial", "scored"]


class JudgeRow(BaseModel):
    """Single row from eval/e1_tier2_judge_sheet.csv with parsed types."""

    row_index: int = Field(..., ge=0, description="Zero-based row index in CSV")
    source: str = Field(..., description="Evaluation tier source identifier")
    suite: str = Field(..., description="Scenario suite (contrastive or stratified)")
    scenario_id: str = Field(..., description="Scenario identifier")
    task_type: str = Field(..., description="Scenario task or contrast category")
    mode: str = Field(..., description="ClarificationMode used")
    turns_to_completion: int = Field(..., ge=0, description="Conversation turn count")
    question: str = Field(..., description="Initial customer query")
    final_reply: str = Field(..., description="Final assistant reply")
    clarification_asked: bool = Field(
        ..., description="Whether assistant asked clarification on turn 1"
    )
    asked_slots: str = Field(default="", description="Slots requested in form")
    decision_source: str = Field(default="", description="Routing decision source")
    reason_codes: str = Field(default="", description="Structured reason codes")
    tools_used: str = Field(default="", description="Tools invoked across turns")
    applied_rule_ids: str = Field(default="", description="KG rule IDs applied")
    products_summary: str = Field(default="", description="Retrieved products summary")
    outfits_summary: str = Field(default="", description="Generated outfits summary")
    grounding_ok: bool = Field(..., description="Automated grounding verification flag")
    recommendation_relevance_1_5: int | None = Field(
        default=None, ge=1, le=5, description="Human score 1-5 for relevance"
    )
    unnecessary_question_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Human flag 0/1 for unnecessary ask"
    )
    missed_clarification_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Human flag 0/1 for missed ask"
    )
    task_completion_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Human flag 0/1 for task completion"
    )
    explanation_faithfulness_1_5: int | None = Field(
        default=None, ge=1, le=5, description="Human score 1-5 for faithfulness"
    )
    judge_notes: str = Field(default="", description="Free-text judge notes")
    judge_name: str = Field(
        default="", description="Name of judge who last edited this row"
    )
    updated_at: str = Field(
        default="",
        description="ISO UTC timestamp of last edit (optimistic lock token)",
    )
    status: JudgeRowStatus = Field(
        ..., description="Scoring completion status for this row"
    )


class JudgeScoreUpdate(BaseModel):
    """Editable rubric score columns and judge notes for a single row."""

    recommendation_relevance_1_5: int | None = Field(
        default=None, ge=1, le=5, description="Relevance score (1-5 or None)"
    )
    unnecessary_question_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Unnecessary question flag (0/1)"
    )
    missed_clarification_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Missed clarification flag (0/1)"
    )
    task_completion_0_1: int | None = Field(
        default=None, ge=0, le=1, description="Task completion flag (0/1)"
    )
    explanation_faithfulness_1_5: int | None = Field(
        default=None, ge=1, le=5, description="Faithfulness score (1-5 or None)"
    )
    judge_notes: str = Field(default="", description="Free-text judge notes")
    judge_name: str | None = Field(
        default=None, description="Optional judge name recorded on save"
    )
    expected_updated_at: str | None = Field(
        default=None,
        description="Expected updated_at token for optimistic locking",
    )
    force: bool = Field(
        default=False,
        description="Force overwrite even when expected_updated_at mismatches",
    )


class JudgeConfigResponse(BaseModel):
    """Default Google Apps Script bridge configuration from server settings."""

    default_bridge_url: str = Field(
        default="", description="Default Apps Script URL from Settings"
    )
    default_bridge_token: str = Field(
        default="", description="Default Apps Script token from Settings"
    )


class JudgeModeSummary(BaseModel):
    """Aggregated human evaluation statistics for a single ClarificationMode."""

    mode: str = Field(..., description="ClarificationMode name")
    total: int = Field(..., ge=0, description="Total rows in this mode")
    scored: int = Field(..., ge=0, description="Fully scored rows")
    partial: int = Field(..., ge=0, description="Partially scored rows")
    unscored: int = Field(..., ge=0, description="Unscored rows")
    avg_relevance: float | None = Field(
        default=None, description="Average relevance (1-5)"
    )
    unnecessary_question_rate: float | None = Field(
        default=None, description="Percentage of unnecessary questions (%)"
    )
    missed_clarification_rate: float | None = Field(
        default=None, description="Percentage of missed clarifications (%)"
    )
    task_completion_rate: float | None = Field(
        default=None, description="Percentage of task completion (%)"
    )
    avg_faithfulness: float | None = Field(
        default=None, description="Average explanation faithfulness (1-5)"
    )


class JudgeSummaryResponse(BaseModel):
    """Aggregated summary across modes with paste-ready Markdown table."""

    suite: str = Field(..., description="Applied suite filter (all/contrastive/etc.)")
    total_rows: int = Field(..., ge=0, description="Total rows included")
    scored_rows: int = Field(..., ge=0, description="Fully scored rows")
    partial_rows: int = Field(..., ge=0, description="Partially scored rows")
    unscored_rows: int = Field(..., ge=0, description="Unscored rows")
    by_mode: list[JudgeModeSummary] = Field(
        default_factory=list, description="Per-mode summary rows"
    )
    markdown_table: str = Field(
        ..., description="GitHub-flavored Markdown summary table"
    )


class JudgeSheetResponse(BaseModel):
    """Response payload containing all judge sheet rows and progress counters."""

    csv_path: str = Field(..., description="Resolved path to the CSV file")
    total: int = Field(..., ge=0, description="Total rows in the CSV")
    scored: int = Field(..., ge=0, description="Count of fully scored rows")
    partial: int = Field(..., ge=0, description="Count of partially scored rows")
    unscored: int = Field(..., ge=0, description="Count of unscored rows")
    rows: list[JudgeRow] = Field(
        default_factory=list, description="All parsed judge rows"
    )


class JudgeUpdateResponse(BaseModel):
    """Response returned after updating a single judge row."""

    row: JudgeRow = Field(..., description="Updated row")
    total: int = Field(..., ge=0, description="Total rows in the CSV")
    scored: int = Field(..., ge=0, description="Updated fully scored count")
    partial: int = Field(..., ge=0, description="Updated partially scored count")
    unscored: int = Field(..., ge=0, description="Updated unscored count")
