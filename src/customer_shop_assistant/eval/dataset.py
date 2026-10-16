import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.chat import ClarificationMode, UserProfile


class GroundTruth(BaseModel):
    """Expected evaluation criteria and ground-truth values."""

    expected_intent: str | None = Field(
        default=None, description="Expected intent string"
    )
    expected_tool: str | None = Field(default=None, description="Expected tool called")
    expected_policy_doc: str | None = Field(
        default=None, description="Expected policy markdown filename"
    )
    required_keywords: list[str] = Field(
        default_factory=list, description="Keywords that must appear in reply"
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list, description="Keywords that must not appear in reply"
    )
    should_clarify: bool = Field(
        default=False, description="Whether clarification question should be asked"
    )
    is_out_of_scope: bool = Field(
        default=False, description="Whether question is outside assistant domain"
    )
    is_out_of_stock: bool = Field(
        default=False, description="Whether inquiry involves an out-of-stock item"
    )


class EvaluationCase(BaseModel):
    """Single benchmark evaluation test case."""

    id: str = Field(description="Unique case identifier")
    category: str = Field(description="Category (product_suitability, size_fit, etc.)")
    sub_category: str | None = Field(
        default=None, description="Fine-grained scenario tag"
    )
    question: str = Field(description="Customer Vietnamese query")
    user_profile: UserProfile | None = Field(
        default=None, description="Customer body metrics"
    )
    clarification_mode: ClarificationMode = Field(
        default=ClarificationMode.MINIMAL_CLARIFICATION,
        description="Clarification mode under test",
    )
    ground_truth: GroundTruth = Field(
        default_factory=GroundTruth, description="Target evaluation criteria"
    )


def load_golden_set(
    filepath: Path | str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[EvaluationCase]:
    """Load and validate the evaluation golden set from JSON file.

    Parameters
    ----------
    filepath:
        Optional path to questions JSON. Defaults to eval/questions.json.
    category:
        Optional category filter (e.g. 'size_fit', 'product_suitability').
    limit:
        Optional cap on number of cases returned.
    """
    if filepath is None:
        target = Path("eval/questions.json")
    else:
        target = Path(filepath)

    if not target.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {target}")

    with target.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    raw_list = data.get("questions", [])
    cases: list[EvaluationCase] = []

    for item in raw_list:
        case = EvaluationCase.model_validate(item)
        if category and case.category.lower() != category.lower():
            continue
        cases.append(case)
        if limit is not None and len(cases) >= limit:
            break

    return cases
