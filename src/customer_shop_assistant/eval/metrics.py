from pydantic import BaseModel, Field

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.eval.dataset import EvaluationCase
from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
)
from customer_shop_assistant.schemas.chat import OutfitSuggestion
from customer_shop_assistant.schemas.shop import GenderTarget


def _compute_outfit_violation_rate(
    case: EvaluationCase, outfits: list[OutfitSuggestion]
) -> float:
    ont_dir = (get_settings().DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    cats_by_id = {c.id: c for c in ontology.categories}
    user_gender = case.user_profile.gender if case.user_profile else None

    violations = 0
    for outfit in outfits:
        violated = False
        top_id = resolve_category_id(outfit.top.category, ontology)
        bot_id = resolve_category_id(outfit.bottom.category, ontology)
        top_node = cats_by_id.get(top_id) if top_id else None
        bot_node = cats_by_id.get(bot_id) if bot_id else None

        if outfit.top.category and (
            top_node is None or top_node.role not in ("TOP", "OUTERWEAR")
        ):
            violated = True
        if outfit.bottom.category and (bot_node is None or bot_node.role != "BOTTOM"):
            violated = True

        if user_gender == GenderTarget.MALE and bot_id == "chan_vay":
            violated = True

        if (
            top_node is not None
            and bot_node is not None
            and abs(top_node.base_formality - bot_node.base_formality) > 2
        ):
            violated = True

        if violated:
            violations += 1

    return round(violations / len(outfits), 4)


class EvaluationMetricResult(BaseModel):
    """Detailed score breakdown for a single evaluation case."""

    case_id: str
    category: str
    question: str
    reply: str
    clarification_asked: bool
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    estimated_tokens: int = 0

    # Individual binary/fractional scores
    intent_match: bool = True
    tool_match: bool = True
    clarification_compliance: bool = True
    policy_citation_match: bool = True
    out_of_scope_match: bool = True
    out_of_stock_match: bool = True
    keyword_score: float = 1.0
    latency_pass: bool = True
    outfit_returned: bool | None = None
    outfit_rule_violation_rate: float | None = None

    # Overall outcome
    passed: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


def evaluate_response(
    case: EvaluationCase,
    reply: str,
    intent: str,
    tools_used: list[str],
    clarification_asked: bool,
    latency_ms: float,
    estimated_tokens: int,
    outfits: list[OutfitSuggestion] | None = None,
) -> EvaluationMetricResult:
    """Evaluate an assistant turn against ground-truth criteria."""
    gt = case.ground_truth
    lower_reply = reply.lower()
    reasons: list[str] = []

    # 1. Intent Match
    intent_match = True
    if gt.expected_intent:
        if intent.upper() != gt.expected_intent.upper():
            # Allow CATALOG_SEARCH <-> PRODUCT_ADVICE interchangeability
            if not {intent.upper(), gt.expected_intent.upper()}.issubset(
                {"CATALOG_SEARCH", "PRODUCT_ADVICE"}
            ):
                intent_match = False
                reasons.append(
                    f"Intent mismatch: got {intent}, expected {gt.expected_intent}"
                )

    # 2. Tool Match
    tool_match = True
    if gt.expected_tool:
        if gt.expected_tool not in tools_used and not clarification_asked:
            tool_match = False
            reasons.append(
                f"Expected tool '{gt.expected_tool}' not called. "
                f"Tools used: {tools_used}"
            )
    elif gt.should_clarify:
        # If clarifying, should ideally not execute unnecessary heavy tools
        pass

    # 3. Clarification Compliance
    clarification_compliance = True
    if gt.should_clarify and not clarification_asked:
        clarification_compliance = False
        reasons.append(
            f"Mode {case.clarification_mode.value}: "
            "Expected clarification question but got direct response."
        )
    elif not gt.should_clarify and clarification_asked:
        clarification_compliance = False
        reasons.append(
            f"Mode {case.clarification_mode.value}: "
            "Asked clarification when direct answer was expected."
        )

    # 4. Policy Citation Match
    policy_citation_match = True
    if gt.expected_policy_doc:
        doc_token = f"[{gt.expected_policy_doc.lower()}"
        if (
            doc_token not in lower_reply
            and gt.expected_policy_doc.lower() not in lower_reply
        ):
            policy_citation_match = False
            reasons.append(
                f"Policy doc '{gt.expected_policy_doc}' citation not found in response."
            )

    # 5. Out of Scope Rejection
    out_of_scope_match = True
    if gt.is_out_of_scope:
        refusal_cues = [
            "chưa hỗ trợ",
            "liên hệ nhân viên",
            "thời trang",
            "không hỗ trợ",
        ]
        if not any(cue in lower_reply for cue in refusal_cues):
            out_of_scope_match = False
            reasons.append("Failed to politely refuse out-of-scope query.")

    # 6. Out of Stock Handling
    out_of_stock_match = True
    if gt.is_out_of_stock:
        stock_cues = ["hết", "tạm hết", "thay thế", "mẫu khác", "gợi ý"]
        if not any(cue in lower_reply for cue in stock_cues):
            out_of_stock_match = False
            reasons.append(
                "Did not properly identify out-of-stock item or provide alternative."
            )

    # 7. Keyword & Forbidden Check
    keyword_score = 1.0
    if gt.required_keywords:
        found_count = sum(1 for kw in gt.required_keywords if kw.lower() in lower_reply)
        keyword_score = found_count / len(gt.required_keywords)
        if keyword_score < 0.4 and not clarification_asked and not gt.is_out_of_scope:
            reasons.append(
                f"Low keyword overlap: {keyword_score:.1%} for {gt.required_keywords}"
            )

    for fkw in gt.forbidden_keywords:
        if fkw.lower() in lower_reply:
            reasons.append(f"Forbidden keyword found: '{fkw}'")

    # 8. Latency threshold (target < 4000ms from PRD, allow dry-run)
    latency_pass = latency_ms < 6000.0

    # 9. Outfit KG metrics (for mix & match cases)
    outfit_returned: bool | None = None
    outfit_rule_violation_rate: float | None = None
    if gt.expected_tool == "suggest_outfit" and outfits is not None:
        outfit_returned = len(outfits) > 0
        if outfits:
            outfit_rule_violation_rate = _compute_outfit_violation_rate(case, outfits)

    passed = (
        intent_match
        and tool_match
        and clarification_compliance
        and policy_citation_match
        and out_of_scope_match
        and out_of_stock_match
        and (keyword_score >= 0.4 or clarification_asked or gt.is_out_of_scope)
    )

    return EvaluationMetricResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        reply=reply,
        clarification_asked=clarification_asked,
        tools_used=tools_used,
        latency_ms=round(latency_ms, 1),
        estimated_tokens=estimated_tokens,
        intent_match=intent_match,
        tool_match=tool_match,
        clarification_compliance=clarification_compliance,
        policy_citation_match=policy_citation_match,
        out_of_scope_match=out_of_scope_match,
        out_of_stock_match=out_of_stock_match,
        keyword_score=round(keyword_score, 2),
        latency_pass=latency_pass,
        outfit_returned=outfit_returned,
        outfit_rule_violation_rate=outfit_rule_violation_rate,
        passed=passed,
        failure_reasons=reasons,
    )
