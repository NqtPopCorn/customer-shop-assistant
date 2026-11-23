import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.eval.dataset import EvaluationCase
from customer_shop_assistant.schemas.chat import OutfitSuggestion
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.outfit_rules import (
    ProductAttrs,
    compute_formality,
    load_outfit_rules,
    normalize_text,
    resolve_category_role,
)


@lru_cache(maxsize=4)
def _load_attrs_for_eval(outfit_dir: Path) -> dict[str, ProductAttrs]:
    attrs_path = outfit_dir / "product_attrs.json"
    if not attrs_path.exists():
        return {}
    with attrs_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {slug: ProductAttrs.model_validate(item) for slug, item in raw.items()}


def _compute_outfit_violation_rate(
    case: EvaluationCase, outfits: list[OutfitSuggestion]
) -> float:
    outfit_dir = get_settings().DATA_DIR / "outfit_kg"
    rules = load_outfit_rules(outfit_dir)
    attrs_by_slug = _load_attrs_for_eval(outfit_dir.resolve())
    user_gender = case.user_profile.gender if case.user_profile else None

    violations = 0
    for outfit in outfits:
        violated = False
        top_role = resolve_category_role(outfit.top.category, rules)
        bot_role = resolve_category_role(outfit.bottom.category, rules)

        if outfit.top.category and (
            top_role is None or top_role[1] not in ("TOP", "JACKET")
        ):
            violated = True
        if outfit.bottom.category and (bot_role is None or bot_role[1] != "BOTTOM"):
            violated = True

        if (
            user_gender == GenderTarget.MALE
            and outfit.bottom.category
            and normalize_text(outfit.bottom.category) == "chân váy"
        ):
            violated = True

        top_attrs = attrs_by_slug.get(outfit.top.slug) if outfit.top.slug else None
        bot_attrs = (
            attrs_by_slug.get(outfit.bottom.slug) if outfit.bottom.slug else None
        )

        f_top: int | None = None
        f_bot: int | None = None
        if top_attrs is not None:
            f_top = top_attrs.formality
        elif outfit.top.category:
            f_top = compute_formality(outfit.top.category, None, rules)
        if bot_attrs is not None:
            f_bot = bot_attrs.formality
        elif outfit.bottom.category:
            f_bot = compute_formality(outfit.bottom.category, None, rules)

        if f_top is not None and f_bot is not None and abs(f_top - f_bot) > 2:
            violated = True

        if (
            top_attrs is not None
            and bot_attrs is not None
            and top_attrs.pattern_class == "PATTERN"
            and bot_attrs.pattern_class == "PATTERN"
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
