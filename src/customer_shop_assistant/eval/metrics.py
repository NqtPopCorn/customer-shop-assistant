"""Automated evaluation metrics, structural checks, and grounding validators."""

import re
from typing import Any

from pydantic import BaseModel, Field

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.eval.dataset import EvaluationCase
from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
)
from customer_shop_assistant.schemas.chat import (
    ChatRecommendedProduct,
    ClarificationForm,
    ClarificationMode,
    OutfitSuggestion,
)
from customer_shop_assistant.schemas.shop import GenderTarget

_PRODUCT_ID_PATTERN = re.compile(
    r"(?:\[ID\]\s*:?\s*(\d+)\b|(?:ID:\s*\[|\[)(\d+)\])",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(
    r"\b\d{1,3}(?:[.,]\d{3})+\s*(?:đ|vnđ|vnd)\b",
    re.IGNORECASE,
)
_CLARIFY_TEXT_CUES = (
    "bạn muốn tìm",
    "đối tượng nào",
    "nam hay nữ",
    "cho mình biết",
    "cho shop biết",
    "cần biết thêm",
    "cung cấp thêm",
    "một số thông tin",
    "chiều cao và cân nặng",
    "bạn cao bao nhiêu",
)
_PHANTOM_FORM_CUES = (
    "tích chọn",
    "bảng dưới đây",
    "bảng bên dưới",
    "form bên dưới",
    "form dưới đây",
    "chọn nhanh vài tiêu chí",
)
_CLOSING_QUESTION_CUES = (
    "hỗ trợ thêm",
    "giúp gì thêm",
    "cần mình hỗ trợ",
    "cần shop hỗ trợ",
    "thắc mắc nào khác",
    "câu hỏi nào khác",
)
_MALE_FORBIDDEN_CATEGORIES = frozenset({"chan_vay", "vay_lien", "dam_da_hoi"})


class TokenUsageSummary(BaseModel):
    """Normalized LLM token usage extracted from TurnDebugMeta."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    is_estimated: bool = False


class SlotAccuracyResult(BaseModel):
    """Precision, recall, and exact match for asked vs expected clarify slots."""

    precision: float | None = None
    recall: float | None = None
    exact_match: bool | None = None


class StructuralCheckResult(BaseModel):
    """Mode compliance, form integrity, and catalog grounding check results."""

    mode_compliant: bool = True
    form_integrity_ok: bool = True
    id_grounding_ok: bool = True
    price_grounding_ok: bool = True
    gender_grounding_ok: bool = True
    budget_grounding_ok: bool = True
    outfit_rule_violation_rate: float | None = None
    violations: list[str] = Field(default_factory=list)

    @property
    def grounding_ok(self) -> bool:
        """Return True when all grounding and outfit rule checks pass."""
        return (
            self.id_grounding_ok
            and self.price_grounding_ok
            and self.gender_grounding_ok
            and self.budget_grounding_ok
            and (
                self.outfit_rule_violation_rate is None
                or self.outfit_rule_violation_rate == 0.0
            )
        )


def _normalize_mode_value(mode: ClarificationMode | str) -> ClarificationMode:
    if isinstance(mode, ClarificationMode):
        if mode == ClarificationMode.MINIMAL_CLARIFICATION:
            return ClarificationMode.CONDITIONAL_CLARIFICATION
        return mode
    clean = str(mode).strip().upper()
    if clean in ("MINIMAL_CLARIFICATION", "MINIMAL"):
        return ClarificationMode.CONDITIONAL_CLARIFICATION
    if clean in ClarificationMode.__members__:
        return ClarificationMode[clean]
    return ClarificationMode.CONDITIONAL_CLARIFICATION


def _extract_product_ids(text: str) -> list[int]:
    ids: list[int] = []
    for g1, g2 in _PRODUCT_ID_PATTERN.findall(text):
        raw = g1 or g2
        if raw and raw.isdigit():
            val = int(raw)
            if val not in ids:
                ids.append(val)
    return ids


def _extract_prices_vnd(text: str) -> list[float]:
    amounts: list[float] = []
    for m in _PRICE_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if digits:
            amounts.append(float(digits))
    return amounts


def _reply_leaks_clarification(
    reply: str,
    *,
    has_grounded_items: bool,
) -> bool:
    """Detect clarifying questions leaked into reply text in Mode A."""
    if not reply or "?" not in reply:
        return False
    lower = reply.lower()
    # Check sentences containing '?' excluding polite closing offers
    sentences = [s.strip() for s in re.split(r"[.\n!]+", lower) if "?" in s]
    non_closing = [
        s for s in sentences if not any(c in s for c in _CLOSING_QUESTION_CUES)
    ]
    if not non_closing:
        return False
    if any(any(cue in s for cue in _CLARIFY_TEXT_CUES) for s in non_closing):
        return True
    if not has_grounded_items and "size" not in lower:
        return True
    return False


def extract_token_usage(
    debug_meta: Any,
    *,
    question: str = "",
    reply: str = "",
    allow_estimate: bool = False,
) -> TokenUsageSummary:
    """Extract real token counts from debug_meta.usage_metadata."""
    raw_usage: Any = None
    if isinstance(debug_meta, BaseModel):
        raw_usage = getattr(debug_meta, "usage_metadata", None)
    elif isinstance(debug_meta, dict):
        raw_usage = debug_meta.get("usage_metadata")

    usage: dict[str, Any] = {}
    if isinstance(raw_usage, BaseModel):
        usage = raw_usage.model_dump()
    elif isinstance(raw_usage, dict):
        usage = raw_usage

    in_tok = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    out_tok = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    tot_tok = int(usage.get("total_tokens", 0) or (in_tok + out_tok))

    if tot_tok > 0 or not allow_estimate:
        return TokenUsageSummary(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            is_estimated=False,
        )

    est_in = max(1, int(len(question) / 3.5)) if question else 1
    est_out = max(1, int(len(reply) / 3.5)) if reply else 1
    return TokenUsageSummary(
        input_tokens=est_in,
        output_tokens=est_out,
        total_tokens=est_in + est_out,
        is_estimated=True,
    )


def compute_slot_accuracy(
    expected_slots: list[str], asked_slots: list[str]
) -> SlotAccuracyResult:
    """Compute precision, recall, and exact match for asked vs expected slots."""
    exp_set = {s.strip() for s in expected_slots if s and s.strip()}
    ask_set = {s.strip() for s in asked_slots if s and s.strip()}

    if not exp_set and not ask_set:
        return SlotAccuracyResult()
    if not exp_set:
        return SlotAccuracyResult(precision=0.0, recall=None, exact_match=False)

    tp = len(exp_set & ask_set)
    precision = round(tp / len(ask_set), 4) if ask_set else 0.0
    recall = round(tp / len(exp_set), 4)
    return SlotAccuracyResult(
        precision=precision,
        recall=recall,
        exact_match=(exp_set == ask_set),
    )


def _compute_outfit_violation_rate(
    outfits: list[OutfitSuggestion],
    target_gender: GenderTarget | str | None = None,
) -> float:
    ont_dir = (get_settings().DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    cats_by_id = {c.id: c for c in ontology.categories}

    gender_str: str | None = None
    if isinstance(target_gender, GenderTarget):
        gender_str = target_gender.value
    elif isinstance(target_gender, str) and target_gender.strip():
        gender_str = target_gender.strip().upper()

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
        if gender_str == GenderTarget.MALE.value and bot_id == "chan_vay":
            violated = True
        if (
            top_node is not None
            and bot_node is not None
            and abs(top_node.base_formality - bot_node.base_formality) > 2
        ):
            violated = True
        if not outfit.top.in_stock or not outfit.bottom.in_stock:
            violated = True

        if violated:
            violations += 1

    return round(violations / len(outfits), 4)


def evaluate_turn_structure_and_grounding(
    *,
    mode: ClarificationMode | str,
    reply: str,
    intent: str | None = None,
    clarification_asked: bool = False,
    clarification_form: ClarificationForm | dict[str, Any] | None = None,
    clarification_turns: int = 0,
    tools_used: list[str] | None = None,
    products: list[ChatRecommendedProduct] | None = None,
    outfits: list[OutfitSuggestion] | None = None,
    target_gender: GenderTarget | str | None = None,
    max_price: float | None = None,
    user_question: str = "",
) -> StructuralCheckResult:
    """Validate mode compliance, form integrity, and grounding on a live/mock turn."""
    norm_mode = _normalize_mode_value(mode)
    used_tools = list(tools_used or [])
    prod_list = list(products or [])
    outfit_list = list(outfits or [])
    violations: list[str] = []

    form_obj: ClarificationForm | None = None
    if isinstance(clarification_form, ClarificationForm):
        form_obj = clarification_form
    elif isinstance(clarification_form, dict):
        form_obj = ClarificationForm.model_validate(clarification_form)

    has_grounded_items = bool(prod_list or outfit_list)
    mode_compliant = True
    form_integrity_ok = True
    id_grounding_ok = True
    price_grounding_ok = True
    gender_grounding_ok = True
    budget_grounding_ok = True

    # 1. Mode compliance
    if norm_mode == ClarificationMode.IMMEDIATE_ANSWER:
        if clarification_asked or form_obj is not None:
            mode_compliant = False
            violations.append("Mode A asked clarification or attached a form.")
        elif _reply_leaks_clarification(reply, has_grounded_items=has_grounded_items):
            mode_compliant = False
            violations.append("Mode A leaked clarifying question in reply text.")
    elif norm_mode == ClarificationMode.ALWAYS_CLARIFY:
        if clarification_asked:
            if any(t in used_tools for t in ("search_products", "suggest_outfit")):
                mode_compliant = False
                violations.append("Mode B ran search/outfit tool during ASK turn.")
            if form_obj is not None and len(form_obj.questions) > 3:
                mode_compliant = False
                violations.append(
                    f"Mode B form exceeded 3 questions ({len(form_obj.questions)})."
                )
            if clarification_turns > 2:
                mode_compliant = False
                violations.append(
                    f"Mode B exceeded max 2 turns (turn={clarification_turns})."
                )
    elif norm_mode == ClarificationMode.CONDITIONAL_CLARIFICATION:
        if clarification_asked:
            if form_obj is not None and len(form_obj.questions) > 2:
                mode_compliant = False
                violations.append(
                    f"Mode C form exceeded 2 questions ({len(form_obj.questions)})."
                )
            if clarification_turns > 2:
                mode_compliant = False
                violations.append(
                    f"Mode C exceeded max 2 turns (turn={clarification_turns})."
                )

    # 2. Form integrity
    lower_reply = reply.lower()
    if form_obj is None and any(cue in lower_reply for cue in _PHANTOM_FORM_CUES):
        form_integrity_ok = False
        violations.append("Phantom form reference in reply when form is None.")
    if clarification_asked and form_obj is not None and len(form_obj.questions) == 0:
        form_integrity_ok = False
        violations.append("clarification_asked=True but clarification_form is empty.")

    # 3. ID & price grounding (when catalog items/outfits are returned)
    if has_grounded_items:
        allowed_ids: set[int] = {p.id for p in prod_list}
        allowed_prices: set[float] = set()
        for p in prod_list:
            allowed_prices.add(float(p.price))
            if p.original_price is not None:
                allowed_prices.add(float(p.original_price))
        for o in outfit_list:
            allowed_ids.add(o.top.product_id)
            allowed_ids.add(o.bottom.product_id)
            allowed_prices.add(float(o.top.price))
            allowed_prices.add(float(o.bottom.price))
            allowed_prices.add(float(o.top.price + o.bottom.price))

        mentioned_ids = _extract_product_ids(reply)
        bad_ids = [pid for pid in mentioned_ids if pid not in allowed_ids]
        if bad_ids:
            id_grounding_ok = False
            violations.append(f"Hallucinated product IDs in reply: {bad_ids}")

        if (intent or "").upper() != "NON_SHOPPING":
            user_prices = set(_extract_prices_vnd(user_question))
            mentioned_prices = _extract_prices_vnd(reply)
            bad_prices = [
                mp
                for mp in mentioned_prices
                if not any(abs(mp - ap) <= 1000.0 for ap in allowed_prices)
                and not any(abs(mp - up) <= 1000.0 for up in user_prices)
            ]
            if bad_prices:
                price_grounding_ok = False
                violations.append(f"Unmatched product prices in reply: {bad_prices}")

    # 4. Gender & budget hard constraints on returned items
    gender_str: str | None = None
    if isinstance(target_gender, GenderTarget):
        gender_str = target_gender.value
    elif isinstance(target_gender, str) and target_gender.strip():
        gender_str = target_gender.strip().upper()

    ont_dir = (get_settings().DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))

    if gender_str == GenderTarget.MALE.value:
        for p in prod_list:
            cid = resolve_category_id(p.category, ontology) if p.category else None
            if cid in _MALE_FORBIDDEN_CATEGORIES:
                gender_grounding_ok = False
                violations.append(
                    f"Gender violation: product {p.id} ({p.category}) for MALE."
                )
        for o in outfit_list:
            bot_cid = (
                resolve_category_id(o.bottom.category, ontology)
                if o.bottom.category
                else None
            )
            if bot_cid in _MALE_FORBIDDEN_CATEGORIES:
                gender_grounding_ok = False
                violations.append(
                    f"Gender violation: outfit bottom {o.bottom.product_id} "
                    f"({o.bottom.category}) for MALE."
                )

    if max_price is not None:
        for p in prod_list:
            if p.price > max_price:
                budget_grounding_ok = False
                violations.append(
                    f"Budget violation: product {p.id} price {p.price} > {max_price}."
                )
        for o in outfit_list:
            if o.top.price > max_price or o.bottom.price > max_price:
                budget_grounding_ok = False
                violations.append(
                    f"Budget violation: outfit item exceeds max_price {max_price}."
                )

    # 5. Outfit rule violation rate
    outfit_viol_rate: float | None = None
    if outfit_list:
        outfit_viol_rate = _compute_outfit_violation_rate(outfit_list, gender_str)
        if outfit_viol_rate > 0.0:
            violations.append(
                f"Outfit KG/stock rule violation rate = {outfit_viol_rate:.1%}."
            )

    return StructuralCheckResult(
        mode_compliant=mode_compliant,
        form_integrity_ok=form_integrity_ok,
        id_grounding_ok=id_grounding_ok,
        price_grounding_ok=price_grounding_ok,
        gender_grounding_ok=gender_grounding_ok,
        budget_grounding_ok=budget_grounding_ok,
        outfit_rule_violation_rate=outfit_viol_rate,
        violations=violations,
    )


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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    provider: str = ""
    model_name: str = ""
    decision_source: str | None = None
    asked_slots: list[str] = Field(default_factory=list)

    # Individual binary/fractional scores
    intent_match: bool = True
    tool_match: bool = True
    clarification_compliance: bool = True
    policy_citation_match: bool = True
    out_of_scope_match: bool = True
    out_of_stock_match: bool = True
    forbidden_keyword_ok: bool = True
    mode_compliant: bool = True
    form_integrity_ok: bool = True
    grounding_ok: bool = True
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
    estimated_tokens: int = 0,
    outfits: list[OutfitSuggestion] | None = None,
    *,
    products: list[ChatRecommendedProduct] | None = None,
    clarification_form: ClarificationForm | dict[str, Any] | None = None,
    debug_meta: Any = None,
    provider: str = "",
    model_name: str = "",
    allow_token_estimate: bool = False,
) -> EvaluationMetricResult:
    """Evaluate an assistant turn against ground-truth and structural checks."""
    gt = case.ground_truth
    lower_reply = reply.lower()
    reasons: list[str] = []

    tok = extract_token_usage(
        debug_meta,
        question=case.question,
        reply=reply,
        allow_estimate=allow_token_estimate,
    )
    eff_total_tokens = tok.total_tokens if tok.total_tokens > 0 else estimated_tokens

    # 1. Intent Match
    intent_match = True
    if gt.expected_intent:
        got_i = intent.upper()
        exp_i = gt.expected_intent.upper()
        if got_i != exp_i:
            if not {got_i, exp_i}.issubset({"CATALOG_SEARCH", "PRODUCT_ADVICE"}):
                intent_match = False
                reasons.append(f"Intent mismatch: got {intent}, expected {exp_i}")

    # 2. Tool Match
    tool_match = True
    if (
        gt.expected_tool
        and gt.expected_tool not in tools_used
        and not clarification_asked
    ):
        tool_match = False
        reasons.append(
            f"Expected tool '{gt.expected_tool}' not called. Tools used: {tools_used}"
        )

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

    # 7. Diagnostic Keyword Overlap (soft signal) & Hard Forbidden Check
    keyword_score = 1.0
    if gt.required_keywords:
        found_count = sum(1 for kw in gt.required_keywords if kw.lower() in lower_reply)
        keyword_score = found_count / len(gt.required_keywords)

    forbidden_keyword_ok = True
    for fkw in gt.forbidden_keywords:
        if fkw.lower() in lower_reply:
            forbidden_keyword_ok = False
            reasons.append(f"Forbidden keyword found: '{fkw}'")

    # 8. Structural & Grounding Checks
    meta_dict = (
        debug_meta.model_dump()
        if isinstance(debug_meta, BaseModel)
        else (debug_meta if isinstance(debug_meta, dict) else {})
    )
    clar_turns = int(meta_dict.get("clarification_turns", 0) or 0)
    dec_source = meta_dict.get("decision_source")
    asked_slots = list(meta_dict.get("asked_slots") or [])
    user_gender = case.user_profile.gender if case.user_profile else None

    struct = evaluate_turn_structure_and_grounding(
        mode=case.clarification_mode,
        reply=reply,
        intent=intent,
        clarification_asked=clarification_asked,
        clarification_form=clarification_form,
        clarification_turns=clar_turns,
        tools_used=tools_used,
        products=products,
        outfits=outfits,
        target_gender=user_gender,
        user_question=case.question,
    )
    reasons.extend(struct.violations)

    # 9. Outfit KG metrics
    outfit_returned: bool | None = None
    outfit_rule_violation_rate = struct.outfit_rule_violation_rate
    if gt.expected_tool == "suggest_outfit" and outfits is not None:
        outfit_returned = len(outfits) > 0

    latency_pass = latency_ms < 6000.0
    grounding_ok = struct.grounding_ok

    passed = (
        intent_match
        and tool_match
        and clarification_compliance
        and policy_citation_match
        and out_of_scope_match
        and out_of_stock_match
        and forbidden_keyword_ok
        and struct.mode_compliant
        and struct.form_integrity_ok
        and grounding_ok
    )

    return EvaluationMetricResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        reply=reply,
        clarification_asked=clarification_asked,
        tools_used=tools_used,
        latency_ms=round(latency_ms, 1),
        estimated_tokens=eff_total_tokens,
        input_tokens=tok.input_tokens,
        output_tokens=tok.output_tokens,
        total_tokens=eff_total_tokens,
        provider=provider,
        model_name=model_name,
        decision_source=dec_source,
        asked_slots=asked_slots,
        intent_match=intent_match,
        tool_match=tool_match,
        clarification_compliance=clarification_compliance,
        policy_citation_match=policy_citation_match,
        out_of_scope_match=out_of_scope_match,
        out_of_stock_match=out_of_stock_match,
        forbidden_keyword_ok=forbidden_keyword_ok,
        mode_compliant=struct.mode_compliant,
        form_integrity_ok=struct.form_integrity_ok,
        grounding_ok=grounding_ok,
        keyword_score=round(keyword_score, 2),
        latency_pass=latency_pass,
        outfit_returned=outfit_returned,
        outfit_rule_violation_rate=outfit_rule_violation_rate,
        passed=passed,
        failure_reasons=reasons,
    )
