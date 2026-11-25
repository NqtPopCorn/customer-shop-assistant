"""Pure functional clarification policy, hybrid guards, and state transitions."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.schemas.clarification import ClarificationDecision
from customer_shop_assistant.services.session_manager import ClarifyPhase
from customer_shop_assistant.services.slot_tracker import (
    ConversationSlots,
    TurnIntent,
    get_conditional_slots,
    get_missing_required_slots,
    get_required_slots,
)

MAX_GATED_TURNS = 1
MAX_MODE_B_TURNS = 2
MAX_MODE_B_QUESTIONS = 3
MAX_MODE_C_TURNS = 1
MAX_MODE_C_CONFLICT_TURNS = 2
MAX_MODE_C_QUESTIONS = 2


class PromptKey(StrEnum):
    """Identifier selecting the policy prompt block for the current turn."""

    MODE_A = "MODE_A"
    MODE_B_INTERVIEW = "MODE_B_INTERVIEW"
    MODE_B_OPEN = "MODE_B_OPEN"
    MODE_C = "MODE_C"
    NON_SHOPPING = "NON_SHOPPING"


@dataclass(frozen=True)
class TurnPolicy:
    """Policy resolved for the current turn."""

    gate_search: bool
    prompt_key: PromptKey


class GrayZoneEvaluator(Protocol):
    """Hook for plug-in gray-zone evaluation (e.g. Spec v1.2 Counterfactual Policy)."""

    def evaluate(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Return a ClarificationDecision or None to fall back to LLM evaluation."""


def _normalize_mode(mode: ClarificationMode | str) -> ClarificationMode:
    if isinstance(mode, str):
        mode_clean = mode.strip().upper()
        if mode_clean in (
            ClarificationMode.CONDITIONAL_CLARIFICATION,
            "MINIMAL_CLARIFICATION",
        ):
            return ClarificationMode.CONDITIONAL_CLARIFICATION
        if mode_clean in ClarificationMode.__members__:
            return ClarificationMode[mode_clean]
        return ClarificationMode.CONDITIONAL_CLARIFICATION
    if mode == ClarificationMode.MINIMAL_CLARIFICATION:
        return ClarificationMode.CONDITIONAL_CLARIFICATION
    return mode


def is_broad_empty_inquiry(slots: ConversationSlots) -> bool:
    """Return True when the shopping request lacks any actionable anchor slot."""
    anchor_slots = (
        "category",
        "occasion",
        "top_category",
        "bottom_category",
        "body_concern",
        "measurements",
        "budget",
        "product_id",
    )
    return not any(slots.has(s) for s in anchor_slots)


def evaluate_clarification_policy(
    mode: ClarificationMode | str,
    intent: TurnIntent,
    slots: ConversationSlots,
    clarification_turns: int = 0,
    gray_zone_evaluator: GrayZoneEvaluator | None = None,
) -> ClarificationDecision:
    """Evaluate clarification decision across Mode A, Mode B, and Hybrid Mode C."""
    if intent == TurnIntent.NON_SHOPPING:
        return ClarificationDecision(
            action="PROCEED",
            decision_source="non_shopping_bypass",
            decision_reason="Non-shopping intent bypasses clarification.",
            reason_codes=["NON_SHOPPING_BYPASS"],
        )

    norm_mode = _normalize_mode(mode)
    missing_req = get_missing_required_slots(intent, slots)
    req_slots = get_required_slots(intent, slots)
    declined_req = [s for s in req_slots if slots.is_declined(s)]

    if norm_mode == ClarificationMode.IMMEDIATE_ANSWER:
        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_a_immediate",
            decision_reason="IMMEDIATE_ANSWER enforces strict 0-clarification.",
            reason_codes=["MODE_A_STRICT_ZERO"],
            use_fallback=bool(missing_req or declined_req),
            fallback_slots=list(dict.fromkeys([*missing_req, *declined_req])),
        )

    if norm_mode == ClarificationMode.ALWAYS_CLARIFY:
        if not missing_req and not declined_req:
            cat_val = slots.value_of("category")
            if (
                intent == TurnIntent.PRODUCT_SEARCH
                and clarification_turns < MAX_MODE_B_TURNS
                and cat_val in ("tops", "pants")
                and not slots.is_declined("category_subtype")
                and not slots.has("occasion")
                and not slots.has("budget")
            ):
                refine_targets = [
                    s
                    for s in ("category_subtype", "occasion", "budget")
                    if not slots.has(s) and not slots.is_declined(s)
                ][:MAX_MODE_B_QUESTIONS]
                if refine_targets:
                    return ClarificationDecision(
                        action="ASK",
                        target_slots=refine_targets,
                        decision_source="mode_b_required_check",
                        decision_reason=(
                            f"Broad category '{cat_val}' needs refinement: "
                            f"{', '.join(refine_targets)} "
                            f"(turn {clarification_turns + 1}/{MAX_MODE_B_TURNS})."
                        ),
                        reason_codes=["BROAD_CATEGORY_REFINEMENT"],
                        max_questions=MAX_MODE_B_QUESTIONS,
                    )
            return ClarificationDecision(
                action="PROCEED",
                decision_source="mode_b_required_check",
                decision_reason="All required slots are resolved.",
                reason_codes=["REQUIRED_SLOTS_COMPLETE"],
            )
        if not missing_req and declined_req:
            return ClarificationDecision(
                action="PROCEED",
                decision_source="fallback",
                decision_reason=(
                    f"Required slots declined: {', '.join(declined_req)}; "
                    "applying deterministic fallback."
                ),
                reason_codes=["REQUIRED_SLOT_DECLINED_FALLBACK"],
                use_fallback=True,
                fallback_slots=declined_req,
            )
        if clarification_turns >= MAX_MODE_B_TURNS:
            fb_slots = list(dict.fromkeys([*missing_req, *declined_req]))
            return ClarificationDecision(
                action="PROCEED",
                decision_source="fallback",
                decision_reason=(
                    f"Mode B turn cap ({MAX_MODE_B_TURNS}) reached with unresolved "
                    f"required slots: {', '.join(missing_req)}."
                ),
                reason_codes=["MODE_B_TURN_CAP_FALLBACK"],
                use_fallback=True,
                fallback_slots=fb_slots,
            )
        targets = missing_req[:MAX_MODE_B_QUESTIONS]
        return ClarificationDecision(
            action="ASK",
            target_slots=targets,
            decision_source="mode_b_required_check",
            decision_reason=(
                f"Missing required slots: {', '.join(targets)} "
                f"(turn {clarification_turns + 1}/{MAX_MODE_B_TURNS})."
            ),
            reason_codes=["MISSING_REQUIRED_SLOTS"],
            max_questions=MAX_MODE_B_QUESTIONS,
        )

    # Mode C: CONDITIONAL_CLARIFICATION (Hybrid: Hard Guards + Gray Zone)
    conflicting = slots.get_conflicting_slots()

    # 1. Hard Ask Guard: CONFLICTING within conflict turn cap
    if conflicting and clarification_turns < MAX_MODE_C_CONFLICT_TURNS:
        targets = conflicting[:MAX_MODE_C_QUESTIONS]
        return ClarificationDecision(
            action="ASK",
            target_slots=targets,
            decision_source="mode_c_hard_ask",
            decision_reason=(
                f"Conflicting constraints in request: {', '.join(targets)}."
            ),
            reason_codes=["HARD_ASK_CONFLICTING_SLOTS"],
            max_questions=MAX_MODE_C_QUESTIONS,
        )

    # 2. Hard Ask Guard: SIZE_ADVICE missing measurements & Broad Empty Inquiry
    if clarification_turns < MAX_MODE_C_TURNS:
        if intent == TurnIntent.SIZE_ADVICE and "measurements" in missing_req:
            size_targets = ["measurements"]
            for s in missing_req:
                if s != "measurements" and len(size_targets) < MAX_MODE_C_QUESTIONS:
                    size_targets.append(s)
            return ClarificationDecision(
                action="ASK",
                target_slots=size_targets,
                decision_source="mode_c_hard_ask",
                decision_reason="SIZE_ADVICE requires measurements before sizing.",
                reason_codes=["HARD_ASK_SIZE_MISSING_MEASUREMENTS"],
                max_questions=MAX_MODE_C_QUESTIONS,
            )

        if is_broad_empty_inquiry(slots):
            if intent == TurnIntent.OUTFIT_ADVICE:
                broad_candidates = ["target_gender", "occasion"]
            elif intent == TurnIntent.BODY_FIT_ADVICE:
                broad_candidates = ["target_gender", "body_concern"]
            else:
                broad_candidates = ["target_gender", "occasion", "category"]

            broad_targets = [
                s
                for s in broad_candidates
                if not slots.has(s) and not slots.is_declined(s)
            ][:MAX_MODE_C_QUESTIONS]

            if broad_targets:
                return ClarificationDecision(
                    action="ASK",
                    target_slots=broad_targets,
                    decision_source="mode_c_hard_ask",
                    decision_reason=(
                        "Broad empty shopping inquiry lacks actionable anchor."
                    ),
                    reason_codes=["HARD_ASK_BROAD_EMPTY_INQUIRY"],
                    max_questions=MAX_MODE_C_QUESTIONS,
                )

    # 3. Hard Answer Guard: Turn cap reached OR all required slots resolved/declined
    if clarification_turns >= MAX_MODE_C_TURNS:
        if missing_req or declined_req:
            fb_slots = list(dict.fromkeys([*missing_req, *declined_req]))
            return ClarificationDecision(
                action="PROCEED",
                decision_source="fallback",
                decision_reason=(
                    f"Mode C turn cap ({MAX_MODE_C_TURNS}) reached; "
                    "applying deterministic fallback."
                ),
                reason_codes=["MODE_C_TURN_CAP_FALLBACK"],
                use_fallback=True,
                fallback_slots=fb_slots,
            )
        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_c_hard_answer",
            decision_reason="Turn cap reached and required slots resolved.",
            reason_codes=["HARD_ANSWER_COMPLETE"],
        )

    if not missing_req and not conflicting:
        if declined_req:
            return ClarificationDecision(
                action="PROCEED",
                decision_source="fallback",
                decision_reason=(
                    f"Required slots declined: {', '.join(declined_req)}; "
                    "applying deterministic fallback."
                ),
                reason_codes=["HARD_ANSWER_DECLINED_FALLBACK"],
                use_fallback=True,
                fallback_slots=declined_req,
            )
        if gray_zone_evaluator is not None and clarification_turns < MAX_MODE_C_TURNS:
            check_hc = getattr(gray_zone_evaluator, "check_hard_constraints", None)
            if callable(check_hc):
                hc_decision = check_hc(
                    intent=intent,
                    slots=slots,
                    clarification_turns=clarification_turns,
                )
                if hc_decision is not None:
                    return hc_decision
        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_c_hard_answer",
            decision_reason="All required slots resolved with no conflict.",
            reason_codes=["HARD_ANSWER_REQUIRED_COMPLETE"],
        )

    # 4. Gray Zone: Plug-in evaluator (Spec 2) or LLM evaluation (Spec 1)
    if gray_zone_evaluator is not None:
        plugin_decision = gray_zone_evaluator.evaluate(
            intent=intent,
            slots=slots,
            clarification_turns=clarification_turns,
        )
        if plugin_decision is not None:
            return plugin_decision

    gray_candidates: list[str] = list(missing_req)
    for cond_slot in get_conditional_slots(intent, slots):
        if (
            cond_slot not in gray_candidates
            and not slots.has(cond_slot)
            and not slots.is_declined(cond_slot)
        ):
            gray_candidates.append(cond_slot)

    return ClarificationDecision(
        action="GRAY_ZONE",
        target_slots=gray_candidates[:MAX_MODE_C_QUESTIONS],
        decision_source="mode_c_llm",
        decision_reason="Gray zone: evaluating 4 clarification conditions via LLM.",
        reason_codes=["GRAY_ZONE_LLM_EVAL"],
        max_questions=MAX_MODE_C_QUESTIONS,
    )


def policy_to_turn_policy(
    decision: ClarificationDecision,
    mode: ClarificationMode | str,
    intent: TurnIntent,
) -> TurnPolicy:
    """Translate a ClarificationDecision into tool-gating and prompt selection."""
    if intent == TurnIntent.NON_SHOPPING:
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.NON_SHOPPING)

    norm_mode = _normalize_mode(mode)
    if decision.action == "ASK":
        return TurnPolicy(gate_search=True, prompt_key=PromptKey.MODE_B_INTERVIEW)

    if norm_mode == ClarificationMode.IMMEDIATE_ANSWER:
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_A)

    if decision.action == "PROCEED":
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_B_OPEN)

    return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_C)


def resolve_turn_policy(
    mode: ClarificationMode | str,
    phase: ClarifyPhase,
    is_shopping_query: bool = True,
) -> TurnPolicy:
    """Resolve tool gating and prompt instructions (legacy phase-based interface)."""
    norm_mode = _normalize_mode(mode)

    if not is_shopping_query:
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.NON_SHOPPING)

    if norm_mode == ClarificationMode.ALWAYS_CLARIFY:
        if phase == ClarifyPhase.INTERVIEW:
            return TurnPolicy(gate_search=True, prompt_key=PromptKey.MODE_B_INTERVIEW)
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_B_OPEN)

    if norm_mode == ClarificationMode.IMMEDIATE_ANSWER:
        return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_A)

    return TurnPolicy(gate_search=False, prompt_key=PromptKey.MODE_C)


def next_phase(
    mode: ClarificationMode | str,
    current_phase: ClarifyPhase,
    gated_turns: int,
    needs_clarification: bool,
    is_shopping_query: bool = True,
) -> ClarifyPhase:
    """Transition phase one-way from INTERVIEW to OPEN on shopping turns."""
    norm_mode = _normalize_mode(mode)

    if (
        norm_mode != ClarificationMode.ALWAYS_CLARIFY
        or current_phase == ClarifyPhase.OPEN
        or not is_shopping_query
    ):
        return current_phase
    if not needs_clarification or gated_turns >= MAX_MODE_B_TURNS:
        return ClarifyPhase.OPEN
    return ClarifyPhase.INTERVIEW
