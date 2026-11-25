"""Pure functional clarification policy and turn state transitions."""

from dataclasses import dataclass
from enum import StrEnum

from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.services.session_manager import ClarifyPhase

MAX_GATED_TURNS = 1


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


def resolve_turn_policy(
    mode: ClarificationMode | str,
    phase: ClarifyPhase,
    is_shopping_query: bool = True,
) -> TurnPolicy:
    """Resolve tool gating and prompt instructions deterministically."""
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
    if needs_clarification or gated_turns >= MAX_GATED_TURNS:
        return ClarifyPhase.OPEN
    return ClarifyPhase.INTERVIEW
