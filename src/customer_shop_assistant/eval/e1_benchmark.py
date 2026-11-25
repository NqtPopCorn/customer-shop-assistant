"""Experiment E1 Benchmark: Clarification Modes, KG Ablations & Unified Reporting."""

import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.eval.dataset import E1Scenario, GoldClarificationLabel
from customer_shop_assistant.eval.e2e_benchmark import (
    E2EAggregateMetrics,
    E2EScenarioRunResult,
    ServiceHealthStatus,
    aggregate_e2e_results,
    resolve_active_model_name,
)
from customer_shop_assistant.eval.metrics import compute_slot_accuracy
from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
    resolve_color_id,
    resolve_occasion_id,
    resolve_style_id,
)
from customer_shop_assistant.schemas.chat import (
    ClarificationMode,
    OutfitItem,
    OutfitSuggestion,
)
from customer_shop_assistant.schemas.clarification import ClarificationDecision
from customer_shop_assistant.schemas.outfit_kg import (
    CandidateOutcome,
    ColorHarmonyAdvice,
    OutfitBlueprint,
    OutfitOntology,
    SlotImpactSignals,
)
from customer_shop_assistant.services.clarification_planner import (
    build_clarification_form,
)
from customer_shop_assistant.services.clarification_policy import (
    MAX_MODE_B_TURNS,
    MAX_MODE_C_CONFLICT_TURNS,
    MAX_MODE_C_QUESTIONS,
    MAX_MODE_C_TURNS,
    GrayZoneEvaluator,
    evaluate_clarification_policy,
)
from customer_shop_assistant.services.counterfactual_policy import (
    CandidateValueBuilder,
    CounterfactualClarificationPolicy,
    CounterfactualThresholds,
    _mean_pairwise_jaccard,
)
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
    Neo4jOutfitKGRepository,
    OutfitKGRepository,
)
from customer_shop_assistant.services.outfit_retrieval import OutfitRetrievalAndReranker
from customer_shop_assistant.services.slot_tracker import (
    BudgetValue,
    ConversationSlots,
    TurnIntent,
    extract_conversation_slots_with_lifecycle,
    get_missing_required_slots,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    SnapshotCatalogIndex,
    SnapshotProductRecord,
    get_snapshot_catalog_index,
)

AblationVariant = Literal["kg_plus_catalog", "catalog_only", "kg_only"]


class E1ScenarioResult(BaseModel):
    """Detailed Tier-1 telemetry & accuracy metrics for one scenario execution."""

    scenario_id: str
    split: Literal["dev", "test"]
    expected_intent: str
    detected_intent: str
    task_type: str
    mode: str
    variant: AblationVariant
    turns_to_completion: int
    clarification_asked: bool
    clarification_turns: int
    question_count: int
    asked_slots: list[str] = Field(default_factory=list)
    expected_clarify_slots: list[str] = Field(default_factory=list)
    slot_precision: float | None = None
    slot_recall: float | None = None
    slot_exact_match: bool | None = None
    decision_source: str
    decision_reason: str
    reason_codes: list[str] = Field(default_factory=list)
    evidence_rule_ids: list[str] = Field(default_factory=list)
    policy_compliant: bool
    policy_violations: list[str] = Field(default_factory=list)
    gold_label: GoldClarificationLabel
    gold_aligned: bool
    unnecessary_question: bool
    missed_necessary: bool
    constraint_satisfied: bool | None = None
    outfit_returned: bool | None = None
    outfit_rule_violation_rate: float | None = None
    mean_color_score: float | None = None
    color_harmony_rate: float | None = None
    applied_rule_ids: list[str] = Field(default_factory=list)
    outfits_summary: str = ""
    probe_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class E1AggregateMetrics(BaseModel):
    """Aggregated Tier-1 metrics across a set of E1 scenario results."""

    label: str
    total: int
    clarification_rate: float
    avg_question_count: float
    avg_turns_to_completion: float
    policy_compliance_rate: float
    gold_alignment_rate: float
    avg_slot_precision: float | None = None
    avg_slot_recall: float | None = None
    slot_exact_match_rate: float | None = None
    unnecessary_question_rate: float
    missed_necessary_rate: float
    constraint_satisfaction_rate: float | None = None
    outfit_returned_rate: float | None = None
    outfit_rule_violation_rate: float | None = None
    mean_color_score: float | None = None
    color_harmony_rate: float | None = None
    evidence_rule_coverage_rate: float = 0.0
    avg_probe_latency_ms: float = 0.0
    p95_probe_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0


class ColorRerankAblationMetrics(BaseModel):
    """Metrics for one color-reranking variant (v1.2 blueprint-only vs v1.3)."""

    variant: str
    total_scenarios: int
    outfit_returned_rate: float
    mean_color_score: float
    mean_pair_harmony: float
    mean_context_color_fit: float
    color_harmony_rate: float


class ColorRerankAblationResult(BaseModel):
    """Direct comparison of v1.2 (blueprint-only) vs v1.3 (color-aware 70/30)."""

    v12: ColorRerankAblationMetrics
    v13: ColorRerankAblationMetrics
    delta_color_score: float
    delta_harmony_rate: float


def load_e1_scenarios(
    filepath: Path | str = Path("eval/e1_scenarios.json"),
    split: Literal["dev", "test"] | None = None,
) -> list[E1Scenario]:
    """Load validated E1 controlled scenarios from JSON."""
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(f"E1 dataset not found at: {target}")
    with target.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = [E1Scenario.model_validate(item) for item in raw.get("scenarios", [])]
    if split is not None:
        return [s for s in items if s.split == split]
    return items


def load_frozen_thresholds(
    filepath: Path | str = Path("eval/frozen_thresholds.json"),
) -> tuple[CounterfactualThresholds, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load frozen thresholds and calibration sweeps if file exists."""
    target = Path(filepath)
    if not target.exists():
        return CounterfactualThresholds(), [], []
    raw = json.loads(target.read_text(encoding="utf-8"))
    thresh = CounterfactualThresholds.model_validate(raw.get("thresholds", {}))
    sweep = list(raw.get("sweep") or [])
    color_sweep = list(raw.get("color_sweep") or [])
    return thresh, sweep, color_sweep


def _resolve_anchor_category(
    slots: ConversationSlots,
    ontology: OutfitOntology,
    snapshot_index: SnapshotCatalogIndex | None = None,
) -> tuple[str | None, str | None, str | None, int | None]:
    top_raw = slots.value_of("top_category")
    bot_raw = slots.value_of("bottom_category")
    top_cat_id = resolve_category_id(top_raw, ontology) if top_raw else None
    bot_cat_id = resolve_category_id(bot_raw, ontology) if bot_raw else None
    anchor_pid = slots.value_of("product_id")
    cats_by_id = {c.id: c for c in ontology.categories}

    if anchor_pid is not None and not (top_cat_id or bot_cat_id) and snapshot_index:
        rec = snapshot_index.by_id.get(int(anchor_pid))
        if rec is not None:
            c_node = cats_by_id.get(rec.category_id)
            if c_node and c_node.role in ("TOP", "OUTERWEAR"):
                top_cat_id = rec.category_id
            elif c_node and c_node.role == "BOTTOM":
                bot_cat_id = rec.category_id

    anchor_cat_id = top_cat_id or bot_cat_id
    if anchor_cat_id is None:
        cat_raw = slots.value_of("category")
        if cat_raw:
            anchor_cat_id = resolve_category_id(str(cat_raw), ontology)
            if anchor_cat_id is None:
                c_low = str(cat_raw).strip().lower()
                if c_low == "accessory":
                    anchor_cat_id = "tui_xach"
                elif c_low == "dress":
                    anchor_cat_id = "vay_lien"
    return anchor_cat_id, top_cat_id, bot_cat_id, anchor_pid


def _resolve_color_slots(
    slots: ConversationSlots, ontology: OutfitOntology
) -> tuple[str | None, set[str]]:
    color_raw = (
        slots.value_of("top_color")
        or slots.value_of("bottom_color")
        or slots.value_of("color")
    )
    anchor_color_id = resolve_color_id(color_raw, ontology) if color_raw else None
    disliked_raw = slots.value_of("disliked_colors") or []
    disliked_ids = {
        cid
        for raw in disliked_raw
        if (cid := resolve_color_id(str(raw), ontology)) is not None
    }
    return anchor_color_id, disliked_ids


class CatalogOnlyCounterfactualPolicy:
    """Ablation evaluator: uses SnapshotCatalogIndex only (no KG blueprints)."""

    def __init__(
        self,
        snapshot_index: SnapshotCatalogIndex,
        *,
        ontology: OutfitOntology | None = None,
        thresholds: CounterfactualThresholds | None = None,
        candidate_builder: CandidateValueBuilder | None = None,
    ) -> None:
        self.snapshot_index = snapshot_index
        self.ontology = ontology or get_cached_ontology()
        self.thresholds = thresholds or CounterfactualThresholds()
        self.candidate_builder = candidate_builder or CandidateValueBuilder()
        self._cats_by_id = {c.id: c for c in self.ontology.categories}

    def _category_pools(
        self,
        top_category_id: str | None,
        bottom_category_id: str | None,
    ) -> tuple[list[str], list[str]]:
        top_cats = (
            [top_category_id]
            if top_category_id
            else [
                c.id
                for c in self.ontology.categories
                if c.supports_top_bottom_blueprint and c.role in ("TOP", "OUTERWEAR")
            ]
        )
        bot_cats = (
            [bottom_category_id]
            if bottom_category_id
            else [
                c.id
                for c in self.ontology.categories
                if c.supports_top_bottom_blueprint and c.role == "BOTTOM"
            ]
        )
        return top_cats, bot_cats

    def _has_catalog_color_match(
        self,
        *,
        target_color_id: str,
        target_gender: str | None,
        top_category_id: str | None,
        bottom_category_id: str | None,
        disliked_color_ids: set[str],
        min_price: float | None,
        max_price: float | None,
    ) -> bool:
        if target_color_id in disliked_color_ids:
            return False
        top_cats, bot_cats = self._category_pools(top_category_id, bottom_category_id)
        check_bot_side = bottom_category_id is not None and top_category_id is None
        top_cols = None if check_bot_side else {target_color_id}
        bot_cols = {target_color_id} if check_bot_side else None

        tops = [
            rec
            for cid in top_cats
            for rec in self.snapshot_index.query_products(
                cid,
                target_gender=target_gender,
                color_ids=top_cols,
                disliked_color_ids=disliked_color_ids,
                min_price=min_price,
                max_price=max_price,
            )
        ]
        bots = [
            rec
            for cid in bot_cats
            for rec in self.snapshot_index.query_products(
                cid,
                target_gender=target_gender,
                color_ids=bot_cols,
                disliked_color_ids=disliked_color_ids,
                min_price=min_price,
                max_price=max_price,
            )
        ]
        return bool(tops and bots)

    def _probe_raw_catalog(
        self,
        *,
        candidate_value: Any,
        target_gender: str | None,
        top_category_id: str | None,
        bottom_category_id: str | None,
        anchor_category_id: str | None,
        disliked_color_ids: set[str],
        min_price: float | None,
        max_price: float | None,
    ) -> CandidateOutcome:
        top_cats, bot_cats = self._category_pools(top_category_id, bottom_category_id)

        tops: list[SnapshotProductRecord] = []
        for cid in top_cats:
            tops.extend(
                self.snapshot_index.query_products(
                    cid,
                    target_gender=target_gender,
                    disliked_color_ids=disliked_color_ids,
                    min_price=min_price,
                    max_price=max_price,
                )
            )
        bots: list[SnapshotProductRecord] = []
        for cid in bot_cats:
            bots.extend(
                self.snapshot_index.query_products(
                    cid,
                    target_gender=target_gender,
                    disliked_color_ids=disliked_color_ids,
                    min_price=min_price,
                    max_price=max_price,
                )
            )

        if not tops or not bots:
            return CandidateOutcome(
                candidate_value=candidate_value,
                feasible_product_count=0,
                feasible_blueprint_count=0,
            )

        pairs: list[tuple[str, str]] = []
        top_ids: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()
        for t in tops[:4]:
            for b in bots[:4]:
                if t.deterministic_id == b.deterministic_id:
                    continue
                pair = (t.category_id, b.category_id)
                if anchor_category_id and anchor_category_id not in pair:
                    continue
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs.append(pair)
                token = f"{t.deterministic_id}+{b.deterministic_id}"
                if len(top_ids) < self.thresholds.top_k and token not in top_ids:
                    top_ids.append(token)

        unique_prods = {p.deterministic_id for p in (*tops, *bots)}
        return CandidateOutcome(
            candidate_value=candidate_value,
            feasible_product_count=len(unique_prods),
            feasible_blueprint_count=len(pairs),
            top_k_ids=top_ids,
            category_pairs=pairs,
            applied_rule_ids=[],
        )

    def check_hard_constraints(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Check catalog feasibility and requested color stockouts (no KG)."""
        if intent != TurnIntent.OUTFIT_ADVICE or clarification_turns >= 1:
            return None
        anchor_cat_id, top_cat_id, bot_cat_id, _ = _resolve_anchor_category(
            slots, self.ontology, self.snapshot_index
        )
        if anchor_cat_id and not (
            (c := self._cats_by_id.get(anchor_cat_id))
            and c.supports_top_bottom_blueprint
        ):
            return None
        anchor_color_id, disliked_ids = _resolve_color_slots(slots, self.ontology)
        budget_val = slots.value_of("budget")
        min_p = getattr(budget_val, "min_price", None) if budget_val else None
        max_p = getattr(budget_val, "max_price", None) if budget_val else None
        target_gender = slots.value_of("target_gender")

        outcome = self._probe_raw_catalog(
            candidate_value="baseline",
            target_gender=target_gender,
            top_category_id=top_cat_id,
            bottom_category_id=bot_cat_id,
            anchor_category_id=anchor_cat_id,
            disliked_color_ids=disliked_ids,
            min_price=min_p,
            max_price=max_p,
        )
        if outcome.feasible_blueprint_count == 0:
            ask_target = "budget" if budget_val else "target_gender"
            return ClarificationDecision(
                action="ASK",
                target_slots=[ask_target],
                max_questions=1,
                decision_source="mode_c_counterfactual",
                decision_reason="Catalog-only check found 0 feasible pairs.",
                reason_codes=["UNFULFILLABLE_HARD_CONSTRAINT"],
                evidence_rule_ids=[],
            )
        if anchor_color_id is not None and not self._has_catalog_color_match(
            target_color_id=anchor_color_id,
            target_gender=target_gender,
            top_category_id=top_cat_id,
            bottom_category_id=bot_cat_id,
            disliked_color_ids=disliked_ids,
            min_price=min_p,
            max_price=max_p,
        ):
            return ClarificationDecision(
                action="ASK",
                target_slots=["color"],
                max_questions=1,
                decision_source="mode_c_counterfactual",
                decision_reason=(
                    "Catalog-only check found 0 in-stock items "
                    f"for '{anchor_color_id}'."
                ),
                reason_codes=["UNFULFILLABLE_COLOR_CONSTRAINT"],
                evidence_rule_ids=[],
                question_intent="resolve_unfulfillable_color",
            )
        return None

    def evaluate(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Evaluate OUTFIT_ADVICE gray zone using catalog index only."""
        if intent != TurnIntent.OUTFIT_ADVICE:
            return None
        anchor_cat_id, top_cat_id, bot_cat_id, anchor_pid = _resolve_anchor_category(
            slots, self.ontology, self.snapshot_index
        )
        if anchor_cat_id is not None:
            cat_node = self._cats_by_id.get(anchor_cat_id)
            if cat_node is not None and not cat_node.supports_top_bottom_blueprint:
                return ClarificationDecision(
                    action="PROCEED",
                    decision_source="mode_c_counterfactual",
                    decision_reason=f"Out-of-blueprint category '{cat_node.name}'.",
                    reason_codes=["COUNTERFACTUAL_OUT_OF_SCOPE_PROCEED"],
                )

        hc = self.check_hard_constraints(intent, slots, clarification_turns)
        if hc is not None:
            return hc

        target_gender = slots.value_of("target_gender")
        _, disliked_ids = _resolve_color_slots(slots, self.ontology)
        budget_val = slots.value_of("budget")
        min_p = getattr(budget_val, "min_price", None) if budget_val else None
        max_p = getattr(budget_val, "max_price", None) if budget_val else None

        eligible: list[str] = []
        if not slots.has("target_gender") and not slots.is_declined("target_gender"):
            eligible.append("target_gender")
        if not slots.has("occasion") and not slots.is_declined("occasion"):
            eligible.append("occasion")

        has_anchor = bool(top_cat_id or bot_cat_id or anchor_pid)
        impact_signals: dict[str, SlotImpactSignals] = {}
        ask_candidates: list[tuple[tuple[int, int, float], str]] = []

        for slot_name in eligible:
            cands = (
                ["MALE", "FEMALE"]
                if slot_name == "target_gender"
                else ["office", "hangout", "party"]
            )
            outcomes = [
                self._probe_raw_catalog(
                    candidate_value=val,
                    target_gender=val
                    if slot_name == "target_gender"
                    else target_gender,
                    top_category_id=top_cat_id,
                    bottom_category_id=bot_cat_id,
                    anchor_category_id=anchor_cat_id,
                    disliked_color_ids=disliked_ids,
                    min_price=min_p,
                    max_price=max_p,
                )
                for val in cands
            ]
            viable = [
                o.feasible_blueprint_count >= self.thresholds.min_viable_outfits
                and o.feasible_product_count >= self.thresholds.min_viable_products
                for o in outcomes
            ]
            feas_chg = any(viable) and not all(viable)
            top_pairs = {o.category_pairs[0] for o in outcomes if o.category_pairs}
            cat_shift = len(top_pairs) > 1
            mean_jac = _mean_pairwise_jaccard([o.top_k_ids for o in outcomes])
            can_alt = (
                slot_name in ("target_gender", "occasion")
                and has_anchor
                and all(viable)
                and len(cands) <= self.thresholds.alternatives_max_branches
            )
            sig = SlotImpactSignals(
                feasibility_change=feas_chg,
                category_shift=cat_shift,
                mean_top_k_jaccard=mean_jac,
                can_cover_with_alternatives=can_alt,
            )
            impact_signals[slot_name] = sig
            if (
                feas_chg or cat_shift or mean_jac < self.thresholds.jaccard_threshold
            ) and not can_alt:
                ask_candidates.append(
                    (
                        (int(feas_chg), int(cat_shift), round(1.0 - mean_jac, 4)),
                        slot_name,
                    )
                )

        if ask_candidates and clarification_turns < 1:
            ask_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen = [slot for _, slot in ask_candidates[:2]]
            return ClarificationDecision(
                action="ASK",
                target_slots=chosen,
                max_questions=min(2, len(chosen)),
                decision_source="mode_c_counterfactual",
                decision_reason=f"Catalog-only sensitivity on {', '.join(chosen)}.",
                reason_codes=["COUNTERFACTUAL_HIGH_IMPACT_ASK"],
                impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
                evidence_rule_ids=[],
            )

        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_c_counterfactual",
            decision_reason="Catalog-only check found low sensitivity or alternatives.",
            reason_codes=["COUNTERFACTUAL_PROCEED"],
            impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
            evidence_rule_ids=[],
            use_fallback=not slots.has("target_gender"),
            fallback_slots=["target_gender"] if not slots.has("target_gender") else [],
        )


class KGOnlyCounterfactualPolicy:
    """Ablation evaluator: queries OutfitKGRepository without SnapshotCatalogIndex."""

    def __init__(
        self,
        kg_repo: OutfitKGRepository,
        *,
        ontology: OutfitOntology | None = None,
        thresholds: CounterfactualThresholds | None = None,
        candidate_builder: CandidateValueBuilder | None = None,
    ) -> None:
        self.kg_repo = kg_repo
        self.ontology = ontology or get_cached_ontology()
        self.thresholds = thresholds or CounterfactualThresholds()
        self.candidate_builder = candidate_builder or CandidateValueBuilder()

    def _probe_kg_only(
        self,
        *,
        candidate_value: Any,
        occasion_id: str | None,
        style_id: str | None,
        top_category_id: str | None,
        bottom_category_id: str | None,
        anchor_category_id: str | None,
    ) -> CandidateOutcome:
        bps: list[OutfitBlueprint] = self.kg_repo.get_outfit_blueprints(
            style_id=style_id,
            occasion_id=occasion_id,
            anchor_category_id=anchor_category_id,
            limit=10,
        )
        if top_category_id and bottom_category_id:
            bps = [
                bp
                for bp in bps
                if bp.top_category_id == top_category_id
                and bp.bottom_category_id == bottom_category_id
            ]
        pairs = [(bp.top_category_id, bp.bottom_category_id) for bp in bps]
        top_k_ids = [f"{t}+{b}" for t, b in pairs[: self.thresholds.top_k]]
        rules: list[str] = []
        for bp in bps:
            for rid in bp.applied_rule_ids:
                if rid not in rules:
                    rules.append(rid)
        return CandidateOutcome(
            candidate_value=candidate_value,
            feasible_product_count=len(bps) * 10,
            feasible_blueprint_count=len(bps),
            top_k_ids=top_k_ids,
            category_pairs=pairs,
            applied_rule_ids=rules,
        )

    def check_hard_constraints(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        del intent, slots, clarification_turns
        return None  # KG-only lacks catalog price/stock index

    def evaluate(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Evaluate OUTFIT_ADVICE gray zone using category KG only."""
        if intent != TurnIntent.OUTFIT_ADVICE:
            return None
        anchor_cat_id, top_cat_id, bot_cat_id, _ = _resolve_anchor_category(
            slots, self.ontology, None
        )
        if anchor_cat_id is not None:
            c_node = self.kg_repo.get_category(anchor_cat_id)
            if c_node is not None and not c_node.supports_top_bottom_blueprint:
                return ClarificationDecision(
                    action="PROCEED",
                    decision_source="mode_c_counterfactual",
                    decision_reason=f"Out-of-blueprint category '{c_node.name}'.",
                    reason_codes=["COUNTERFACTUAL_OUT_OF_SCOPE_PROCEED"],
                )

        occ_raw = slots.value_of("occasion")
        occ_id = resolve_occasion_id(occ_raw, self.ontology) if occ_raw else None
        style_raw = slots.value_of("style")
        style_id = resolve_style_id(style_raw, self.ontology) if style_raw else None

        eligible: list[str] = []
        if not slots.has("target_gender") and not slots.is_declined("target_gender"):
            eligible.append("target_gender")
        if not slots.has("occasion") and not slots.is_declined("occasion"):
            eligible.append("occasion")
        if (
            slots.has("target_gender")
            and not slots.has("style")
            and not slots.is_declined("style")
        ):
            eligible.append("style")

        has_anchor = bool(top_cat_id or bot_cat_id or slots.value_of("product_id"))
        impact_signals: dict[str, SlotImpactSignals] = {}
        evidence_rules: list[str] = []
        ask_candidates: list[tuple[tuple[int, int, float], str]] = []

        for slot_name in eligible:
            cands = self.candidate_builder.build_candidates(
                slot_name,
                anchor_category_id=anchor_cat_id,
                occasion_id=occ_id,
                kg_repo=self.kg_repo,
            )
            if len(cands) < 2:
                continue
            outcomes: list[CandidateOutcome] = []
            for val in cands:
                out = self._probe_kg_only(
                    candidate_value=val,
                    occasion_id=val if slot_name == "occasion" else occ_id,
                    style_id=val if slot_name == "style" else style_id,
                    top_category_id=top_cat_id,
                    bottom_category_id=bot_cat_id,
                    anchor_category_id=anchor_cat_id,
                )
                outcomes.append(out)
                for rid in out.applied_rule_ids:
                    if rid not in evidence_rules:
                        evidence_rules.append(rid)

            viable = [
                o.feasible_blueprint_count >= self.thresholds.min_viable_outfits
                for o in outcomes
            ]
            feas_chg = any(viable) and not all(viable)
            top_pairs = {o.category_pairs[0] for o in outcomes if o.category_pairs}
            cat_shift = len(top_pairs) > 1
            mean_jac = _mean_pairwise_jaccard([o.top_k_ids for o in outcomes])
            can_alt = (
                slot_name in ("target_gender", "occasion")
                and has_anchor
                and all(viable)
                and len(cands) <= self.thresholds.alternatives_max_branches
            )
            sig = SlotImpactSignals(
                feasibility_change=feas_chg,
                category_shift=cat_shift,
                mean_top_k_jaccard=mean_jac,
                can_cover_with_alternatives=can_alt,
            )
            impact_signals[slot_name] = sig
            if (
                feas_chg or cat_shift or mean_jac < self.thresholds.jaccard_threshold
            ) and not can_alt:
                ask_candidates.append(
                    (
                        (int(feas_chg), int(cat_shift), round(1.0 - mean_jac, 4)),
                        slot_name,
                    )
                )

        if ask_candidates and clarification_turns < 1:
            ask_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen = [slot for _, slot in ask_candidates[:2]]
            return ClarificationDecision(
                action="ASK",
                target_slots=chosen,
                max_questions=min(2, len(chosen)),
                decision_source="mode_c_counterfactual",
                decision_reason=f"KG-only blueprint impact for {', '.join(chosen)}.",
                reason_codes=["COUNTERFACTUAL_HIGH_IMPACT_ASK"],
                impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
                evidence_rule_ids=evidence_rules,
            )

        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_c_counterfactual",
            decision_reason="KG-only check found consistent blueprints.",
            reason_codes=["COUNTERFACTUAL_PROCEED"],
            impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
            evidence_rule_ids=evidence_rules,
            use_fallback=not slots.has("target_gender"),
            fallback_slots=["target_gender"] if not slots.has("target_gender") else [],
        )


def _open_kg_repo(
    backend: Literal["inmemory", "neo4j"], settings: Settings
) -> OutfitKGRepository:
    ont_dir = (settings.DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    if backend == "neo4j":
        repo = Neo4jOutfitKGRepository(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        repo.verify_connectivity()
        return repo
    return InMemoryOutfitKGRepository(ontology=ontology)


def _close_kg_repo(repo: Any) -> None:
    close_fn = getattr(repo, "close", None)
    if callable(close_fn):
        close_fn()


def _build_evaluator(
    variant: AblationVariant,
    kg_repo: OutfitKGRepository,
    snapshot_index: SnapshotCatalogIndex,
    ontology: OutfitOntology,
    thresholds: CounterfactualThresholds,
) -> GrayZoneEvaluator:
    if variant == "catalog_only":
        return CatalogOnlyCounterfactualPolicy(
            snapshot_index=snapshot_index, ontology=ontology, thresholds=thresholds
        )
    if variant == "kg_only":
        return KGOnlyCounterfactualPolicy(
            kg_repo=kg_repo, ontology=ontology, thresholds=thresholds
        )
    return CounterfactualClarificationPolicy(
        kg_repo=kg_repo,
        snapshot_index=snapshot_index,
        ontology=ontology,
        thresholds=thresholds,
    )


def _compute_kg_outfit_violation_rate(
    outfits: list[OutfitSuggestion],
    target_gender: str | None,
    ontology: OutfitOntology,
    snapshot_index: SnapshotCatalogIndex,
) -> float:
    if not outfits:
        return 0.0
    cats = {c.id: c for c in ontology.categories}
    violations = 0
    for o in outfits:
        violated = False
        top_id = resolve_category_id(o.top.category, ontology)
        bot_id = resolve_category_id(o.bottom.category, ontology)
        top_node = cats.get(top_id) if top_id else None
        bot_node = cats.get(bot_id) if bot_id else None
        if top_node is None or top_node.role not in ("TOP", "OUTERWEAR"):
            violated = True
        if bot_node is None or bot_node.role != "BOTTOM":
            violated = True
        if target_gender == "MALE" and bot_id == "chan_vay":
            violated = True
        if (
            top_node
            and bot_node
            and abs(top_node.base_formality - bot_node.base_formality) > 2
        ):
            violated = True
        if not o.top.in_stock or not o.bottom.in_stock:
            violated = True
        if target_gender in ("MALE", "FEMALE"):
            t_rec = snapshot_index.by_id.get(o.top.product_id)
            b_rec = snapshot_index.by_id.get(o.bottom.product_id)
            if (t_rec and not t_rec.matches_gender(target_gender)) or (
                b_rec and not b_rec.matches_gender(target_gender)
            ):
                violated = True
        if violated:
            violations += 1
    return round(violations / len(outfits), 4)


def _check_hard_constraints(
    scenario: E1Scenario,
    outfits: list[OutfitSuggestion],
    retrieval_error: str | None,
    ontology: OutfitOntology,
    snapshot_index: SnapshotCatalogIndex,
) -> bool:
    hc = scenario.hard_constraints
    if hc.get("out_of_scope"):
        return retrieval_error == "out_of_scope"
    if not outfits:
        return False

    req_gender = hc.get("target_gender")
    max_price = hc.get("max_price")
    req_top = hc.get("top_category")
    req_bot = hc.get("bottom_category")
    req_top_id = resolve_category_id(req_top, ontology) if req_top else None
    req_bot_id = resolve_category_id(req_bot, ontology) if req_bot else None

    for o in outfits:
        if req_gender in ("MALE", "FEMALE"):
            t_rec = snapshot_index.by_id.get(o.top.product_id)
            b_rec = snapshot_index.by_id.get(o.bottom.product_id)
            if (t_rec and not t_rec.matches_gender(req_gender)) or (
                b_rec and not b_rec.matches_gender(req_gender)
            ):
                return False
        if max_price is not None:
            if (o.top.price and o.top.price > max_price) or (
                o.bottom.price and o.bottom.price > max_price
            ):
                return False
        if req_top_id and resolve_category_id(o.top.category, ontology) != req_top_id:
            return False
        if (
            req_bot_id
            and resolve_category_id(o.bottom.category, ontology) != req_bot_id
        ):
            return False
    return True


def evaluate_scenario_policy(
    scenario: E1Scenario,
    *,
    mode: ClarificationMode,
    variant: AblationVariant,
    kg_repo: OutfitKGRepository,
    snapshot_index: SnapshotCatalogIndex,
    ontology: OutfitOntology,
    thresholds: CounterfactualThresholds,
) -> E1ScenarioResult:
    """Simulate a single multi-turn scenario under given mode & gray-zone variant."""
    t_total_start = time.perf_counter()
    evaluator = _build_evaluator(variant, kg_repo, snapshot_index, ontology, thresholds)

    history: list[str] = [scenario.question]
    asked_slots_history: list[str] = []
    declined_slots: set[str] = set()
    last_asked: list[str] = []
    total_questions = 0
    clar_turns = 0
    probe_latency_ms = 0.0
    violations: list[str] = []

    turn1_asked = False
    turn1_asked_slots: list[str] = []
    turn1_source = ""
    turn1_reason = ""
    turn1_reason_codes: list[str] = []
    turn1_evidence_rules: list[str] = []
    final_slots = ConversationSlots()
    final_intent = TurnIntent.PRODUCT_SEARCH

    for turn_idx in range(1, 5):
        slots, intent, _ = extract_conversation_slots_with_lifecycle(
            history,
            user_profile=scenario.user_profile,
            declined_slots=declined_slots,
            last_asked_slots=last_asked,
        )
        final_slots = slots
        final_intent = intent

        t0 = time.perf_counter()
        decision = evaluate_clarification_policy(
            mode,
            intent,
            slots,
            clarification_turns=clar_turns,
            gray_zone_evaluator=evaluator,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if turn_idx == 1:
            probe_latency_ms = round(dt_ms, 3)

        effective_action = decision.action
        ask_targets = list(decision.target_slots)
        if decision.action == "GRAY_ZONE":
            missing_req = get_missing_required_slots(intent, slots)
            if missing_req and clar_turns < MAX_MODE_C_TURNS:
                effective_action = "ASK"
                ask_targets = [s for s in missing_req if s in decision.target_slots][
                    :MAX_MODE_C_QUESTIONS
                ] or missing_req[:1]
            else:
                effective_action = "PROCEED"

        if turn_idx == 1:
            turn1_asked = effective_action == "ASK"
            turn1_source = decision.decision_source
            turn1_reason = decision.decision_reason
            turn1_reason_codes = list(decision.reason_codes)
            turn1_evidence_rules = list(decision.evidence_rule_ids)

            if intent == TurnIntent.NON_SHOPPING and turn1_asked:
                violations.append("NON_SHOPPING asked clarification")
            if mode == ClarificationMode.IMMEDIATE_ANSWER and turn1_asked:
                violations.append("IMMEDIATE_ANSWER asked clarification")
            if (
                mode == ClarificationMode.CONDITIONAL_CLARIFICATION
                and turn1_source == "mode_c_counterfactual"
                and turn1_asked
                and not turn1_reason_codes
            ):
                violations.append("Mode C counterfactual ASK missing reason_codes")

        if effective_action != "ASK":
            break

        allow_reconfirm = any(
            rc in ("UNFULFILLABLE_HARD_CONSTRAINT", "UNFULFILLABLE_COLOR_CONSTRAINT")
            for rc in decision.reason_codes
        )
        form = build_clarification_form(
            slots,
            intent,
            max_questions=decision.max_questions,
            target_slots=ask_targets or None,
            allow_provided=allow_reconfirm,
        )
        if not form.questions:
            break

        q_ids = [q.id for q in form.questions]
        if turn_idx == 1:
            turn1_asked_slots = list(q_ids)

        if mode == ClarificationMode.ALWAYS_CLARIFY and len(q_ids) > 3:
            violations.append(f"Mode B form exceeded 3 questions: {len(q_ids)}")
        if (
            mode == ClarificationMode.CONDITIONAL_CLARIFICATION
            and len(q_ids) > MAX_MODE_C_QUESTIONS
        ):
            violations.append(f"Mode C form exceeded 2 questions: {len(q_ids)}")

        total_questions += len(q_ids)
        asked_slots_history.extend(q_ids)
        clar_turns += 1
        last_asked = q_ids

        if scenario.user_answers.get("_decline") == "DECLINED":
            declined_slots.update(q_ids)
            reply_parts = ["Không quan trọng, bạn cứ gợi ý giúp mình"]
        else:
            reply_parts = [
                scenario.user_answers[qid]
                for qid in q_ids
                if qid in scenario.user_answers
            ]
            if not reply_parts:
                reply_parts = ["Bình thường thôi bạn"]
        history.append(" ".join(reply_parts))

    if mode == ClarificationMode.ALWAYS_CLARIFY and clar_turns > MAX_MODE_B_TURNS:
        violations.append(f"Mode B exceeded turn cap: {clar_turns}")
    if (
        mode == ClarificationMode.CONDITIONAL_CLARIFICATION
        and clar_turns > MAX_MODE_C_CONFLICT_TURNS
    ):
        violations.append(f"Mode C exceeded max conflict turn cap: {clar_turns}")

    # Gold alignment & slot accuracy on Turn 1
    if scenario.gold_label == "must_clarify":
        gold_aligned = turn1_asked
        missed_necessary = not turn1_asked
        unnecessary_question = False
    elif scenario.gold_label == "proceed":
        gold_aligned = not turn1_asked
        unnecessary_question = turn1_asked
        missed_necessary = False
    else:  # optional_clarify
        gold_aligned = True
        unnecessary_question = False
        missed_necessary = False

    if scenario.gold_label == "must_clarify" and scenario.expected_clarify_slots:
        slot_acc = compute_slot_accuracy(
            scenario.expected_clarify_slots, turn1_asked_slots
        )
    elif scenario.gold_label == "proceed" and turn1_asked_slots:
        slot_acc = compute_slot_accuracy([], turn1_asked_slots)
    else:
        slot_acc = compute_slot_accuracy([], [])

    # Run Outfit Retrieval & Grounding check for OUTFIT_ADVICE
    outfit_returned: bool | None = None
    outfit_rule_violation_rate: float | None = None
    mean_color_score: float | None = None
    color_harmony_rate: float | None = None
    constraint_satisfied: bool | None = None
    applied_rules: list[str] = []
    outfits_summary = ""

    if scenario.expected_intent == "OUTFIT_ADVICE":
        top_cat = final_slots.value_of("top_category")
        bot_cat = final_slots.value_of("bottom_category")
        raw_cat = final_slots.value_of("category")
        if top_cat is None and bot_cat is None:
            if raw_cat == "dress":
                top_cat = "Váy liền"
            elif raw_cat == "accessory":
                top_cat = "Túi xách"

        budget_val = final_slots.value_of("budget")
        min_p = (
            getattr(budget_val, "min_price", None)
            if isinstance(budget_val, BudgetValue)
            else None
        )
        max_p = (
            getattr(budget_val, "max_price", None)
            if isinstance(budget_val, BudgetValue)
            else None
        )

        retriever = OutfitRetrievalAndReranker(
            kg_repo=kg_repo, snapshot_index=snapshot_index, ontology=ontology
        )
        res = retriever.retrieve_outfits(
            target_gender=final_slots.value_of("target_gender"),
            occasion=final_slots.value_of("occasion"),
            style=final_slots.value_of("style"),
            top_category=top_cat,
            bottom_category=bot_cat,
            color=final_slots.value_of("color"),
            top_color=final_slots.value_of("top_color"),
            bottom_color=final_slots.value_of("bottom_color"),
            disliked_colors=final_slots.value_of("disliked_colors"),
            min_price=min_p,
            max_price=max_p,
            limit=3,
        )
        outfit_returned = len(res.outfits) > 0
        if res.outfits:
            outfit_rule_violation_rate = _compute_kg_outfit_violation_rate(
                res.outfits,
                final_slots.value_of("target_gender"),
                ontology,
                snapshot_index,
            )
            col_scores = [
                o.color_score_breakdown.color_score
                for o in res.outfits
                if o.color_score_breakdown is not None
            ]
            if col_scores:
                mean_color_score = round(sum(col_scores) / len(col_scores), 4)
            harm_cnt = sum(
                1
                for o in res.outfits
                if o.color_score_breakdown is not None
                and o.color_score_breakdown.harmony_kind is not None
            )
            color_harmony_rate = round(harm_cnt / len(res.outfits), 4)
            for o in res.outfits:
                for rid in o.applied_rule_ids:
                    if rid not in applied_rules:
                        applied_rules.append(rid)
            outfits_summary = " | ".join(
                f"[{o.top.product_id}] {o.top.name} ({o.top.color}) + "
                f"[{o.bottom.product_id}] {o.bottom.name} ({o.bottom.color})"
                for o in res.outfits[:2]
            )
        elif res.error == "out_of_scope":
            outfits_summary = "OUT_OF_SCOPE_NOTICE"

        constraint_satisfied = _check_hard_constraints(
            scenario, res.outfits, res.error, ontology, snapshot_index
        )

    total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 3)

    return E1ScenarioResult(
        scenario_id=scenario.id,
        split=scenario.split,
        expected_intent=scenario.expected_intent,
        detected_intent=final_intent.value,
        task_type=scenario.task_type,
        mode=mode.value,
        variant=variant,
        turns_to_completion=clar_turns + 1,
        clarification_asked=turn1_asked,
        clarification_turns=clar_turns,
        question_count=total_questions,
        asked_slots=asked_slots_history,
        expected_clarify_slots=list(scenario.expected_clarify_slots),
        slot_precision=slot_acc.precision,
        slot_recall=slot_acc.recall,
        slot_exact_match=slot_acc.exact_match,
        decision_source=turn1_source,
        decision_reason=turn1_reason,
        reason_codes=turn1_reason_codes,
        evidence_rule_ids=turn1_evidence_rules,
        policy_compliant=not violations,
        policy_violations=violations,
        gold_label=scenario.gold_label,
        gold_aligned=gold_aligned,
        unnecessary_question=unnecessary_question,
        missed_necessary=missed_necessary,
        constraint_satisfied=constraint_satisfied,
        outfit_returned=outfit_returned,
        outfit_rule_violation_rate=outfit_rule_violation_rate,
        mean_color_score=mean_color_score,
        color_harmony_rate=color_harmony_rate,
        applied_rule_ids=applied_rules,
        outfits_summary=outfits_summary,
        probe_latency_ms=probe_latency_ms,
        total_latency_ms=total_latency_ms,
    )


def _retrieve_outfits_v12_baseline(
    *,
    kg_repo: OutfitKGRepository,
    snapshot_index: SnapshotCatalogIndex,
    ontology: OutfitOntology,
    target_gender: str | None,
    occasion: str | None,
    style: str | None,
    top_category: str | None,
    bottom_category: str | None,
    color: str | None,
    top_color: str | None,
    bottom_color: str | None,
    disliked_colors: list[str] | None,
    min_price: float | None,
    max_price: float | None,
    limit: int = 3,
) -> list[OutfitSuggestion]:
    """Simulate v1.2 blueprint-only retrieval (first in-stock color, no rerank)."""
    top_cat_id = resolve_category_id(top_category, ontology)
    bot_cat_id = resolve_category_id(bottom_category, ontology)
    if top_cat_id is not None and bot_cat_id is None:
        t_node = kg_repo.get_category(top_cat_id)
        if t_node and t_node.role == "BOTTOM":
            bot_cat_id, top_cat_id = top_cat_id, None
            if bottom_color is None and top_color is not None:
                bottom_color, top_color = top_color, None

    occasion_id = resolve_occasion_id(occasion, ontology) if occasion else None
    style_id = resolve_style_id(style, ontology) if style else None
    disliked_ids = {
        cid
        for c in (disliked_colors or [])
        if (cid := resolve_color_id(c, ontology)) is not None
    }
    colors_by_id = {c.id: c for c in ontology.colors}
    palette = kg_repo.get_context_color_palette(
        style_id=style_id, occasion_id=occasion_id
    )
    harmony_lookup: dict[str, ColorHarmonyAdvice] = {}

    eff_top_col_id = resolve_color_id(top_color or color, ontology)
    eff_bot_col_id = resolve_color_id(bottom_color, ontology)
    if color and bot_cat_id is not None and top_cat_id is None and not bottom_color:
        eff_bot_col_id = resolve_color_id(color, ontology)
        eff_top_col_id = None

    anchor_cat_id = top_cat_id or bot_cat_id
    bps = kg_repo.get_outfit_blueprints(
        style_id=style_id,
        occasion_id=occasion_id,
        anchor_category_id=anchor_cat_id,
        limit=12,
    )
    if not bps and anchor_cat_id is not None:
        bps = kg_repo.get_outfit_blueprints(anchor_category_id=anchor_cat_id, limit=12)
    if top_cat_id and bot_cat_id:
        bps = [
            bp
            for bp in bps
            if bp.top_category_id == top_cat_id and bp.bottom_category_id == bot_cat_id
        ]
    if not bps:
        return []

    def _pick_v12_color(
        rec: SnapshotProductRecord, req_cid: str | None
    ) -> tuple[str, str, bool]:
        if req_cid and req_cid not in disliked_ids:
            if req_cid in rec.in_stock_color_ids:
                return req_cid, rec.color_name_by_id.get(req_cid, req_cid), True
        pool = sorted(c for c in rec.in_stock_color_ids if c not in disliked_ids)
        if pool:
            cid = pool[0]
            return cid, rec.color_name_by_id.get(cid, cid), True
        return "", "Tiêu chuẩn", rec.in_stock

    def _to_item(rec: SnapshotProductRecord, label: str, stock: bool) -> OutfitItem:
        cid = next((k for k, v in rec.color_name_by_id.items() if v == label), None)
        price = rec.color_price_map.get(cid, rec.min_price) if cid else rec.min_price
        return OutfitItem(
            product_id=rec.deterministic_id,
            name=rec.name,
            slug=rec.slug,
            category=rec.category_name,
            color=label,
            in_stock=stock,
            image_url=rec.primary_image_url,
            price=price,
        )

    eff_gender = target_gender.strip().upper() if target_gender else None
    if eff_gender is None and bot_cat_id == "chan_vay":
        eff_gender = "FEMALE"
    genders = [eff_gender] if eff_gender else ["FEMALE", "MALE"]

    picked: list[OutfitSuggestion] = []
    used_pairs: set[tuple[int, int]] = set()
    for g_filter in genders:
        for bp in bps:
            tops = snapshot_index.query_products(
                bp.top_category_id,
                target_gender=g_filter,
                color_ids={eff_top_col_id} if eff_top_col_id else None,
                disliked_color_ids=disliked_ids,
                min_price=min_price,
                max_price=max_price,
            ) or snapshot_index.query_products(
                bp.top_category_id,
                target_gender=g_filter,
                disliked_color_ids=disliked_ids,
                min_price=min_price,
                max_price=max_price,
            )
            bots = snapshot_index.query_products(
                bp.bottom_category_id,
                target_gender=g_filter,
                color_ids={eff_bot_col_id} if eff_bot_col_id else None,
                disliked_color_ids=disliked_ids,
                min_price=min_price,
                max_price=max_price,
            ) or snapshot_index.query_products(
                bp.bottom_category_id,
                target_gender=g_filter,
                disliked_color_ids=disliked_ids,
                min_price=min_price,
                max_price=max_price,
            )
            if not tops or not bots:
                continue
            for t_rec in tops[:2]:
                for b_rec in bots[:2]:
                    if t_rec.deterministic_id == b_rec.deterministic_id:
                        continue
                    pair_key = (t_rec.deterministic_id, b_rec.deterministic_id)
                    if pair_key in used_pairs:
                        continue
                    used_pairs.add(pair_key)
                    t_cid, t_lbl, t_stock = _pick_v12_color(t_rec, eff_top_col_id)
                    b_cid, b_lbl, b_stock = _pick_v12_color(b_rec, eff_bot_col_id)
                    brk, _ = OutfitRetrievalAndReranker._score_color_pair(
                        top_cid=t_cid,
                        top_label=t_lbl,
                        bot_cid=b_cid,
                        bot_label=b_lbl,
                        palette=palette,
                        colors_by_id=colors_by_id,
                        harmony_lookup=harmony_lookup,
                        get_harmony_fn=kg_repo.get_color_harmony,
                        req_top_cid=eff_top_col_id,
                        req_bot_cid=eff_bot_col_id,
                        fav_ids=set(),
                    )
                    picked.append(
                        OutfitSuggestion(
                            top=_to_item(t_rec, t_lbl, t_stock),
                            bottom=_to_item(b_rec, b_lbl, b_stock),
                            score=round(bp.score, 4),
                            blueprint_score=round(bp.score, 4),
                            color_score_breakdown=brk,
                            reasons=list(bp.reasons),
                            applied_rule_ids=list(bp.applied_rule_ids),
                        )
                    )
                    if len(picked) >= limit:
                        return picked
    return picked[:limit]


def run_color_rerank_ablation(
    scenarios: list[E1Scenario],
    *,
    kg_repo: OutfitKGRepository,
    snapshot_index: SnapshotCatalogIndex,
    ontology: OutfitOntology,
) -> ColorRerankAblationResult:
    """Compare v1.2 (blueprint-only) vs v1.3 (color-aware 70/30) on OUTFIT_ADVICE."""
    retriever = OutfitRetrievalAndReranker(
        kg_repo=kg_repo, snapshot_index=snapshot_index, ontology=ontology
    )
    target_scenarios = [
        sc
        for sc in scenarios
        if sc.expected_intent == "OUTFIT_ADVICE"
        and not sc.hard_constraints.get("out_of_scope")
    ]

    def _collect(use_v13: bool) -> ColorRerankAblationMetrics:
        ret_cnt = 0
        cs_vals: list[float] = []
        ph_vals: list[float] = []
        cf_vals: list[float] = []
        harm_cnt = 0
        outfit_cnt = 0

        for sc in target_scenarios:
            history = [sc.question]
            for k, v in sc.user_answers.items():
                if k != "_decline" and v:
                    history.append(v)
            slots, _, _ = extract_conversation_slots_with_lifecycle(
                history, user_profile=sc.user_profile
            )
            top_cat = slots.value_of("top_category")
            bot_cat = slots.value_of("bottom_category")
            bval = slots.value_of("budget")
            min_p = (
                getattr(bval, "min_price", None)
                if isinstance(bval, BudgetValue)
                else None
            )
            max_p = (
                getattr(bval, "max_price", None)
                if isinstance(bval, BudgetValue)
                else None
            )

            if use_v13:
                outfits = retriever.retrieve_outfits(
                    target_gender=slots.value_of("target_gender"),
                    occasion=slots.value_of("occasion"),
                    style=slots.value_of("style"),
                    top_category=top_cat,
                    bottom_category=bot_cat,
                    color=slots.value_of("color"),
                    top_color=slots.value_of("top_color"),
                    bottom_color=slots.value_of("bottom_color"),
                    disliked_colors=slots.value_of("disliked_colors"),
                    min_price=min_p,
                    max_price=max_p,
                    limit=3,
                ).outfits
            else:
                outfits = _retrieve_outfits_v12_baseline(
                    kg_repo=kg_repo,
                    snapshot_index=snapshot_index,
                    ontology=ontology,
                    target_gender=slots.value_of("target_gender"),
                    occasion=slots.value_of("occasion"),
                    style=slots.value_of("style"),
                    top_category=top_cat,
                    bottom_category=bot_cat,
                    color=slots.value_of("color"),
                    top_color=slots.value_of("top_color"),
                    bottom_color=slots.value_of("bottom_color"),
                    disliked_colors=slots.value_of("disliked_colors"),
                    min_price=min_p,
                    max_price=max_p,
                    limit=3,
                )

            if outfits:
                ret_cnt += 1
            for o in outfits:
                outfit_cnt += 1
                if o.color_score_breakdown is not None:
                    cs_vals.append(o.color_score_breakdown.color_score)
                    ph_vals.append(o.color_score_breakdown.pair_harmony)
                    cf_vals.append(o.color_score_breakdown.context_color_fit)
                    if o.color_score_breakdown.harmony_kind is not None:
                        harm_cnt += 1

        n_sc = len(target_scenarios)
        return ColorRerankAblationMetrics(
            variant="v1.3 Color-aware (70/30)" if use_v13 else "v1.2 Blueprint-only",
            total_scenarios=n_sc,
            outfit_returned_rate=round(ret_cnt / n_sc * 100.0, 1) if n_sc else 0.0,
            mean_color_score=round(sum(cs_vals) / len(cs_vals), 4) if cs_vals else 0.0,
            mean_pair_harmony=round(sum(ph_vals) / len(ph_vals), 4) if ph_vals else 0.0,
            mean_context_color_fit=round(sum(cf_vals) / len(cf_vals), 4)
            if cf_vals
            else 0.0,
            color_harmony_rate=round(harm_cnt / outfit_cnt * 100.0, 1)
            if outfit_cnt
            else 0.0,
        )

    v12_m = _collect(False)
    v13_m = _collect(True)
    return ColorRerankAblationResult(
        v12=v12_m,
        v13=v13_m,
        delta_color_score=round(v13_m.mean_color_score - v12_m.mean_color_score, 4),
        delta_harmony_rate=round(
            v13_m.color_harmony_rate - v12_m.color_harmony_rate, 1
        ),
    )


def aggregate_e1_results(
    label: str, results: list[E1ScenarioResult]
) -> E1AggregateMetrics:
    """Aggregate Tier-1 E1 metrics cleanly across a list of scenario runs."""
    tot = len(results)
    if tot == 0:
        return E1AggregateMetrics(
            label=label,
            total=0,
            clarification_rate=0.0,
            avg_question_count=0.0,
            avg_turns_to_completion=0.0,
            policy_compliance_rate=0.0,
            gold_alignment_rate=0.0,
            unnecessary_question_rate=0.0,
            missed_necessary_rate=0.0,
        )

    probes = sorted(r.probe_latency_ms for r in results)
    p95_idx = min(tot - 1, int(math.ceil(0.95 * tot)) - 1)
    outfit_res = [r for r in results if r.outfit_returned is not None]
    viol_res = [
        r.outfit_rule_violation_rate
        for r in outfit_res
        if r.outfit_rule_violation_rate is not None
    ]
    col_res = [r.mean_color_score for r in outfit_res if r.mean_color_score is not None]
    harm_res = [
        r.color_harmony_rate for r in outfit_res if r.color_harmony_rate is not None
    ]
    cs_res = [r for r in results if r.constraint_satisfied is not None]
    ev_cov = sum(1 for r in results if r.evidence_rule_ids or r.applied_rule_ids)

    slot_p = [r.slot_precision for r in results if r.slot_precision is not None]
    slot_r = [r.slot_recall for r in results if r.slot_recall is not None]
    slot_ex = [r.slot_exact_match for r in results if r.slot_exact_match is not None]

    return E1AggregateMetrics(
        label=label,
        total=tot,
        clarification_rate=round(
            sum(1 for r in results if r.clarification_asked) / tot * 100.0, 2
        ),
        avg_question_count=round(sum(r.question_count for r in results) / tot, 2),
        avg_turns_to_completion=round(
            sum(r.turns_to_completion for r in results) / tot, 2
        ),
        policy_compliance_rate=round(
            sum(1 for r in results if r.policy_compliant) / tot * 100.0, 2
        ),
        gold_alignment_rate=round(
            sum(1 for r in results if r.gold_aligned) / tot * 100.0, 2
        ),
        avg_slot_precision=(
            round(sum(slot_p) / len(slot_p) * 100.0, 1) if slot_p else None
        ),
        avg_slot_recall=(
            round(sum(slot_r) / len(slot_r) * 100.0, 1) if slot_r else None
        ),
        slot_exact_match_rate=(
            round(sum(1 for x in slot_ex if x) / len(slot_ex) * 100.0, 1)
            if slot_ex
            else None
        ),
        unnecessary_question_rate=round(
            sum(1 for r in results if r.unnecessary_question) / tot * 100.0, 2
        ),
        missed_necessary_rate=round(
            sum(1 for r in results if r.missed_necessary) / tot * 100.0, 2
        ),
        constraint_satisfaction_rate=(
            round(
                sum(1 for r in cs_res if r.constraint_satisfied) / len(cs_res) * 100.0,
                2,
            )
            if cs_res
            else None
        ),
        outfit_returned_rate=(
            round(
                sum(1 for r in outfit_res if r.outfit_returned)
                / len(outfit_res)
                * 100.0,
                2,
            )
            if outfit_res
            else None
        ),
        outfit_rule_violation_rate=(
            round(sum(viol_res) / len(viol_res) * 100.0, 2) if viol_res else None
        ),
        mean_color_score=(round(sum(col_res) / len(col_res), 4) if col_res else None),
        color_harmony_rate=(
            round(sum(harm_res) / len(harm_res) * 100.0, 2) if harm_res else None
        ),
        evidence_rule_coverage_rate=round(ev_cov / tot * 100.0, 2),
        avg_probe_latency_ms=round(sum(probes) / tot, 2),
        p95_probe_latency_ms=round(probes[p95_idx], 2),
        avg_total_latency_ms=round(sum(r.total_latency_ms for r in results) / tot, 2),
    )


def calibrate_thresholds_on_dev(
    dev_scenarios: list[E1Scenario],
    *,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
    settings: Settings | None = None,
    output_path: Path | None = None,
) -> tuple[CounterfactualThresholds, list[dict[str, Any]]]:
    """Tune & freeze CounterfactualThresholds strictly on the Dev split."""
    cfg = settings or get_settings()
    ont_dir = (cfg.DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    snapshot_index = get_snapshot_catalog_index(cfg)
    kg_repo = _open_kg_repo(backend, cfg)

    sweep: list[dict[str, Any]] = []
    color_sweep: list[dict[str, Any]] = []
    best_thresh = CounterfactualThresholds()
    best_key = (-1.0, -999.0, -1.0, -999.0)

    try:
        # Stage A: Structural Jaccard & Min Viable Outfits sweep
        for jt in (0.25, 0.35, 0.45, 0.55, 0.65):
            for mvo in (1, 2):
                thresh = CounterfactualThresholds(
                    jaccard_threshold=jt,
                    min_viable_outfits=mvo,
                )
                results = [
                    evaluate_scenario_policy(
                        sc,
                        mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
                        variant="kg_plus_catalog",
                        kg_repo=kg_repo,
                        snapshot_index=snapshot_index,
                        ontology=ontology,
                        thresholds=thresh,
                    )
                    for sc in dev_scenarios
                ]
                agg = aggregate_e1_results(f"jt={jt}_mvo={mvo}", results)
                sweep.append(
                    {
                        "jaccard_threshold": jt,
                        "min_viable_outfits": mvo,
                        "gold_alignment_rate": agg.gold_alignment_rate,
                        "avg_slot_recall": agg.avg_slot_recall,
                        "missed_necessary_rate": agg.missed_necessary_rate,
                        "unnecessary_question_rate": agg.unnecessary_question_rate,
                        "policy_compliance_rate": agg.policy_compliance_rate,
                        "avg_probe_latency_ms": agg.avg_probe_latency_ms,
                    }
                )
                rank_key = (
                    agg.gold_alignment_rate,
                    -(agg.missed_necessary_rate + agg.unnecessary_question_rate),
                    agg.avg_slot_recall or 0.0,
                    -abs(jt - 0.45),
                )
                if rank_key > best_key:
                    best_key = rank_key
                    best_thresh = thresh

        # Stage B: Color Quality & Color Gain sweep at Stage A winner
        best_color_key = (-1.0, -999.0, -1.0, -999.0)
        stage_a_jt = best_thresh.jaccard_threshold
        stage_a_mvo = best_thresh.min_viable_outfits
        for mcq in (0.65, 0.72, 0.80):
            for cgt in (0.08, 0.12, 0.18):
                thresh = CounterfactualThresholds(
                    jaccard_threshold=stage_a_jt,
                    min_viable_outfits=stage_a_mvo,
                    min_color_quality=mcq,
                    color_gain_threshold=cgt,
                )
                results = [
                    evaluate_scenario_policy(
                        sc,
                        mode=ClarificationMode.CONDITIONAL_CLARIFICATION,
                        variant="kg_plus_catalog",
                        kg_repo=kg_repo,
                        snapshot_index=snapshot_index,
                        ontology=ontology,
                        thresholds=thresh,
                    )
                    for sc in dev_scenarios
                ]
                agg = aggregate_e1_results(f"mcq={mcq}_cgt={cgt}", results)
                color_sweep.append(
                    {
                        "min_color_quality": mcq,
                        "color_gain_threshold": cgt,
                        "gold_alignment_rate": agg.gold_alignment_rate,
                        "avg_slot_recall": agg.avg_slot_recall,
                        "missed_necessary_rate": agg.missed_necessary_rate,
                        "unnecessary_question_rate": agg.unnecessary_question_rate,
                        "avg_probe_latency_ms": agg.avg_probe_latency_ms,
                    }
                )
                c_key = (
                    agg.gold_alignment_rate,
                    -(agg.missed_necessary_rate + agg.unnecessary_question_rate),
                    agg.avg_slot_recall or 0.0,
                    -abs(mcq - 0.72) - abs(cgt - 0.12),
                )
                if c_key > best_color_key:
                    best_color_key = c_key
                    best_thresh = thresh
    finally:
        _close_kg_repo(kg_repo)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frozen_at": datetime.now(UTC).isoformat(),
            "dev_count": len(dev_scenarios),
            "backend": backend,
            "thresholds": best_thresh.model_dump(),
            "sweep": sweep,
            "color_sweep": color_sweep,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return best_thresh, sweep


def run_e1_policy_benchmark(
    scenarios: list[E1Scenario],
    *,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
    thresholds: CounterfactualThresholds | None = None,
    include_ablation: bool = True,
    settings: Settings | None = None,
    output_jsonl_path: Path | None = None,
) -> tuple[dict[str, E1AggregateMetrics], list[E1ScenarioResult]]:
    """Run primary 3-mode comparison and optional Mode C ablations on scenarios."""
    cfg = settings or get_settings()
    ont_dir = (cfg.DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    snapshot_index = get_snapshot_catalog_index(cfg)
    thresh = thresholds or CounterfactualThresholds()
    kg_repo = _open_kg_repo(backend, cfg)

    runs: list[tuple[str, ClarificationMode, AblationVariant]] = [
        ("IMMEDIATE_ANSWER", ClarificationMode.IMMEDIATE_ANSWER, "kg_plus_catalog"),
        ("ALWAYS_CLARIFY", ClarificationMode.ALWAYS_CLARIFY, "kg_plus_catalog"),
        (
            "CONDITIONAL_CLARIFICATION (KG+Catalog)",
            ClarificationMode.CONDITIONAL_CLARIFICATION,
            "kg_plus_catalog",
        ),
    ]
    if include_ablation:
        runs.extend(
            [
                (
                    "CONDITIONAL_CLARIFICATION (Catalog-only)",
                    ClarificationMode.CONDITIONAL_CLARIFICATION,
                    "catalog_only",
                ),
                (
                    "CONDITIONAL_CLARIFICATION (KG-only)",
                    ClarificationMode.CONDITIONAL_CLARIFICATION,
                    "kg_only",
                ),
            ]
        )

    all_results: list[E1ScenarioResult] = []
    aggregates: dict[str, E1AggregateMetrics] = {}

    try:
        for label, mode, variant in runs:
            group_results = [
                evaluate_scenario_policy(
                    sc,
                    mode=mode,
                    variant=variant,
                    kg_repo=kg_repo,
                    snapshot_index=snapshot_index,
                    ontology=ontology,
                    thresholds=thresh,
                )
                for sc in scenarios
            ]
            all_results.extend(group_results)
            aggregates[label] = aggregate_e1_results(label, group_results)
    finally:
        _close_kg_repo(kg_repo)

    if output_jsonl_path is not None:
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(item.model_dump_json() + "\n")

    return aggregates, all_results


def export_tier2_judge_sheet(
    results: list[E1ScenarioResult],
    scenarios: list[E1Scenario],
    output_path: Path = Path("eval/e1_tier2_judge_sheet.csv"),
    sample_size: int = 10,
    e2e_results: list[E2EScenarioRunResult] | None = None,
) -> int:
    """Export stratified 3-mode CSV with UTF-8 BOM for Tier-2 human judging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if e2e_results:
        rows = e2e_results
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source",
                    "suite",
                    "scenario_id",
                    "task_type",
                    "mode",
                    "turns_to_completion",
                    "question",
                    "final_reply",
                    "clarification_asked",
                    "asked_slots",
                    "decision_source",
                    "reason_codes",
                    "tools_used",
                    "applied_rule_ids",
                    "products_summary",
                    "outfits_summary",
                    "grounding_ok",
                    "recommendation_relevance_1_5",
                    "unnecessary_question_0_1",
                    "missed_clarification_0_1",
                    "task_completion_0_1",
                    "explanation_faithfulness_1_5",
                    "judge_notes",
                ]
            )
            for er in rows:
                writer.writerow(
                    [
                        "tier2_e2e_llm",
                        er.suite,
                        er.scenario_id,
                        er.task_type,
                        er.mode,
                        er.turns_to_completion,
                        er.question,
                        er.final_reply,
                        er.clarification_asked_turn1,
                        ",".join(er.asked_slots_turn1),
                        er.decision_source_turn1 or "",
                        ",".join(er.reason_codes_turn1),
                        ",".join(er.tools_used),
                        ",".join(er.applied_rule_ids),
                        er.products_summary,
                        er.outfits_summary,
                        er.grounding_ok,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        return len(rows)

    scen_map = {s.id: s for s in scenarios}
    selected_ids: list[str] = []
    by_task: dict[str, list[str]] = {}
    for s in scenarios:
        by_task.setdefault(s.task_type, []).append(s.id)

    while len(selected_ids) < sample_size:
        progressed = False
        for task_key in (
            "open",
            "anchor",
            "color_implicit",
            "color_ambiguous",
            "color_constraint",
            "search",
            "size",
            "body_fit",
            "non_shopping",
        ):
            pool = by_task.get(task_key, [])
            if pool and len(selected_ids) < sample_size:
                selected_ids.append(pool.pop(0))
                progressed = True
        if not progressed:
            break

    id_set = set(selected_ids)
    filtered = [
        r for r in results if r.scenario_id in id_set and r.variant == "kg_plus_catalog"
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "scenario_id",
                "task_type",
                "mode",
                "question",
                "clarification_asked",
                "asked_slots",
                "decision_source",
                "decision_reason",
                "applied_rule_ids",
                "outfits_summary",
                "recommendation_relevance_1_5",
                "unnecessary_question_0_1",
                "missed_clarification_0_1",
                "task_completion_0_1",
                "explanation_faithfulness_1_5",
                "judge_notes",
            ]
        )
        for r in filtered:
            sc = scen_map.get(r.scenario_id)
            writer.writerow(
                [
                    "tier1_offline",
                    r.scenario_id,
                    r.task_type,
                    r.mode,
                    sc.question if sc else "",
                    r.clarification_asked,
                    ",".join(r.asked_slots),
                    r.decision_source,
                    r.decision_reason,
                    ",".join(r.applied_rule_ids or r.evidence_rule_ids),
                    r.outfits_summary,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return len(filtered)


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1f}%" if val is not None else "N/A"


def _fmt_slot_triple(m: E1AggregateMetrics) -> str:
    p = f"{m.avg_slot_precision:.0f}%" if m.avg_slot_precision is not None else "-"
    r = f"{m.avg_slot_recall:.0f}%" if m.avg_slot_recall is not None else "-"
    ex = (
        f"{m.slot_exact_match_rate:.0f}%"
        if m.slot_exact_match_rate is not None
        else "-"
    )
    return f"{p}/{r}/{ex}"


def write_unified_experiment_report(
    aggregates: dict[str, E1AggregateMetrics],
    all_results: list[E1ScenarioResult],
    *,
    thresholds: CounterfactualThresholds,
    dev_sweep: list[dict[str, Any]] | None = None,
    color_sweep: list[dict[str, Any]] | None = None,
    color_ablation: ColorRerankAblationResult | None = None,
    e2e_contrastive: list[E2EScenarioRunResult] | None = None,
    e2e_stratified: list[E2EScenarioRunResult] | None = None,
    service_status: ServiceHealthStatus | None = None,
    settings: Settings | None = None,
    backend: str = "inmemory",
    dev_count: int = 28,
    test_count: int = 56,
    output_path: Path = Path("docs/research/experiment_report.md"),
) -> None:
    """Write unified, data-driven research experiment report (Tier 1 + Tier 2)."""
    cfg = settings or get_settings()
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    active_model = resolve_active_model_name(cfg)
    primary_labels = (
        "IMMEDIATE_ANSWER",
        "ALWAYS_CLARIFY",
        "CONDITIONAL_CLARIFICATION (KG+Catalog)",
    )

    be_note = (
        f"`{'OK' if service_status.main_be_ok else 'OFFLINE'}` "
        f"({service_status.main_be_detail})"
        if service_status
        else "Chưa kiểm tra"
    )
    neo_note = (
        (
            f"`{'OK' if service_status.neo4j_ok else 'OFFLINE'}` "
            f"({service_status.neo4j_detail})"
        )
        if service_status and service_status.neo4j_ok is not None
        else f"`{backend}`"
    )

    lines: list[str] = [
        "# Báo cáo Thực nghiệm Nghiên cứu Trợ lý Chatbot Thời trang",
        "",
        f"- **Thời điểm chạy**: {now_str}",
        f"- **LLM Cấu hình**: `{cfg.LLM_PROVIDER}` / `{active_model}`",
        f"- **KG Backend**: `{backend}` (Neo4j: {neo_note}) | **Main BE**: {be_note}",
        "- **Snapshot**: `data/hm_seed_1000.json` (1,000 SP, 14 categories, 33 màu)",
        (
            f"- **Dataset**: `eval/e1_scenarios.json` "
            f"(**{dev_count} Dev** / **{test_count} Frozen Test**) "
            "+ `eval/e1_contrastive_s1_s10.json` (12 nhánh S1–S10)"
        ),
        (
            "- **Ngưỡng đóng băng (`CounterfactualThresholds`)**: "
            f"`jaccard_threshold={thresholds.jaccard_threshold}`, "
            f"`min_viable_outfits={thresholds.min_viable_outfits}`, "
            f"`min_color_quality={thresholds.min_color_quality}`, "
            f"`color_gain_threshold={thresholds.color_gain_threshold}`, "
            f"`alternatives_max_branches={thresholds.alternatives_max_branches}`"
        ),
        "",
        "---",
        "",
        "## 1. Tầng 1 — So sánh 3 Chiến lược trên Tập Frozen Test",
        "",
        (
            "| Chế độ | Clarify (%) | Qs/Req | Turns | Policy (%) | Gold Align (%) | "
            "Slot P/R/Exact | Unnecessary (%) | Missed (%) | Constraint (%) | "
            "Color Score | Harmony (%) | Probe TB (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for lbl in primary_labels:
        m = aggregates.get(lbl)
        if m is None:
            continue
        col_str = (
            f"{m.mean_color_score:.3f}" if m.mean_color_score is not None else "N/A"
        )
        lines.append(
            f"| `{m.label}` | {m.clarification_rate:.1f}% | "
            f"{m.avg_question_count:.2f} | {m.avg_turns_to_completion:.2f} | "
            f"{m.policy_compliance_rate:.1f}% | {m.gold_alignment_rate:.1f}% | "
            f"{_fmt_slot_triple(m)} | "
            f"{m.unnecessary_question_rate:.1f}% | "
            f"{m.missed_necessary_rate:.1f}% | "
            f"{_fmt_pct(m.constraint_satisfaction_rate)} | "
            f"{col_str} | "
            f"{_fmt_pct(m.color_harmony_rate)} | "
            f"{m.avg_probe_latency_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "### 1.1 Phân rã theo nhóm tác vụ "
            "(`OPEN` vs `ANCHOR` vs `COLOR` vs `Other`)",
            "",
            (
                "| Chế độ | Nhóm | Số ca | Clarify (%) | "
                "Gold Align (%) | Slot Exact (%) | Unnecessary (%) | Missed (%) |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lbl in primary_labels:
        for group_name, pred in (
            ("OPEN", lambda r: r.task_type == "open"),
            ("ANCHOR", lambda r: r.task_type == "anchor"),
            ("COLOR", lambda r: r.task_type.startswith("color_")),
            (
                "Other Intents",
                lambda r: (
                    r.task_type not in ("open", "anchor")
                    and not r.task_type.startswith("color_")
                ),
            ),
        ):
            sub = [
                r
                for r in all_results
                if (
                    r.variant == "kg_plus_catalog"
                    and (
                        (lbl == "IMMEDIATE_ANSWER" and r.mode == "IMMEDIATE_ANSWER")
                        or (lbl == "ALWAYS_CLARIFY" and r.mode == "ALWAYS_CLARIFY")
                        or (
                            lbl.startswith("CONDITIONAL_CLARIFICATION")
                            and r.mode == "CONDITIONAL_CLARIFICATION"
                        )
                    )
                    and pred(r)
                )
            ]
            sm = aggregate_e1_results(group_name, sub)
            lines.append(
                f"| `{lbl}` | {group_name} | {sm.total} | "
                f"{sm.clarification_rate:.1f}% | {sm.gold_alignment_rate:.1f}% | "
                f"{_fmt_pct(sm.slot_exact_match_rate)} | "
                f"{sm.unnecessary_question_rate:.1f}% | "
                f"{sm.missed_necessary_rate:.1f}% |"
            )

    ablation_labels = (
        "CONDITIONAL_CLARIFICATION (Catalog-only)",
        "CONDITIONAL_CLARIFICATION (KG-only)",
        "CONDITIONAL_CLARIFICATION (KG+Catalog)",
    )
    if any(lbl in aggregates for lbl in ablation_labels[:2]) or color_ablation:
        lines.extend(["", "## 2. Nghiên cứu Ablation", ""])
    if any(lbl in aggregates for lbl in ablation_labels[:2]):
        lines.extend(
            [
                "### 2.1 Ablation Chiến lược Hỏi lại (`CONDITIONAL_CLARIFICATION`)",
                "",
                (
                    "| Biến thể | Clarify (%) | Gold Align (%) | Slot Exact (%) | "
                    "Unnecessary (%) | Missed (%) | Evidence Rule Cov (%) | "
                    "Probe TB (ms) |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for lbl in ablation_labels:
            m = aggregates.get(lbl)
            if m is None:
                continue
            lines.append(
                f"| `{m.label}` | {m.clarification_rate:.1f}% | "
                f"{m.gold_alignment_rate:.1f}% | "
                f"{_fmt_pct(m.slot_exact_match_rate)} | "
                f"{m.unnecessary_question_rate:.1f}% | "
                f"{m.missed_necessary_rate:.1f}% | "
                f"{m.evidence_rule_coverage_rate:.1f}% | "
                f"{m.avg_probe_latency_ms:.2f} |"
            )

    if color_ablation is not None:
        v12 = color_ablation.v12
        v13 = color_ablation.v13
        lines.extend(
            [
                "",
                "### 2.2 Ablation Color-aware Reranking (v1.2 Blueprint-only vs v1.3)",
                "",
                (
                    "| Phiên bản | Số ca | Outfit Returned (%) | Mean Color Score | "
                    "Pair Harmony | Context Color Fit | Color Harmony Rate (%) |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    f"| `{v12.variant}` | {v12.total_scenarios} | "
                    f"{v12.outfit_returned_rate:.1f}% | {v12.mean_color_score:.4f} | "
                    f"{v12.mean_pair_harmony:.4f} | "
                    f"{v12.mean_context_color_fit:.4f} | "
                    f"{v12.color_harmony_rate:.1f}% |"
                ),
                (
                    f"| `{v13.variant}` | {v13.total_scenarios} | "
                    f"{v13.outfit_returned_rate:.1f}% | {v13.mean_color_score:.4f} | "
                    f"{v13.mean_pair_harmony:.4f} | "
                    f"{v13.mean_context_color_fit:.4f} | "
                    f"{v13.color_harmony_rate:.1f}% |"
                ),
                "",
                (
                    f"- **Chênh lệch (v1.3 − v1.2)**: "
                    f"`Δ Mean Color Score = {color_ablation.delta_color_score:+.4f}`, "
                    "`Δ Color Harmony Rate = "
                    f"{color_ablation.delta_harmony_rate:+.1f}%`"
                ),
            ]
        )

    if dev_sweep or color_sweep:
        lines.extend(
            ["", "## 3. Hiệu chỉnh Ngưỡng trên Tập Dev (`calibrate_thresholds_on_dev`)"]
        )
    if dev_sweep:
        lines.extend(
            [
                "",
                "### 3.1 Stage A — `jaccard_threshold` × `min_viable_outfits`",
                "",
                (
                    "| `jaccard_threshold` | `min_viable_outfits` | "
                    "Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dev_sweep:
            lines.append(
                f"| {row['jaccard_threshold']} | {row['min_viable_outfits']} | "
                f"{row['gold_alignment_rate']:.1f}% | "
                f"{row['missed_necessary_rate']:.1f}% | "
                f"{row['unnecessary_question_rate']:.1f}% | "
                f"{row['avg_probe_latency_ms']:.2f} |"
            )
    if color_sweep:
        lines.extend(
            [
                "",
                "### 3.2 Stage B — `min_color_quality` × `color_gain_threshold`",
                "",
                (
                    "| `min_color_quality` | `color_gain_threshold` | "
                    "Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in color_sweep:
            lines.append(
                f"| {row['min_color_quality']} | {row['color_gain_threshold']} | "
                f"{row['gold_alignment_rate']:.1f}% | "
                f"{row['missed_necessary_rate']:.1f}% | "
                f"{row['unnecessary_question_rate']:.1f}% | "
                f"{row['avg_probe_latency_ms']:.2f} |"
            )

    if dev_sweep and all(r.get("gold_alignment_rate") == 100.0 for r in dev_sweep):
        lines.extend(
            [
                "",
                (
                    "> *Ghi chú độ vững (Robustness)*: Toàn bộ lưới hiệu chuẩn "
                    "Dev đạt **100.0%** `Gold Align`, cho thấy ngưỡng quyết định "
                    "vững trên tập Dev; cơ chế tie-break chọn cấu hình mặc định "
                    "trung tâm."
                ),
            ]
        )

    all_e2e = [*(e2e_contrastive or []), *(e2e_stratified or [])]
    lines.extend(["", "## 4. Tầng 2 — Kiểm thử End-to-End Đa lượt với Local LLM", ""])
    if not all_e2e:
        lines.append(
            "- Chạy ở chế độ `--tier1-only`; chưa chạy Tầng 2 E2E LLM trong phiên này."
        )
    else:
        if e2e_contrastive:
            lines.extend(
                [
                    "### 4.1 Ma trận Đối chứng 3 Chế độ (S1–S10)",
                    "",
                    (
                        "| Scenario | Mode A (`IMMEDIATE`) | Mode B (`ALWAYS`) | "
                        "Mode C (`CONDITIONAL`) |"
                    ),
                    "|---|---|---|---|",
                ]
            )
            by_sid: dict[str, dict[str, E2EScenarioRunResult]] = {}
            for item in e2e_contrastive:
                by_sid.setdefault(item.scenario_id, {})[item.mode] = item

            def _cell(rec: E2EScenarioRunResult | None) -> str:
                if rec is None:
                    return "-"
                tag = "PASS" if rec.passed else "FAIL"
                act = "ASK" if rec.clarification_asked_turn1 else "PROCEED"
                src = rec.decision_source_turn1 or "-"
                return f"{tag} ({act}, `{src}`, {rec.turns_to_completion}t)"

            for sid, mode_map in by_sid.items():
                lines.append(
                    f"| `{sid}` | "
                    f"{_cell(mode_map.get('IMMEDIATE_ANSWER'))} | "
                    f"{_cell(mode_map.get('ALWAYS_CLARIFY'))} | "
                    f"{_cell(mode_map.get('CONDITIONAL_CLARIFICATION'))} |"
                )

        def _append_e2e_summary_table(
            heading: str, subset: list[E2EScenarioRunResult]
        ) -> None:
            e2e_agg = aggregate_e2e_results(subset)
            lines.extend(
                [
                    "",
                    heading,
                    "",
                    (
                        "| Chế độ | Số ca | Errors | Pass (%) | Turns TB | "
                        "Clarify (%) | Mode Comp (%) | Grounding (%) | "
                        "Task Comp (%) | Tokens (In/Out/Tot) | Latency TB (ms) |"
                    ),
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for m_name in (
                "IMMEDIATE_ANSWER",
                "ALWAYS_CLARIFY",
                "CONDITIONAL_CLARIFICATION",
            ):
                em: E2EAggregateMetrics | None = e2e_agg.get(m_name)
                if em is None:
                    continue
                lines.append(
                    f"| `{em.mode}` | {em.total} | {em.error_count} | "
                    f"{em.pass_rate:.1f}% | "
                    f"{em.avg_turns:.2f} | {em.clarification_rate:.1f}% | "
                    f"{em.mode_compliance_rate:.1f}% | {em.grounding_rate:.1f}% | "
                    f"{em.task_completion_rate:.1f}% | "
                    f"{em.avg_input_tokens:.0f}/{em.avg_output_tokens:.0f}/"
                    f"{em.avg_total_tokens:.0f} | {em.avg_latency_ms:.1f} |"
                )

        if e2e_contrastive:
            _append_e2e_summary_table(
                "### 4.2 Tổng hợp E2E — Bộ Đối chứng S1–S10 (Stress-test)",
                e2e_contrastive,
            )
        if e2e_stratified:
            _append_e2e_summary_table(
                "### 4.3 Tổng hợp E2E — Bộ Stratified Frozen Test (Đại diện)",
                e2e_stratified,
            )

    misses = [
        r
        for r in all_results
        if r.variant == "kg_plus_catalog"
        and r.mode == "CONDITIONAL_CLARIFICATION"
        and not r.gold_aligned
    ]
    slot_misses = [
        r
        for r in all_results
        if r.variant == "kg_plus_catalog"
        and r.mode == "CONDITIONAL_CLARIFICATION"
        and r.slot_exact_match is False
    ]
    e2e_failures = [r for r in all_e2e if not r.passed]

    lines.extend(["", "## 5. Phân tích Sai lệch tự động (Error Analysis)", ""])
    if not misses and not slot_misses and not e2e_failures:
        lines.append("- **0** sai lệch ở cả Tầng 1 và Tầng 2 trong lần chạy này.")
    else:
        if misses:
            lines.append(
                "### 5.1 Sai lệch Nhãn Gold ở Tầng 1 (`CONDITIONAL_CLARIFICATION`)"
            )
            for r in misses:
                lines.append(
                    f"- `{r.scenario_id}` (gold=`{r.gold_label}`, "
                    f"asked={r.clarification_asked}, "
                    f"source=`{r.decision_source}`, reason_codes=`{r.reason_codes}`)"
                )
        if slot_misses:
            lines.append("")
            lines.append(
                "### 5.2 Sai lệch Slot hỏi ở Tầng 1 (`CONDITIONAL_CLARIFICATION`)"
            )
            for r in slot_misses:
                lines.append(
                    f"- `{r.scenario_id}`: asked=`{r.asked_slots}`, "
                    f"expected=`{r.expected_clarify_slots}` "
                    f"(P={r.slot_precision}, R={r.slot_recall})"
                )
        if e2e_failures:
            lines.append("")
            lines.append("### 5.3 Lỗi Runtime / Grounding / Tool bắt được ở Tầng 2 E2E")
            for ef in e2e_failures:
                lines.append(
                    f"- `[{ef.suite}] {ef.scenario_id}` (`{ef.mode}`): "
                    f"{'; '.join(ef.failure_reasons)}"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_e1_markdown_report(
    aggregates: dict[str, E1AggregateMetrics],
    all_results: list[E1ScenarioResult],
    *,
    thresholds: CounterfactualThresholds,
    dev_sweep: list[dict[str, Any]] | None = None,
    e2e_samples: list[E2EScenarioRunResult] | None = None,
    backend: str = "inmemory",
    dev_count: int = 28,
    test_count: int = 56,
    output_path: Path = Path("docs/research/experiment_report.md"),
) -> None:
    """Compatibility wrapper writing unified experiment report."""
    write_unified_experiment_report(
        aggregates,
        all_results,
        thresholds=thresholds,
        dev_sweep=dev_sweep,
        e2e_stratified=e2e_samples,
        backend=backend,
        dev_count=dev_count,
        test_count=test_count,
        output_path=output_path,
    )
