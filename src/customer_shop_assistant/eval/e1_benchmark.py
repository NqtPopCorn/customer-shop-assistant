"""Experiment E1 Benchmark: Clarification Modes, KG Ablations & Neo4j Live Eval."""

import asyncio
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
    resolve_occasion_id,
    resolve_style_id,
)
from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ClarificationMode,
    OutfitSuggestion,
    UserProfile,
)
from customer_shop_assistant.schemas.clarification import ClarificationDecision
from customer_shop_assistant.schemas.outfit_kg import (
    CandidateOutcome,
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

GoldClarificationLabel = Literal["must_clarify", "optional_clarify", "proceed"]
AblationVariant = Literal["kg_plus_catalog", "catalog_only", "kg_only"]


class E1Scenario(BaseModel):
    """Single controlled scenario in the E1 Dev/Test benchmark."""

    id: str
    split: Literal["dev", "test"]
    expected_intent: str
    task_type: str
    question: str
    user_profile: UserProfile | None = None
    gold_label: GoldClarificationLabel
    gold_rationale: str = Field(..., min_length=1)
    expected_clarify_slots: list[str] = Field(default_factory=list)
    user_answers: dict[str, str] = Field(default_factory=dict)
    hard_constraints: dict[str, Any] = Field(default_factory=dict)


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
    unnecessary_question_rate: float
    missed_necessary_rate: float
    constraint_satisfaction_rate: float | None = None
    outfit_returned_rate: float | None = None
    outfit_rule_violation_rate: float | None = None
    evidence_rule_coverage_rate: float = 0.0
    avg_probe_latency_ms: float = 0.0
    p95_probe_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0


class E1E2ESampleResult(BaseModel):
    """End-to-end live LLM + Neo4j sample turn record."""

    scenario_id: str
    task_type: str
    mode: str
    question: str
    clarification_asked: bool
    decision_source: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    outfit_count: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


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
        self._inmem_occ_fallback = InMemoryOutfitKGRepository(self.ontology)

    def _probe_raw_catalog(
        self,
        *,
        candidate_value: Any,
        target_gender: str | None,
        top_category_id: str | None,
        bottom_category_id: str | None,
        anchor_category_id: str | None,
        min_price: float | None,
        max_price: float | None,
    ) -> CandidateOutcome:
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

        tops: list[SnapshotProductRecord] = []
        for cid in top_cats:
            tops.extend(
                self.snapshot_index.query_products(
                    cid,
                    target_gender=target_gender,
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
        budget_val = slots.value_of("budget")
        outcome = self._probe_raw_catalog(
            candidate_value="baseline",
            target_gender=slots.value_of("target_gender"),
            top_category_id=top_cat_id,
            bottom_category_id=bot_cat_id,
            anchor_category_id=anchor_cat_id,
            min_price=getattr(budget_val, "min_price", None) if budget_val else None,
            max_price=getattr(budget_val, "max_price", None) if budget_val else None,
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
        return None

    def evaluate(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
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

        form = build_clarification_form(
            slots,
            intent,
            max_questions=decision.max_questions,
            target_slots=ask_targets or None,
        )
        if not form.questions:
            break

        q_ids = [q.id for q in form.questions]
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

    # Gold alignment on Turn 1
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

    # Run Outfit Retrieval & Grounding check for OUTFIT_ADVICE
    outfit_returned: bool | None = None
    outfit_rule_violation_rate: float | None = None
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
        applied_rule_ids=applied_rules,
        outfits_summary=outfits_summary,
        probe_latency_ms=probe_latency_ms,
        total_latency_ms=total_latency_ms,
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
    cs_res = [r for r in results if r.constraint_satisfied is not None]
    ev_cov = sum(1 for r in results if r.evidence_rule_ids or r.applied_rule_ids)

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
    best_thresh = CounterfactualThresholds()
    best_key = (-1.0, -999.0, -999.0)

    try:
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
                        "missed_necessary_rate": agg.missed_necessary_rate,
                        "unnecessary_question_rate": agg.unnecessary_question_rate,
                        "policy_compliance_rate": agg.policy_compliance_rate,
                        "avg_probe_latency_ms": agg.avg_probe_latency_ms,
                    }
                )
                rank_key = (
                    agg.gold_alignment_rate,
                    -(agg.missed_necessary_rate + agg.unnecessary_question_rate),
                    -abs(jt - 0.45),
                )
                if rank_key > best_key:
                    best_key = rank_key
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
) -> int:
    """Export stratified 3-mode sample CSV with UTF-8 BOM for Tier-2 rating."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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


async def run_e2e_llm_sample(
    scenarios: list[E1Scenario],
    *,
    limit: int = 3,
    backend: Literal["inmemory", "neo4j"] = "neo4j",
    settings: Settings | None = None,
    timeout_s: float = 45.0,
) -> list[E1E2ESampleResult]:
    """Run a small stratified live LLM + Neo4j end-to-end verification sample."""
    from customer_shop_assistant.services.agent_service import ShopAssistantAgent

    if limit <= 0:
        return []

    cfg = (settings or get_settings()).model_copy(update={"OUTFIT_KG_BACKEND": backend})
    picks: list[E1Scenario] = []
    for preferred_task in ("open", "anchor", "size", "non_shopping"):
        match = next(
            (s for s in scenarios if s.task_type == preferred_task and s not in picks),
            None,
        )
        if match is not None:
            picks.append(match)
        if len(picks) >= limit:
            break
    for s in scenarios:
        if len(picks) >= limit:
            break
        if s not in picks:
            picks.append(s)

    modes = (
        ClarificationMode.IMMEDIATE_ANSWER,
        ClarificationMode.ALWAYS_CLARIFY,
        ClarificationMode.CONDITIONAL_CLARIFICATION,
    )
    agent = ShopAssistantAgent(settings=cfg)
    samples: list[E1E2ESampleResult] = []

    for sc in picks:
        for mode in modes:
            req = ChatRequest(
                message=sc.question,
                clarification_mode=mode,
                user_profile=sc.user_profile,
                session_id=f"e1-e2e-{sc.id}-{mode.value}",
            )
            t0 = time.perf_counter()
            try:
                res = await asyncio.wait_for(agent.chat(req), timeout=timeout_s)
                lat = round((time.perf_counter() - t0) * 1000.0, 2)
                raw_usage = (
                    res.debug_meta.usage_metadata
                    if isinstance(res.debug_meta, BaseModel)
                    else (
                        res.debug_meta.get("usage_metadata")
                        if isinstance(res.debug_meta, dict)
                        else None
                    )
                )
                usage = (
                    raw_usage.model_dump()
                    if isinstance(raw_usage, BaseModel)
                    else (raw_usage or {})
                )
                dec_src = (
                    res.debug_meta.decision_source
                    if isinstance(res.debug_meta, BaseModel)
                    else (
                        res.debug_meta.get("decision_source")
                        if isinstance(res.debug_meta, dict)
                        else None
                    )
                )
                samples.append(
                    E1E2ESampleResult(
                        scenario_id=sc.id,
                        task_type=sc.task_type,
                        mode=mode.value,
                        question=sc.question,
                        clarification_asked=res.clarification_asked,
                        decision_source=dec_src,
                        tools_used=list(res.tools_used),
                        outfit_count=len(res.outfits),
                        latency_ms=lat,
                        prompt_tokens=int(
                            usage.get("input_tokens", usage.get("prompt_tokens", 0))
                        ),
                        completion_tokens=int(
                            usage.get(
                                "output_tokens", usage.get("completion_tokens", 0)
                            )
                        ),
                        total_tokens=int(usage.get("total_tokens", 0)),
                    )
                )
            except Exception as exc:
                lat = round((time.perf_counter() - t0) * 1000.0, 2)
                samples.append(
                    E1E2ESampleResult(
                        scenario_id=sc.id,
                        task_type=sc.task_type,
                        mode=mode.value,
                        question=sc.question,
                        clarification_asked=False,
                        latency_ms=lat,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    return samples


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1f}%" if val is not None else "N/A"


def write_e1_markdown_report(
    aggregates: dict[str, E1AggregateMetrics],
    all_results: list[E1ScenarioResult],
    *,
    thresholds: CounterfactualThresholds,
    dev_sweep: list[dict[str, Any]] | None = None,
    e2e_samples: list[E1E2ESampleResult] | None = None,
    backend: str = "inmemory",
    dev_count: int = 24,
    test_count: int = 48,
    output_path: Path = Path("docs/research/e1_kg_clarification_report.md"),
) -> None:
    """Write structured E1 benchmark report (SPEC v1.1 & v1.2)."""
    now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    primary_labels = (
        "IMMEDIATE_ANSWER",
        "ALWAYS_CLARIFY",
        "CONDITIONAL_CLARIFICATION (KG+Catalog)",
    )

    lines: list[str] = [
        "# Báo cáo Thực nghiệm E1: Clarification Strategy trên Outfit KG",
        "",
        f"- **Thời điểm chạy**: {now_str}",
        f"- **KG Backend**: `{backend}` (`data/outfit_kg/ontology/`)",
        "- **Snapshot**: `data/hm_seed_1000.json` (1,000 SP, 14 categories, 33 màu)",
        (
            f"- **Dataset**: `eval/e1_scenarios.json` "
            f"(**{dev_count} Dev** / **{test_count} Frozen Test**)"
        ),
        (
            "- **Ngưỡng đóng băng (`CounterfactualThresholds`)**: "
            f"`jaccard_threshold={thresholds.jaccard_threshold}`, "
            f"`min_viable_outfits={thresholds.min_viable_outfits}`, "
            f"`alternatives_max_branches={thresholds.alternatives_max_branches}`"
        ),
        "",
        "---",
        "",
        "## 1. So sánh 3 Chiến lược trên Tập Frozen Test",
        "",
        (
            "| Chế độ | Clarify (%) | Qs/Req | Turns | Policy (%) | Gold Align (%) | "
            "Unnecessary (%) | Missed (%) | Constraint (%) | Rule Viol (%) | "
            "Probe TB (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for lbl in primary_labels:
        m = aggregates.get(lbl)
        if m is None:
            continue
        lines.append(
            f"| `{m.label}` | {m.clarification_rate:.1f}% | "
            f"{m.avg_question_count:.2f} | {m.avg_turns_to_completion:.2f} | "
            f"{m.policy_compliance_rate:.1f}% | {m.gold_alignment_rate:.1f}% | "
            f"{m.unnecessary_question_rate:.1f}% | "
            f"{m.missed_necessary_rate:.1f}% | "
            f"{_fmt_pct(m.constraint_satisfaction_rate)} | "
            f"{_fmt_pct(m.outfit_rule_violation_rate)} | "
            f"{m.avg_probe_latency_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "### 1.1 Phân rã theo nhóm tác vụ (`OPEN` vs `ANCHOR` vs `Other Intents`)",
            "",
        ]
    )
    lines.append(
        "| Chế độ | Nhóm | Số ca | Clarify (%) | "
        "Gold Align (%) | Unnecessary (%) | Missed (%) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for lbl in primary_labels:
        for group_name, pred in (
            ("OPEN", lambda r: r.task_type == "open"),
            ("ANCHOR", lambda r: r.task_type == "anchor"),
            ("Other Intents", lambda r: r.task_type not in ("open", "anchor")),
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
                f"{sm.unnecessary_question_rate:.1f}% | "
                f"{sm.missed_necessary_rate:.1f}% |"
            )

    ablation_labels = (
        "CONDITIONAL_CLARIFICATION (Catalog-only)",
        "CONDITIONAL_CLARIFICATION (KG-only)",
        "CONDITIONAL_CLARIFICATION (KG+Catalog)",
    )
    if any(lbl in aggregates for lbl in ablation_labels[:2]):
        lines.extend(
            [
                "",
                "## 2. Nghiên cứu Ablation (`CONDITIONAL_CLARIFICATION`)",
                "",
                (
                    "| Biến thể | Clarify (%) | Gold Align (%) | "
                    "Unnecessary (%) | Missed (%) | Evidence Rule Cov (%) | "
                    "Probe TB (ms) |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for lbl in ablation_labels:
            m = aggregates.get(lbl)
            if m is None:
                continue
            lines.append(
                f"| `{m.label}` | {m.clarification_rate:.1f}% | "
                f"{m.gold_alignment_rate:.1f}% | "
                f"{m.unnecessary_question_rate:.1f}% | "
                f"{m.missed_necessary_rate:.1f}% | "
                f"{m.evidence_rule_coverage_rate:.1f}% | "
                f"{m.avg_probe_latency_ms:.2f} |"
            )

    if dev_sweep:
        lines.extend(
            [
                "",
                "## 3. Hiệu chỉnh Ngưỡng trên Tập Dev (`calibrate_thresholds_on_dev`)",
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

    if e2e_samples:
        lines.extend(
            [
                "",
                "## 4. Mẫu Kiểm thử End-to-End (Local LLM + Live Neo4j)",
                "",
                (
                    "| Scenario | Mode | Clarified | Source | Tools | "
                    "Outfits | Latency (ms) | Total Tokens | Status |"
                ),
                "|---|---|---:|---|---|---:|---:|---:|---|",
            ]
        )
        for s in e2e_samples:
            status = s.error or "OK"
            lines.append(
                f"| `{s.scenario_id}` | `{s.mode}` | {s.clarification_asked} | "
                f"`{s.decision_source or '-'}` | `{','.join(s.tools_used) or '-'}` | "
                f"{s.outfit_count} | {s.latency_ms:.1f} | {s.total_tokens} | {status} |"
            )

    misses = [
        r
        for r in all_results
        if r.variant == "kg_plus_catalog"
        and r.mode == "CONDITIONAL_CLARIFICATION"
        and not r.gold_aligned
    ]
    lines.extend(
        ["", "## 5. Phân tích Sai lệch (`CONDITIONAL_CLARIFICATION KG+Catalog`)", ""]
    )
    if not misses:
        lines.append(
            "- **0** trường hợp lệch nhãn gold (`gold_alignment_rate = 100.0%`)."
        )
    else:
        for r in misses:
            lines.append(
                f"- `{r.scenario_id}` (gold=`{r.gold_label}`, "
                f"asked={r.clarification_asked}, "
                f"source=`{r.decision_source}`, reason_codes=`{r.reason_codes}`)"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
