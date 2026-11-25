"""Single-Slot Counterfactual Clarification Policy for OUTFIT_ADVICE Gray Zone."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
    resolve_color_id,
    resolve_occasion_id,
    resolve_style_id,
)
from customer_shop_assistant.schemas.clarification import ClarificationDecision
from customer_shop_assistant.schemas.outfit_kg import (
    CandidateOutcome,
    OutfitBlueprint,
    OutfitOntology,
    SlotImpactSignals,
)
from customer_shop_assistant.services.outfit_kg_repository import OutfitKGRepository
from customer_shop_assistant.services.slot_tracker import (
    BudgetValue,
    ConversationSlots,
    TurnIntent,
)
from customer_shop_assistant.services.snapshot_catalog_index import SnapshotCatalogIndex

logger = logging.getLogger(__name__)


class CounterfactualThresholds(BaseModel):
    """Tunable counterfactual decision thresholds (Dev-tuned, Test-frozen)."""

    jaccard_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_viable_outfits: int = Field(default=1, ge=1)
    min_viable_products: int = Field(default=2, ge=1)
    alternatives_max_branches: int = Field(default=3, ge=1)
    top_k: int = Field(default=5, ge=1)
    wide_pool_product_threshold: int = Field(default=120, ge=1)


class CandidateValueBuilder:
    """Generates small, ontology-grounded candidate sets V_s per eligible slot."""

    def build_candidates(
        self,
        slot_name: str,
        *,
        anchor_category_id: str | None,
        occasion_id: str | None,
        kg_repo: OutfitKGRepository,
    ) -> list[Any]:
        """Return candidate values V_s for a single slot."""
        if slot_name == "target_gender":
            return ["MALE", "FEMALE"]

        if slot_name == "occasion":
            if anchor_category_id:
                occs = kg_repo.get_occasions_for_category(anchor_category_id)[:3]
                if occs:
                    return occs
            return ["office", "hangout", "party"]

        if slot_name == "style":
            if occasion_id:
                styles = kg_repo.get_styles_for_occasion(occasion_id)[:3]
                if styles:
                    return styles
            return ["smart_casual", "casual", "minimalist"]

        if slot_name == "budget":
            return [
                BudgetValue(max_price=300_000.0, raw_text="<= 300k"),
                BudgetValue(
                    min_price=300_000.0, max_price=600_000.0, raw_text="300k-600k"
                ),
                BudgetValue(min_price=600_000.0, raw_text="> 600k"),
            ]

        return []


def _mean_pairwise_jaccard(id_lists: list[list[str]]) -> float:
    """Compute average pairwise Jaccard@K across candidate top-K ID lists."""
    if len(id_lists) < 2:
        return 1.0
    scores: list[float] = []
    for i in range(len(id_lists)):
        set_i = set(id_lists[i])
        for j in range(i + 1, len(id_lists)):
            set_j = set(id_lists[j])
            union = set_i | set_j
            if not union:
                scores.append(1.0)
            else:
                scores.append(len(set_i & set_j) / len(union))
    return round(sum(scores) / len(scores), 4) if scores else 1.0


class CounterfactualClarificationPolicy:
    """GrayZoneEvaluator using Neo4j/InMemory Category KG + 1,000-Product Index."""

    def __init__(
        self,
        kg_repo: OutfitKGRepository,
        snapshot_index: SnapshotCatalogIndex,
        *,
        ontology: OutfitOntology | None = None,
        thresholds: CounterfactualThresholds | None = None,
        candidate_builder: CandidateValueBuilder | None = None,
    ) -> None:
        self.kg_repo = kg_repo
        self.snapshot_index = snapshot_index
        self.ontology = ontology or get_cached_ontology()
        self.thresholds = thresholds or CounterfactualThresholds()
        self.candidate_builder = candidate_builder or CandidateValueBuilder()

    def _resolve_context(self, slots: ConversationSlots) -> dict[str, Any]:
        target_gender = slots.value_of("target_gender")
        occ_raw = slots.value_of("occasion")
        occasion_id = resolve_occasion_id(occ_raw, self.ontology) if occ_raw else None
        style_raw = slots.value_of("style")
        style_id = resolve_style_id(style_raw, self.ontology) if style_raw else None

        top_raw = slots.value_of("top_category")
        bot_raw = slots.value_of("bottom_category")
        top_cat_id = resolve_category_id(top_raw, self.ontology) if top_raw else None
        bot_cat_id = resolve_category_id(bot_raw, self.ontology) if bot_raw else None
        anchor_product_id = slots.value_of("product_id")
        if anchor_product_id is not None and not (top_cat_id or bot_cat_id):
            rec = self.snapshot_index.by_id.get(int(anchor_product_id))
            if rec is not None:
                cat_node = self.kg_repo.get_category(rec.category_id)
                if cat_node and cat_node.role in ("TOP", "OUTERWEAR"):
                    top_cat_id = rec.category_id
                elif cat_node and cat_node.role == "BOTTOM":
                    bot_cat_id = rec.category_id

        anchor_cat_id = top_cat_id or bot_cat_id
        if anchor_cat_id is None:
            cat_raw = slots.value_of("category")
            if cat_raw:
                anchor_cat_id = resolve_category_id(str(cat_raw), self.ontology)
                if anchor_cat_id is None:
                    cat_lower = str(cat_raw).strip().lower()
                    if cat_lower == "accessory":
                        anchor_cat_id = "tui_xach"
                    elif cat_lower == "dress":
                        anchor_cat_id = "vay_lien"

        color_raw = (
            slots.value_of("top_color")
            or slots.value_of("bottom_color")
            or slots.value_of("color")
        )
        anchor_color_id = (
            resolve_color_id(color_raw, self.ontology) if color_raw else None
        )

        disliked_raw = slots.value_of("disliked_colors") or []
        disliked_ids = {
            cid
            for raw in disliked_raw
            if (cid := resolve_color_id(str(raw), self.ontology)) is not None
        }

        budget_val = slots.value_of("budget")
        min_price = getattr(budget_val, "min_price", None) if budget_val else None
        max_price = getattr(budget_val, "max_price", None) if budget_val else None

        return {
            "target_gender": target_gender,
            "occasion_id": occasion_id,
            "style_id": style_id,
            "top_category_id": top_cat_id,
            "bottom_category_id": bot_cat_id,
            "anchor_category_id": anchor_cat_id,
            "anchor_product_id": anchor_product_id,
            "anchor_color_id": anchor_color_id,
            "disliked_color_ids": disliked_ids,
            "min_price": min_price,
            "max_price": max_price,
        }

    def _query_and_probe(
        self,
        *,
        candidate_value: Any,
        target_gender: str | None,
        occasion_id: str | None,
        style_id: str | None,
        top_category_id: str | None,
        bottom_category_id: str | None,
        anchor_category_id: str | None,
        anchor_product_id: int | None,
        anchor_color_id: str | None,
        disliked_color_ids: set[str],
        min_price: float | None,
        max_price: float | None,
    ) -> CandidateOutcome:
        bps: list[OutfitBlueprint] = self.kg_repo.get_outfit_blueprints(
            style_id=style_id,
            occasion_id=occasion_id,
            anchor_category_id=anchor_category_id,
            limit=10,
        )
        if not bps and anchor_category_id is not None:
            bps = self.kg_repo.get_outfit_blueprints(
                anchor_category_id=anchor_category_id, limit=10
            )
        if top_category_id and bottom_category_id:
            bps = [
                bp
                for bp in bps
                if bp.top_category_id == top_category_id
                and bp.bottom_category_id == bottom_category_id
            ]

        color_advice = self.kg_repo.get_color_harmony(anchor_color_id)
        return self.snapshot_index.probe_blueprints(
            bps,
            candidate_value=candidate_value,
            target_gender=target_gender,
            anchor_product_id=anchor_product_id,
            anchor_color_id=anchor_color_id,
            color_advice=color_advice,
            disliked_color_ids=disliked_color_ids,
            min_price=min_price,
            max_price=max_price,
            top_k=self.thresholds.top_k,
        )

    @staticmethod
    def _build_unfulfillable_decision(
        ctx: dict[str, Any],
        slots: ConversationSlots,
        base_outcome: CandidateOutcome,
    ) -> ClarificationDecision:
        ask_target = (
            "budget"
            if (ctx["min_price"] is not None or ctx["max_price"] is not None)
            else ("occasion" if not slots.has("occasion") else "target_gender")
        )
        return ClarificationDecision(
            action="ASK",
            target_slots=[ask_target],
            max_questions=1,
            decision_source="mode_c_counterfactual",
            decision_reason=(
                "Current hard constraints yield 0 feasible outfits in snapshot; "
                f"clarifying '{ask_target}'."
            ),
            reason_codes=["UNFULFILLABLE_HARD_CONSTRAINT"],
            evidence_rule_ids=base_outcome.applied_rule_ids,
            question_intent="resolve_unfulfillable_constraint",
        )

    def check_hard_constraints(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Return ASK decision if OUTFIT_ADVICE hard constraints yield 0 outfits."""
        if intent != TurnIntent.OUTFIT_ADVICE or clarification_turns >= 1:
            return None
        ctx = self._resolve_context(slots)
        anchor_cat_id: str | None = ctx["anchor_category_id"]
        if anchor_cat_id is not None:
            cat_node = self.kg_repo.get_category(anchor_cat_id)
            if cat_node is not None and not cat_node.supports_top_bottom_blueprint:
                return None

        base_outcome = self._query_and_probe(
            candidate_value="baseline",
            target_gender=ctx["target_gender"],
            occasion_id=ctx["occasion_id"],
            style_id=ctx["style_id"],
            top_category_id=ctx["top_category_id"],
            bottom_category_id=ctx["bottom_category_id"],
            anchor_category_id=anchor_cat_id,
            anchor_product_id=ctx["anchor_product_id"],
            anchor_color_id=ctx["anchor_color_id"],
            disliked_color_ids=ctx["disliked_color_ids"],
            min_price=ctx["min_price"],
            max_price=ctx["max_price"],
        )
        if base_outcome.feasible_blueprint_count == 0:
            return self._build_unfulfillable_decision(ctx, slots, base_outcome)
        return None

    def evaluate(
        self,
        intent: TurnIntent,
        slots: ConversationSlots,
        clarification_turns: int,
    ) -> ClarificationDecision | None:
        """Evaluate OUTFIT_ADVICE gray zone via single-slot counterfactual probing."""
        if intent != TurnIntent.OUTFIT_ADVICE:
            return None

        ctx = self._resolve_context(slots)
        anchor_cat_id: str | None = ctx["anchor_category_id"]

        # Out-of-scope categories (dresses/accessories) proceed directly to tool notice
        if anchor_cat_id is not None:
            cat_node = self.kg_repo.get_category(anchor_cat_id)
            if cat_node is not None and not cat_node.supports_top_bottom_blueprint:
                return ClarificationDecision(
                    action="PROCEED",
                    decision_source="mode_c_counterfactual",
                    decision_reason=(
                        f"Category '{cat_node.name}' is outside top-bottom blueprint "
                        "scope; proceeding directly."
                    ),
                    reason_codes=["COUNTERFACTUAL_OUT_OF_SCOPE_PROCEED"],
                )

        # 1. Check current hard constraint feasibility before counterfactual probing
        base_outcome = self._query_and_probe(
            candidate_value="baseline",
            target_gender=ctx["target_gender"],
            occasion_id=ctx["occasion_id"],
            style_id=ctx["style_id"],
            top_category_id=ctx["top_category_id"],
            bottom_category_id=ctx["bottom_category_id"],
            anchor_category_id=anchor_cat_id,
            anchor_product_id=ctx["anchor_product_id"],
            anchor_color_id=ctx["anchor_color_id"],
            disliked_color_ids=ctx["disliked_color_ids"],
            min_price=ctx["min_price"],
            max_price=ctx["max_price"],
        )

        if base_outcome.feasible_blueprint_count == 0 and clarification_turns < 1:
            return self._build_unfulfillable_decision(ctx, slots, base_outcome)

        # 2. Select eligible unresolved slots for single-slot counterfactual probing
        eligible_slots: list[str] = []
        if not slots.has("target_gender") and not slots.is_declined("target_gender"):
            eligible_slots.append("target_gender")
        if not slots.has("occasion") and not slots.is_declined("occasion"):
            eligible_slots.append("occasion")
        if (
            slots.has("target_gender")
            and not slots.has("style")
            and not slots.is_declined("style")
        ):
            eligible_slots.append("style")
        if not slots.has("budget") and not slots.is_declined("budget"):
            if slots.status_of("budget") == "AMBIGUOUS" or (
                slots.has("target_gender")
                and (slots.has("occasion") or anchor_cat_id is not None)
                and base_outcome.feasible_product_count
                >= self.thresholds.wide_pool_product_threshold
            ):
                eligible_slots.append("budget")

        impact_signals: dict[str, SlotImpactSignals] = {}
        evidence_rules: list[str] = list(base_outcome.applied_rule_ids)
        seen_rules: set[str] = set(evidence_rules)
        ask_candidates: list[tuple[tuple[int, int, float], str]] = []
        has_item_anchor = (
            ctx["top_category_id"] is not None
            or ctx["bottom_category_id"] is not None
            or ctx["anchor_product_id"] is not None
        )

        for slot_name in eligible_slots:
            cand_values = self.candidate_builder.build_candidates(
                slot_name,
                anchor_category_id=anchor_cat_id,
                occasion_id=ctx["occasion_id"],
                kg_repo=self.kg_repo,
            )
            if len(cand_values) < 2:
                continue

            outcomes: list[CandidateOutcome] = []
            for val in cand_values:
                probe_gender = (
                    val if slot_name == "target_gender" else ctx["target_gender"]
                )
                probe_occ = val if slot_name == "occasion" else ctx["occasion_id"]
                probe_style = val if slot_name == "style" else ctx["style_id"]
                probe_min = val.min_price if slot_name == "budget" else ctx["min_price"]
                probe_max = val.max_price if slot_name == "budget" else ctx["max_price"]

                outcome = self._query_and_probe(
                    candidate_value=(
                        val.model_dump() if isinstance(val, BudgetValue) else val
                    ),
                    target_gender=probe_gender,
                    occasion_id=probe_occ,
                    style_id=probe_style,
                    top_category_id=ctx["top_category_id"],
                    bottom_category_id=ctx["bottom_category_id"],
                    anchor_category_id=anchor_cat_id,
                    anchor_product_id=ctx["anchor_product_id"],
                    anchor_color_id=ctx["anchor_color_id"],
                    disliked_color_ids=ctx["disliked_color_ids"],
                    min_price=probe_min,
                    max_price=probe_max,
                )
                outcomes.append(outcome)
                for rid in outcome.applied_rule_ids:
                    if rid not in seen_rules:
                        seen_rules.add(rid)
                        evidence_rules.append(rid)

            viable_flags = [
                o.feasible_blueprint_count >= self.thresholds.min_viable_outfits
                and o.feasible_product_count >= self.thresholds.min_viable_products
                for o in outcomes
            ]
            feasibility_change = any(viable_flags) and not all(viable_flags)

            top_pairs = {o.category_pairs[0] for o in outcomes if o.category_pairs}
            category_shift = len(top_pairs) > 1
            mean_jaccard = _mean_pairwise_jaccard([o.top_k_ids for o in outcomes])

            can_cover_alt = (
                slot_name in ("target_gender", "occasion")
                and has_item_anchor
                and all(viable_flags)
                and len(cand_values) <= self.thresholds.alternatives_max_branches
            )

            sig = SlotImpactSignals(
                feasibility_change=feasibility_change,
                category_shift=category_shift,
                mean_top_k_jaccard=mean_jaccard,
                can_cover_with_alternatives=can_cover_alt,
            )
            impact_signals[slot_name] = sig

            is_high_impact = (
                feasibility_change
                or category_shift
                or mean_jaccard < self.thresholds.jaccard_threshold
            )
            if is_high_impact and not can_cover_alt:
                rank_key = (
                    int(feasibility_change),
                    int(category_shift),
                    round(1.0 - mean_jaccard, 4),
                )
                ask_candidates.append((rank_key, slot_name))

        # 3. Decision Order: ASK top 1-2 high-impact slots, or PROCEED cleanly
        if ask_candidates and clarification_turns < 1:
            ask_candidates.sort(key=lambda item: item[0], reverse=True)
            chosen_slots = [slot for _, slot in ask_candidates[:2]]
            top_sig = impact_signals[chosen_slots[0]]
            reason_codes = ["COUNTERFACTUAL_HIGH_IMPACT_ASK"]
            if top_sig.feasibility_change:
                reason_codes.append("FEASIBILITY_CHANGE")
            if top_sig.category_shift:
                reason_codes.append("CATEGORY_SHIFT")
            if top_sig.mean_top_k_jaccard < self.thresholds.jaccard_threshold:
                reason_codes.append("LOW_TOP_K_JACCARD")

            return ClarificationDecision(
                action="ASK",
                target_slots=chosen_slots,
                max_questions=min(2, len(chosen_slots)),
                decision_source="mode_c_counterfactual",
                decision_reason=(
                    f"Counterfactual probing identified high impact for "
                    f"{', '.join(chosen_slots)} "
                    f"(jaccard={top_sig.mean_top_k_jaccard:.2f}, "
                    f"category_shift={top_sig.category_shift})."
                ),
                reason_codes=reason_codes,
                impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
                evidence_rule_ids=evidence_rules,
                question_intent=f"clarify_{'_'.join(chosen_slots)}",
            )

        reason_codes = ["COUNTERFACTUAL_PROCEED"]
        if any(s.can_cover_with_alternatives for s in impact_signals.values()):
            reason_codes.append("COVERED_WITH_ALTERNATIVES")

        missing_gender = not slots.has("target_gender")
        return ClarificationDecision(
            action="PROCEED",
            decision_source="mode_c_counterfactual",
            decision_reason=(
                "Counterfactual evaluation confirmed item anchor or low sensitivity "
                "can be covered cleanly with grounded alternatives."
            ),
            reason_codes=reason_codes,
            impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
            evidence_rule_ids=evidence_rules,
            use_fallback=missing_gender,
            fallback_slots=["target_gender"] if missing_gender else [],
        )
