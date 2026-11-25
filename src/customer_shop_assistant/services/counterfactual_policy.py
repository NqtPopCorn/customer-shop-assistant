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
    ColorHarmonyAdvice,
    ContextColorPalette,
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

COLOR_TONE_GROUPS: dict[str, set[str]] = {
    "neutral_classic": {
        "den",
        "trang",
        "trang_kem",
        "xam_tieu",
        "xam_dam",
        "xam_nhat",
        "bac",
        "xanh_navy",
    },
    "earth_warm": {
        "be_sua",
        "be_sang",
        "nau_be",
        "nau_camel",
        "xanh_reu_khaki",
        "cam_dat",
        "vang_mu_tat",
    },
    "bright_accent": {
        "xanh_duong",
        "xanh_lam_nhat",
        "xanh_co_vit",
        "xanh_ngoc",
        "xanh_ngoc_bich",
        "xanh_la",
        "xanh_mint",
        "hong_pastel",
        "hong",
        "hong_dat",
        "do_do",
        "do_tuoi",
        "cam_dao",
        "cam",
        "vang_kem",
        "vang_anh_kim",
        "tim_than",
        "tim_lilac",
    },
}


class CounterfactualThresholds(BaseModel):
    """Tunable counterfactual decision thresholds (Dev-tuned, Test-frozen)."""

    jaccard_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    min_viable_outfits: int = Field(default=1, ge=1)
    min_viable_products: int = Field(default=2, ge=1)
    alternatives_max_branches: int = Field(default=3, ge=1)
    top_k: int = Field(default=5, ge=1)
    wide_pool_product_threshold: int = Field(default=120, ge=1)
    min_color_quality: float = Field(default=0.72, ge=0.0, le=1.0)
    color_gain_threshold: float = Field(default=0.12, ge=0.0, le=1.0)


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

    def _has_strict_color_match(
        self,
        *,
        ctx: dict[str, Any],
        target_color_id: str,
    ) -> bool:
        """Return True if explicit requested color has >=1 feasible in-stock match."""
        anchor_cat_id: str | None = ctx["anchor_category_id"]
        bps = self.kg_repo.get_outfit_blueprints(
            style_id=ctx["style_id"],
            occasion_id=ctx["occasion_id"],
            anchor_category_id=anchor_cat_id,
            limit=10,
        )
        if not bps and anchor_cat_id is not None:
            bps = self.kg_repo.get_outfit_blueprints(
                anchor_category_id=anchor_cat_id, limit=10
            )
        if ctx["top_category_id"] and ctx["bottom_category_id"]:
            bps = [
                bp
                for bp in bps
                if bp.top_category_id == ctx["top_category_id"]
                and bp.bottom_category_id == ctx["bottom_category_id"]
            ]

        disliked = ctx["disliked_color_ids"]
        if target_color_id in disliked:
            return False

        check_bot_side = (
            ctx["bottom_category_id"] is not None and ctx["top_category_id"] is None
        )
        for bp in bps:
            top_cols = None if check_bot_side else {target_color_id}
            bot_cols = {target_color_id} if check_bot_side else None
            tops = self.snapshot_index.query_products(
                bp.top_category_id,
                target_gender=ctx["target_gender"],
                color_ids=top_cols,
                disliked_color_ids=disliked,
                min_price=ctx["min_price"],
                max_price=ctx["max_price"],
            )
            bots = self.snapshot_index.query_products(
                bp.bottom_category_id,
                target_gender=ctx["target_gender"],
                color_ids=bot_cols,
                disliked_color_ids=disliked,
                min_price=ctx["min_price"],
                max_price=ctx["max_price"],
            )
            if tops and bots:
                return True
        return False

    def _probe_color_tone_outcome(
        self,
        *,
        ctx: dict[str, Any],
        candidate_label: str,
        tone_color_ids: set[str] | None,
        palette: ContextColorPalette,
        harmony_cache: dict[str, ColorHarmonyAdvice],
    ) -> CandidateOutcome:
        """Probe snapshot feasibility and mean color quality for a tone group."""
        colors_by_id = {c.id: c for c in self.ontology.colors}
        bps = self.kg_repo.get_outfit_blueprints(
            style_id=ctx["style_id"],
            occasion_id=ctx["occasion_id"],
            anchor_category_id=ctx["anchor_category_id"],
            limit=6,
        )
        if not bps and ctx["anchor_category_id"] is not None:
            bps = self.kg_repo.get_outfit_blueprints(
                anchor_category_id=ctx["anchor_category_id"], limit=6
            )
        if ctx["top_category_id"] and ctx["bottom_category_id"]:
            bps = [
                bp
                for bp in bps
                if bp.top_category_id == ctx["top_category_id"]
                and bp.bottom_category_id == ctx["bottom_category_id"]
            ]

        disliked = ctx["disliked_color_ids"]
        feasible_pairs: list[tuple[str, str]] = []
        top_k_ids: list[str] = []
        seen_top_ids: set[str] = set()
        prod_ids: set[int] = set()
        color_scores: list[float] = []

        def _best_rec_color(rec: Any, allowed: set[str] | None) -> str | None:
            pool = [c for c in rec.in_stock_color_ids if c not in disliked]
            if allowed is not None:
                pool = [c for c in pool if c in allowed]
            if not pool:
                return None
            pool.sort(
                key=lambda cid: (
                    -palette.scores_by_color.get(
                        cid,
                        0.65
                        if (cid in colors_by_id and colors_by_id[cid].neutral)
                        else 0.50,
                    ),
                    cid,
                )
            )
            return pool[0]

        for bp in bps:
            tops = self.snapshot_index.query_products(
                bp.top_category_id,
                target_gender=ctx["target_gender"],
                color_ids=tone_color_ids,
                disliked_color_ids=disliked,
                min_price=ctx["min_price"],
                max_price=ctx["max_price"],
            )
            bots = self.snapshot_index.query_products(
                bp.bottom_category_id,
                target_gender=ctx["target_gender"],
                color_ids=tone_color_ids,
                disliked_color_ids=disliked,
                min_price=ctx["min_price"],
                max_price=ctx["max_price"],
            )
            if not tops or not bots:
                continue
            feasible_pairs.append((bp.top_category_id, bp.bottom_category_id))
            for r in tops:
                prod_ids.add(r.deterministic_id)
            for r in bots:
                prod_ids.add(r.deterministic_id)

            for t_rec in tops[:2]:
                t_cid = _best_rec_color(t_rec, tone_color_ids)
                if t_cid is None:
                    continue
                adv = harmony_cache.get(t_cid)
                if adv is None:
                    adv = self.kg_repo.get_color_harmony(t_cid)
                    harmony_cache[t_cid] = adv
                t_ctx = palette.scores_by_color.get(
                    t_cid,
                    0.65
                    if (t_cid in colors_by_id and colors_by_id[t_cid].neutral)
                    else 0.50,
                )
                for b_rec in bots[:2]:
                    if t_rec.deterministic_id == b_rec.deterministic_id:
                        continue
                    b_cid = _best_rec_color(b_rec, tone_color_ids)
                    if b_cid is None:
                        continue
                    if b_cid in adv.compatibility_by_color:
                        harm = adv.compatibility_by_color[b_cid]
                    elif t_cid == b_cid and (
                        t_cid in colors_by_id and colors_by_id[t_cid].neutral
                    ):
                        harm = 0.80
                    elif (t_cid in colors_by_id and colors_by_id[t_cid].neutral) or (
                        b_cid in colors_by_id and colors_by_id[b_cid].neutral
                    ):
                        harm = 0.65
                    else:
                        harm = 0.45
                    b_ctx = palette.scores_by_color.get(
                        b_cid,
                        0.65
                        if (b_cid in colors_by_id and colors_by_id[b_cid].neutral)
                        else 0.50,
                    )
                    c_score = 0.50 * harm + 0.50 * (0.50 * (t_ctx + b_ctx))
                    color_scores.append(c_score)
                    if len(top_k_ids) < self.thresholds.top_k:
                        token = (
                            f"{t_rec.deterministic_id}:{t_cid}"
                            f"+{b_rec.deterministic_id}:{b_cid}"
                        )
                        if token not in seen_top_ids:
                            seen_top_ids.add(token)
                            top_k_ids.append(token)

        mean_cs = (
            round(sum(color_scores) / len(color_scores), 4) if color_scores else 0.0
        )
        return CandidateOutcome(
            candidate_value=candidate_label,
            feasible_product_count=len(prod_ids),
            feasible_blueprint_count=len(feasible_pairs),
            top_k_ids=top_k_ids[: self.thresholds.top_k],
            category_pairs=feasible_pairs,
            mean_color_score=mean_cs,
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

        # Condition A: explicit color yields 0 feasible outfits in stock
        anchor_color_id: str | None = ctx["anchor_color_id"]
        if anchor_color_id is not None and not self._has_strict_color_match(
            ctx=ctx, target_color_id=anchor_color_id
        ):
            return ClarificationDecision(
                action="ASK",
                target_slots=["color"],
                max_questions=1,
                decision_source="mode_c_counterfactual",
                decision_reason=(
                    f"Requested color '{anchor_color_id}' has 0 feasible matching "
                    "items in stock; clarifying alternative color."
                ),
                reason_codes=["UNFULFILLABLE_COLOR_CONSTRAINT"],
                evidence_rule_ids=base_outcome.applied_rule_ids,
                question_intent="resolve_unfulfillable_color",
            )

        color_decision = self._evaluate_color_clarification(
            ctx=ctx,
            slots=slots,
            clarification_turns=clarification_turns,
            has_item_anchor=bool(anchor_cat_id or ctx["anchor_product_id"]),
            impact_signals={},
            evidence_rules=list(base_outcome.applied_rule_ids),
        )
        if color_decision is not None:
            return color_decision
        return None

    def _evaluate_color_clarification(
        self,
        *,
        ctx: dict[str, Any],
        slots: ConversationSlots,
        clarification_turns: int,
        has_item_anchor: bool,
        impact_signals: dict[str, SlotImpactSignals],
        evidence_rules: list[str],
    ) -> ClarificationDecision | None:
        """Apply Spec v1.3 Condition B/C color clarification (never default-ask)."""
        no_explicit_color = (
            not slots.has("color")
            and not slots.has("top_color")
            and not slots.has("bottom_color")
            and not slots.is_declined("color")
        )
        is_ambiguous_color = slots.status_of("color") == "AMBIGUOUS"
        ready_context = slots.has("target_gender") and (
            slots.has("occasion") or has_item_anchor
        )
        if not (
            clarification_turns < 1
            and no_explicit_color
            and (is_ambiguous_color or ready_context)
        ):
            return None

        palette = self.kg_repo.get_context_color_palette(
            style_id=ctx["style_id"], occasion_id=ctx["occasion_id"]
        )
        harmony_cache: dict[str, ColorHarmonyAdvice] = {}
        base_col_out = self._probe_color_tone_outcome(
            ctx=ctx,
            candidate_label="baseline_palette",
            tone_color_ids=None,
            palette=palette,
            harmony_cache=harmony_cache,
        )
        if not (
            is_ambiguous_color
            or base_col_out.mean_color_score < self.thresholds.min_color_quality
        ):
            return None

        tone_outcomes = [
            self._probe_color_tone_outcome(
                ctx=ctx,
                candidate_label=tone_name,
                tone_color_ids=tone_ids,
                palette=palette,
                harmony_cache=harmony_cache,
            )
            for tone_name, tone_ids in COLOR_TONE_GROUPS.items()
        ]
        t_viable = [
            o.feasible_blueprint_count >= self.thresholds.min_viable_outfits
            and o.feasible_product_count >= self.thresholds.min_viable_products
            for o in tone_outcomes
        ]
        t_feas_chg = any(t_viable) and not all(t_viable)
        t_jaccard = _mean_pairwise_jaccard([o.top_k_ids for o in tone_outcomes])
        best_tone_cs = max((o.mean_color_score for o in tone_outcomes), default=0.0)
        col_gain = round(best_tone_cs - base_col_out.mean_color_score, 4)

        col_sig = SlotImpactSignals(
            feasibility_change=t_feas_chg,
            category_shift=False,
            mean_top_k_jaccard=t_jaccard,
            can_cover_with_alternatives=False,
            color_quality_gain=col_gain,
        )
        impact_signals["color"] = col_sig

        should_ask_b = is_ambiguous_color and (
            t_feas_chg or t_jaccard < self.thresholds.jaccard_threshold
        )
        should_ask_c = (not is_ambiguous_color) and (
            t_feas_chg or col_gain >= self.thresholds.color_gain_threshold
        )
        if not (should_ask_b or should_ask_c):
            return None

        col_reason_code = (
            "AMBIGUOUS_COLOR_PREFERENCE"
            if is_ambiguous_color
            else "LOW_BASELINE_COLOR_QUALITY"
        )
        return ClarificationDecision(
            action="ASK",
            target_slots=["color"],
            max_questions=1,
            decision_source="mode_c_counterfactual",
            decision_reason=(
                f"Color clarification triggered ({col_reason_code}): "
                f"baseline_color_score={base_col_out.mean_color_score:.2f}, "
                f"gain={col_gain:.2f}."
            ),
            reason_codes=[col_reason_code],
            impact_signals={k: v.model_dump() for k, v in impact_signals.items()},
            evidence_rule_ids=evidence_rules,
            question_intent="clarify_color",
        )

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

        anchor_color_id: str | None = ctx["anchor_color_id"]
        if (
            anchor_color_id is not None
            and clarification_turns < 1
            and not self._has_strict_color_match(
                ctx=ctx, target_color_id=anchor_color_id
            )
        ):
            return ClarificationDecision(
                action="ASK",
                target_slots=["color"],
                max_questions=1,
                decision_source="mode_c_counterfactual",
                decision_reason=(
                    f"Requested color '{anchor_color_id}' has 0 feasible matching "
                    "items in stock; clarifying alternative color."
                ),
                reason_codes=["UNFULFILLABLE_COLOR_CONSTRAINT"],
                evidence_rule_ids=base_outcome.applied_rule_ids,
                question_intent="resolve_unfulfillable_color",
            )

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

        # 3b. Color Clarification Policy (Spec v1.3 Sec 6.2: never default-ask color)
        color_decision = self._evaluate_color_clarification(
            ctx=ctx,
            slots=slots,
            clarification_turns=clarification_turns,
            has_item_anchor=has_item_anchor,
            impact_signals=impact_signals,
            evidence_rules=evidence_rules,
        )
        if color_decision is not None:
            return color_decision

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
