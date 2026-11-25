"""Category-level Outfit Retrieval & Lightweight Reranker (Spec v1.2)."""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
    resolve_color_id,
    resolve_occasion_id,
    resolve_style_id,
)
from customer_shop_assistant.schemas.chat import (
    ChatRecommendedProduct,
    OutfitItem,
    OutfitSuggestion,
)
from customer_shop_assistant.schemas.outfit_kg import (
    ColorHarmonyAdvice,
    ColorNode,
    ColorScoreBreakdown,
    ContextColorPalette,
    OutfitBlueprint,
    OutfitOntology,
)
from customer_shop_assistant.schemas.shop import GenderTarget, ProductSchema
from customer_shop_assistant.services.outfit_kg_repository import OutfitKGRepository
from customer_shop_assistant.services.snapshot_catalog_index import (
    SnapshotCatalogIndex,
    SnapshotProductRecord,
)
from customer_shop_assistant.tools.catalog_tool import product_schema_to_recommended


class OutfitRetrievalResult(BaseModel):
    """Structured result returned by OutfitRetrievalAndReranker."""

    outfits: list[OutfitSuggestion] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error: Literal["out_of_scope", "not_found"] | None = None
    products_by_id: dict[int, ChatRecommendedProduct] = Field(default_factory=dict)


def _records_from_product_schemas(
    products: list[ProductSchema], ontology: OutfitOntology
) -> list[SnapshotProductRecord]:
    """Convert live/mocked ProductSchema list into SnapshotProductRecord view."""
    records: list[SnapshotProductRecord] = []
    for p in products:
        cat_name = p.category.name if p.category else ""
        cat_id = resolve_category_id(cat_name, ontology) or ""
        in_stock_cols: set[str] = set()
        all_cols: set[str] = set()
        color_prices: dict[str, float] = {}
        color_names: dict[str, str] = {}
        color_images: dict[str, str | None] = {}
        prices: list[float] = []

        for v in p.variants:
            raw_col = (v.color or "").strip() or "Tiêu chuẩn"
            cid = resolve_color_id(raw_col, ontology) or raw_col.lower()
            all_cols.add(cid)
            color_names.setdefault(cid, raw_col)
            if v.image_url and cid not in color_images:
                color_images[cid] = v.image_url
            price = float(
                v.final_price
                if v.final_price is not None
                else (v.selling_price or v.list_price or 0.0)
            )
            prices.append(price)
            if v.stock > 0:
                in_stock_cols.add(cid)
                if cid not in color_prices or price < color_prices[cid]:
                    color_prices[cid] = price

        primary_img = p.images[0].url if p.images else None
        records.append(
            SnapshotProductRecord(
                deterministic_id=p.id,
                slug=p.slug,
                hm_product_code="",
                name=p.name,
                category_id=cat_id,
                category_name=cat_name,
                target_gender=p.target_gender.value,
                brand_name=p.brand.name if p.brand else None,
                min_price=min(prices) if prices else 0.0,
                max_price=max(prices) if prices else 0.0,
                in_stock_color_ids=frozenset(in_stock_cols),
                all_color_ids=frozenset(all_cols),
                color_price_map=color_prices,
                color_name_by_id=color_names,
                color_image_by_id=color_images,
                primary_image_url=primary_img,
                in_stock=bool(in_stock_cols),
            )
        )
    return records


def _record_to_recommended(rec: SnapshotProductRecord) -> ChatRecommendedProduct:
    return ChatRecommendedProduct(
        id=rec.deterministic_id,
        name=rec.name,
        slug=rec.slug,
        price=rec.min_price,
        image_url=rec.primary_image_url,
        category=rec.category_name,
        brand=rec.brand_name,
    )


class OutfitRetrievalAndReranker:
    """Combines Category-level KG blueprints with hard catalog filters & soft color."""

    def __init__(
        self,
        kg_repo: OutfitKGRepository,
        snapshot_index: SnapshotCatalogIndex | None = None,
        ontology: OutfitOntology | None = None,
    ) -> None:
        self.kg_repo = kg_repo
        self.snapshot_index = snapshot_index
        self.ontology = ontology or get_cached_ontology()

    def _build_working_index(
        self, live_products: list[ProductSchema] | None
    ) -> tuple[SnapshotCatalogIndex, dict[int, ChatRecommendedProduct]]:
        if live_products:
            recs = _records_from_product_schemas(live_products, self.ontology)
            prod_map = {p.id: product_schema_to_recommended(p) for p in live_products}
            return SnapshotCatalogIndex(recs, self.ontology), prod_map
        if self.snapshot_index is None:
            return SnapshotCatalogIndex([], self.ontology), {}
        prod_map = {
            r.deterministic_id: _record_to_recommended(r)
            for r in self.snapshot_index.records
        }
        return self.snapshot_index, prod_map

    @staticmethod
    def _candidate_color_slots(
        rec: SnapshotProductRecord,
        preferred_color_ids: list[str],
        disliked_ids: set[str],
        palette: ContextColorPalette,
        colors_by_id: dict[str, ColorNode],
        max_colors: int = 3,
    ) -> list[tuple[str, str, bool]]:
        """Return candidate (cid, label, in_stock) options prioritized by context."""
        picked: list[tuple[str, str, bool]] = []
        seen: set[str] = set()

        for cid in preferred_color_ids:
            if cid in disliked_ids or cid in seen:
                continue
            if cid in rec.in_stock_color_ids:
                picked.append((cid, rec.color_name_by_id.get(cid, cid), True))
                seen.add(cid)
                return picked
            if cid in rec.all_color_ids:
                picked.append((cid, rec.color_name_by_id.get(cid, cid), False))
                seen.add(cid)
                return picked

        def _ctx_rank(cid: str) -> tuple[float, str]:
            if cid in palette.scores_by_color:
                return (-palette.scores_by_color[cid], cid)
            node = colors_by_id.get(cid)
            return (-0.65 if (node and node.neutral) else -0.50, cid)

        in_stock_pool = [
            cid for cid in rec.in_stock_color_ids if cid not in disliked_ids
        ]
        in_stock_pool.sort(key=_ctx_rank)
        for cid in in_stock_pool[:max_colors]:
            picked.append((cid, rec.color_name_by_id.get(cid, cid), True))
            seen.add(cid)

        if not picked:
            oos_pool = [cid for cid in rec.all_color_ids if cid not in disliked_ids]
            oos_pool.sort(key=_ctx_rank)
            for cid in oos_pool[:1]:
                picked.append((cid, rec.color_name_by_id.get(cid, cid), False))
                seen.add(cid)

        if not picked:
            picked.append(("", "Tiêu chuẩn", rec.in_stock))

        return picked

    @staticmethod
    def _score_color_pair(
        *,
        top_cid: str,
        top_label: str,
        bot_cid: str,
        bot_label: str,
        palette: ContextColorPalette,
        colors_by_id: dict[str, ColorNode],
        harmony_lookup: dict[str, ColorHarmonyAdvice],
        get_harmony_fn: Any,
        req_top_cid: str | None,
        req_bot_cid: str | None,
        fav_ids: set[str],
    ) -> tuple[ColorScoreBreakdown, list[str]]:
        """Score a (top_color, bottom_color) pair and build full debug breakdown."""
        unmapped: list[str] = []
        top_known = top_cid if top_cid in colors_by_id else None
        bot_known = bot_cid if bot_cid in colors_by_id else None
        if top_known is None and top_label:
            unmapped.append(top_label)
        if bot_known is None and bot_label and bot_label not in unmapped:
            unmapped.append(bot_label)

        rule_ids: list[str] = []
        reasons: list[str] = []

        # 1. Pair Harmony (HARMONIZES_WITH)
        harmony_kind = None
        if top_known is None or bot_known is None:
            pair_harmony = 0.50
        else:
            adv = harmony_lookup.get(top_known)
            if adv is None:
                adv = get_harmony_fn(top_known)
                harmony_lookup[top_known] = adv
            if bot_known in adv.compatibility_by_color:
                pair_harmony = float(adv.compatibility_by_color[bot_known])
                harmony_kind = adv.kind_by_color.get(bot_known)
                rid = adv.rule_id_by_color.get(bot_known)
                rsn = adv.reason_by_color.get(bot_known)
                if rid:
                    rule_ids.append(rid)
                if rsn:
                    reasons.append(rsn)
            elif top_known == bot_known and colors_by_id[top_known].neutral:
                pair_harmony = 0.80
                harmony_kind = "monochrome_tone"
            elif colors_by_id[top_known].neutral or colors_by_id[bot_known].neutral:
                pair_harmony = 0.65
            else:
                pair_harmony = 0.45

        # 2. Context Color Fit (70/30 normalized Style + Occasion palette)
        def _ctx_for(cid_opt: str | None) -> float:
            if cid_opt is None:
                return 0.50
            if cid_opt in palette.scores_by_color:
                for rid in palette.rule_ids_by_color.get(cid_opt, ()):
                    if rid not in rule_ids:
                        rule_ids.append(rid)
                for rsn in palette.reasons_by_color.get(cid_opt, ()):
                    if rsn not in reasons:
                        reasons.append(rsn)
                return float(palette.scores_by_color[cid_opt])
            return 0.65 if colors_by_id[cid_opt].neutral else 0.50

        t_ctx = _ctx_for(top_known)
        b_ctx = _ctx_for(bot_known)
        context_color_fit = round(0.5 * (t_ctx + b_ctx), 4)

        def _component_fit(score_map: dict[str, float]) -> float | None:
            if not score_map:
                return None
            vals: list[float] = []
            for cid_opt in (top_known, bot_known):
                if cid_opt is None:
                    vals.append(0.50)
                elif cid_opt in score_map:
                    vals.append(float(score_map[cid_opt]))
                else:
                    vals.append(0.65 if colors_by_id[cid_opt].neutral else 0.50)
            return round(0.5 * sum(vals), 4)

        style_fit = _component_fit(palette.style_scores_by_color)
        occ_fit = _component_fit(palette.occasion_scores_by_color)

        # 3. Explicit User Color Match
        user_match: float | None = None
        if req_top_cid or req_bot_cid or fav_ids:
            checks: list[float] = []
            if req_top_cid:
                checks.append(1.0 if top_known == req_top_cid else 0.60)
            if req_bot_cid:
                checks.append(1.0 if bot_known == req_bot_cid else 0.60)
            if fav_ids and not req_top_cid and not req_bot_cid:
                has_fav = (top_known in fav_ids) or (bot_known in fav_ids)
                checks.append(1.0 if has_fav else 0.55)
            user_match = round(sum(checks) / len(checks), 4) if checks else 1.0

        if user_match is None:
            color_score = round(0.50 * pair_harmony + 0.50 * context_color_fit, 4)
        else:
            color_score = round(
                0.40 * pair_harmony + 0.25 * context_color_fit + 0.35 * user_match,
                4,
            )

        breakdown = ColorScoreBreakdown(
            color_score=min(1.0, max(0.0, color_score)),
            pair_harmony=min(1.0, max(0.0, round(pair_harmony, 4))),
            harmony_kind=harmony_kind,
            context_color_fit=min(1.0, max(0.0, context_color_fit)),
            style_color_fit=style_fit,
            occasion_color_fit=occ_fit,
            user_color_match=user_match,
            top_color_id=top_known,
            bottom_color_id=bot_known,
            color_rule_ids=rule_ids,
            unmapped_colors=unmapped,
        )
        return breakdown, reasons

    @staticmethod
    def _to_outfit_item(
        rec: SnapshotProductRecord, color_label: str, in_stock: bool
    ) -> OutfitItem:
        cid = next(
            (k for k, v in rec.color_name_by_id.items() if v == color_label), None
        )
        price = rec.color_price_map.get(cid, rec.min_price) if cid else rec.min_price
        img = (
            rec.color_image_by_id.get(cid) or rec.primary_image_url
            if cid
            else rec.primary_image_url
        )
        return OutfitItem(
            product_id=rec.deterministic_id,
            name=rec.name,
            slug=rec.slug,
            category=rec.category_name,
            color=color_label,
            in_stock=in_stock,
            image_url=img,
            price=price,
        )

    def retrieve_outfits(
        self,
        *,
        live_products: list[ProductSchema] | None = None,
        product_id: int | None = None,
        target_gender: GenderTarget | str | None = None,
        occasion: str | None = None,
        style: str | None = None,
        top_category: str | None = None,
        bottom_category: str | None = None,
        color: str | None = None,
        top_color: str | None = None,
        bottom_color: str | None = None,
        favorite_colors: list[str] | None = None,
        disliked_colors: list[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 3,
    ) -> OutfitRetrievalResult:
        """Retrieve and rerank complete top-bottom outfits grounded in KG + catalog."""
        index, prod_map = self._build_working_index(live_products)
        if not index.records:
            return OutfitRetrievalResult()

        notes: list[str] = []
        cap = max(1, min(5, limit))

        # Resolve canonical IDs
        top_cat_id = resolve_category_id(top_category, self.ontology)
        bot_cat_id = resolve_category_id(bottom_category, self.ontology)
        if top_category and top_cat_id is None:
            return OutfitRetrievalResult(error="out_of_scope")
        if bottom_category and bot_cat_id is None:
            return OutfitRetrievalResult(error="out_of_scope")

        for req_cat_id in (top_cat_id, bot_cat_id):
            if req_cat_id is not None:
                c_node = self.kg_repo.get_category(req_cat_id)
                if c_node is None or not c_node.supports_top_bottom_blueprint:
                    return OutfitRetrievalResult(error="out_of_scope")

        anchor_rec: SnapshotProductRecord | None = None
        if product_id is not None:
            anchor_rec = index.by_id.get(product_id)
            if anchor_rec is None:
                return OutfitRetrievalResult(error="not_found")
            anchor_cat = self.kg_repo.get_category(anchor_rec.category_id)
            if anchor_cat is None or not anchor_cat.supports_top_bottom_blueprint:
                return OutfitRetrievalResult(error="out_of_scope")
            if anchor_cat.role in ("TOP", "OUTERWEAR"):
                top_cat_id = anchor_rec.category_id
            elif anchor_cat.role == "BOTTOM":
                bot_cat_id = anchor_rec.category_id

        # Normalize role placement when user passed bottom category in top_category arg
        if top_cat_id is not None and bot_cat_id is None:
            t_node = self.kg_repo.get_category(top_cat_id)
            if t_node and t_node.role == "BOTTOM":
                bot_cat_id, top_cat_id = top_cat_id, None
                if bottom_color is None and top_color is not None:
                    bottom_color, top_color = top_color, None

        occasion_id = resolve_occasion_id(occasion, self.ontology) if occasion else None
        style_id = resolve_style_id(style, self.ontology) if style else None

        disliked_ids = {
            cid
            for c in (disliked_colors or [])
            if (cid := resolve_color_id(c, self.ontology)) is not None
        }

        fav_ids = {
            cid
            for c in (favorite_colors or [])
            if (cid := resolve_color_id(c, self.ontology)) is not None
        }
        colors_by_id = {c.id: c for c in self.ontology.colors}
        palette = self.kg_repo.get_context_color_palette(
            style_id=style_id, occasion_id=occasion_id
        )
        harmony_lookup: dict[str, ColorHarmonyAdvice] = {}

        eff_top_col_id = resolve_color_id(top_color or color, self.ontology)
        eff_bot_col_id = resolve_color_id(bottom_color, self.ontology)
        if color and bot_cat_id is not None and top_cat_id is None and not bottom_color:
            eff_bot_col_id = resolve_color_id(color, self.ontology)
            eff_top_col_id = None

        anchor_cat_id = top_cat_id or bot_cat_id
        blueprints: list[OutfitBlueprint] = self.kg_repo.get_outfit_blueprints(
            style_id=style_id,
            occasion_id=occasion_id,
            anchor_category_id=anchor_cat_id,
            limit=12,
        )
        if not blueprints and anchor_cat_id is not None:
            blueprints = self.kg_repo.get_outfit_blueprints(
                anchor_category_id=anchor_cat_id, limit=12
            )
            if blueprints:
                notes.append(
                    "Dịp/phong cách nêu trên ít phối với món đồ này, "
                    "hệ thống mở rộng theo các cặp phối tương thích nhất."
                )

        if top_cat_id and bot_cat_id:
            blueprints = [
                bp
                for bp in blueprints
                if bp.top_category_id == top_cat_id
                and bp.bottom_category_id == bot_cat_id
            ]

        if not blueprints:
            return OutfitRetrievalResult(notes=notes, products_by_id=prod_map)

        eff_gender_str: str | None = None
        if isinstance(target_gender, GenderTarget):
            eff_gender_str = target_gender.value
        elif isinstance(target_gender, str) and target_gender.strip():
            eff_gender_str = target_gender.strip().upper()
        if eff_gender_str is None and anchor_rec is not None:
            if anchor_rec.target_gender in ("MALE", "FEMALE"):
                eff_gender_str = anchor_rec.target_gender
        if eff_gender_str is None and bot_cat_id == "chan_vay":
            eff_gender_str = "FEMALE"

        def _pick_for_gender(
            gender_filter: str | None, target_count: int
        ) -> list[OutfitSuggestion]:
            scored: list[tuple[float, OutfitSuggestion]] = []
            for bp in blueprints:
                if (
                    anchor_rec is not None
                    and anchor_rec.category_id == bp.top_category_id
                ):
                    tops = (
                        [anchor_rec] if anchor_rec.matches_gender(gender_filter) else []
                    )
                else:
                    top_pref = [eff_top_col_id] if eff_top_col_id else []
                    tops = index.query_products(
                        bp.top_category_id,
                        target_gender=gender_filter,
                        color_ids=set(top_pref) if top_pref else None,
                        disliked_color_ids=disliked_ids,
                        min_price=min_price,
                        max_price=max_price,
                    )
                    if not tops and top_pref:
                        tops = index.query_products(
                            bp.top_category_id,
                            target_gender=gender_filter,
                            disliked_color_ids=disliked_ids,
                            min_price=min_price,
                            max_price=max_price,
                        )
                        if tops:
                            notes.append(
                                f"Chưa có mẫu áo màu '{top_color or color}', "
                                "gợi ý theo màu hiện còn hàng."
                            )

                if (
                    anchor_rec is not None
                    and anchor_rec.category_id == bp.bottom_category_id
                ):
                    bots = (
                        [anchor_rec] if anchor_rec.matches_gender(gender_filter) else []
                    )
                else:
                    bot_pref = [eff_bot_col_id] if eff_bot_col_id else []
                    bots = index.query_products(
                        bp.bottom_category_id,
                        target_gender=gender_filter,
                        color_ids=set(bot_pref) if bot_pref else None,
                        disliked_color_ids=disliked_ids,
                        min_price=min_price,
                        max_price=max_price,
                    )
                    if not bots and bot_pref:
                        bots = index.query_products(
                            bp.bottom_category_id,
                            target_gender=gender_filter,
                            disliked_color_ids=disliked_ids,
                            min_price=min_price,
                            max_price=max_price,
                        )

                if not tops or not bots:
                    continue

                for t_rec in tops[:3]:
                    t_pref_cols = [eff_top_col_id] if eff_top_col_id else []
                    t_cands = self._candidate_color_slots(
                        t_rec,
                        t_pref_cols,
                        disliked_ids,
                        palette,
                        colors_by_id,
                    )
                    for b_rec in bots[:3]:
                        if t_rec.deterministic_id == b_rec.deterministic_id:
                            continue
                        b_pref_cols = [eff_bot_col_id] if eff_bot_col_id else []
                        b_cands = self._candidate_color_slots(
                            b_rec,
                            b_pref_cols,
                            disliked_ids,
                            palette,
                            colors_by_id,
                        )
                        best_pair: (
                            tuple[
                                float,
                                str,
                                bool,
                                str,
                                bool,
                                ColorScoreBreakdown,
                                list[str],
                            ]
                            | None
                        ) = None
                        for t_cid, t_lbl, t_stock in t_cands:
                            for b_cid, b_lbl, b_stock in b_cands:
                                brk, col_reasons = self._score_color_pair(
                                    top_cid=t_cid,
                                    top_label=t_lbl,
                                    bot_cid=b_cid,
                                    bot_label=b_lbl,
                                    palette=palette,
                                    colors_by_id=colors_by_id,
                                    harmony_lookup=harmony_lookup,
                                    get_harmony_fn=self.kg_repo.get_color_harmony,
                                    req_top_cid=eff_top_col_id,
                                    req_bot_cid=eff_bot_col_id,
                                    fav_ids=fav_ids,
                                )
                                cand_key = (brk.color_score, t_cid, b_cid)
                                if best_pair is None or cand_key > (
                                    best_pair[0],
                                    best_pair[1],
                                    best_pair[3],
                                ):
                                    best_pair = (
                                        brk.color_score,
                                        t_lbl,
                                        t_stock,
                                        b_lbl,
                                        b_stock,
                                        brk,
                                        col_reasons,
                                    )
                        assert best_pair is not None
                        _, t_col_lbl, t_stock, b_col_lbl, b_stock, brk, c_reasons = (
                            best_pair
                        )
                        bp_score = round(bp.score, 4)
                        combo_score = round(
                            min(1.0, 0.70 * bp_score + 0.30 * brk.color_score), 4
                        )
                        all_rules = list(
                            dict.fromkeys([*bp.applied_rule_ids, *brk.color_rule_ids])
                        )
                        all_reasons = list(dict.fromkeys([*bp.reasons, *c_reasons[:2]]))
                        suggestion = OutfitSuggestion(
                            top=self._to_outfit_item(t_rec, t_col_lbl, t_stock),
                            bottom=self._to_outfit_item(b_rec, b_col_lbl, b_stock),
                            score=combo_score,
                            blueprint_score=bp_score,
                            color_score_breakdown=brk,
                            reasons=all_reasons[:4],
                            applied_rule_ids=all_rules,
                        )
                        scored.append((combo_score, suggestion))

            scored.sort(
                key=lambda x: (
                    -x[0],
                    x[1].top.product_id,
                    x[1].bottom.product_id,
                )
            )

            picked: list[OutfitSuggestion] = []
            used_tops: set[int] = set()
            used_bots: set[int] = set()
            for _, cand in scored:
                if (
                    cand.top.product_id in used_tops
                    and cand.bottom.product_id in used_bots
                ):
                    continue
                picked.append(cand)
                used_tops.add(cand.top.product_id)
                used_bots.add(cand.bottom.product_id)
                if len(picked) >= target_count:
                    break
            return picked

        if eff_gender_str is not None:
            outfits = _pick_for_gender(eff_gender_str, cap)
        else:
            half = math.ceil(cap / 2)
            female_picks = _pick_for_gender("FEMALE", half)
            male_picks = _pick_for_gender("MALE", half)
            outfits = []
            for idx in range(max(len(female_picks), len(male_picks))):
                if idx < len(female_picks):
                    outfits.append(female_picks[idx])
                if idx < len(male_picks):
                    outfits.append(male_picks[idx])
            outfits = outfits[:cap]
            if outfits:
                notes.append(
                    "Chưa rõ giới tính, hệ thống gợi ý cân bằng cho cả Nữ và Nam."
                )

        return OutfitRetrievalResult(
            outfits=outfits,
            notes=list(dict.fromkeys(notes)),
            products_by_id=prod_map,
        )
