"""In-memory Outfit Knowledge Graph builder, TTL cache, and pair scoring."""

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import OutfitItem, OutfitSuggestion
from customer_shop_assistant.schemas.shop import (
    GenderTarget,
    ProductFilterParams,
    ProductSchema,
)
from customer_shop_assistant.services.outfit_rules import (
    OutfitRules,
    ProductAttrs,
    combine_scores,
    compute_formality,
    load_outfit_rules,
    normalize_text,
    passes_hard_filters,
    resolve_category_role,
    score_category,
    score_colour,
    score_formality,
    score_pattern,
)
from customer_shop_assistant.services.shop_client import ShopClient

logger = logging.getLogger(__name__)

_CATEGORY_ALIAS_MAP: dict[str, str] = {
    "áo thun": "Áo thun",
    "thun": "Áo thun",
    "tshirt": "Áo thun",
    "t-shirt": "Áo thun",
    "áo phông": "Áo thun",
    "áo polo": "Áo polo",
    "polo": "Áo polo",
    "áo sơ mi": "Áo sơ mi",
    "sơ mi": "Áo sơ mi",
    "shirt": "Áo sơ mi",
    "áo khoác": "Áo khoác",
    "khoác": "Áo khoác",
    "jacket": "Áo khoác",
    "blazer": "Áo khoác",
    "bomber": "Áo khoác",
    "hoodie": "Áo khoác",
    "quần jeans": "Quần jeans",
    "quần jean": "Quần jeans",
    "jeans": "Quần jeans",
    "jean": "Quần jeans",
    "quần bò": "Quần jeans",
    "quần kaki": "Quần kaki",
    "kaki": "Quần kaki",
    "khaki": "Quần kaki",
    "quần short": "Quần short",
    "short": "Quần short",
    "quần đùi": "Quần short",
    "quần âu": "Quần âu",
    "quần tây": "Quần âu",
    "pants": "Quần âu",
    "trousers": "Quần âu",
    "váy": "Chân váy",
    "chân váy": "Chân váy",
    "skirt": "Chân váy",
    "skirts": "Chân váy",
}

_SORTED_CATEGORY_ALIASES: list[tuple[str, str]] = sorted(
    _CATEGORY_ALIAS_MAP.items(), key=lambda item: -len(item[0])
)

_OUT_OF_SCOPE_CATEGORY_CUES: tuple[str, ...] = (
    "váy liền",
    "đầm dạ hội",
    "đầm",
    "dress",
    "dresses",
    "gown",
    "túi xách",
    "túi",
    "mũ nón",
    "mũ",
    "nón",
    "thắt lưng",
    "phụ kiện",
    "accessory",
    "accessories",
)


def resolve_outfit_category(
    raw_cat: str | None, rules: OutfitRules
) -> tuple[str | None, Literal["TOP", "JACKET", "BOTTOM", "OUT_OF_SCOPE"] | None]:
    """Normalize Vietnamese/English category alias to canonical name & outfit role."""
    if not raw_cat or not raw_cat.strip():
        return None, None

    direct = resolve_category_role(raw_cat, rules)
    if direct is not None:
        return direct[0], direct[1]

    norm = normalize_text(raw_cat)
    exact_alias = _CATEGORY_ALIAS_MAP.get(norm)
    if exact_alias is not None:
        resolved = resolve_category_role(exact_alias, rules)
        if resolved is not None:
            return resolved[0], resolved[1]

    for alias_norm, canonical in _SORTED_CATEGORY_ALIASES:
        if alias_norm in norm:
            resolved = resolve_category_role(canonical, rules)
            if resolved is not None:
                return resolved[0], resolved[1]

    if norm in _OUT_OF_SCOPE_CATEGORY_CUES or any(
        cue in norm for cue in _OUT_OF_SCOPE_CATEGORY_CUES
    ):
        return raw_cat.strip(), "OUT_OF_SCOPE"

    return None, None


class OutfitResult(BaseModel):
    """Internal result returned by OutfitGraph query methods."""

    mode: Literal["anchor", "open"] = Field(..., description="Query mode executed")
    outfits: list[OutfitSuggestion] = Field(
        default_factory=list, description="Ranked outfit suggestions"
    )
    notes: list[str] = Field(
        default_factory=list, description="Informational notes (e.g. colour fallback)"
    )
    error: str | None = Field(
        default=None, description="Error code if anchor out_of_scope or not_found"
    )


@dataclass(frozen=True)
class OutfitNode:
    """Graph node combining live BE ProductSchema with derived KG attributes."""

    product: ProductSchema
    attrs: ProductAttrs
    colours: list[str]
    in_stock_by_colour: dict[str, bool]

    def is_in_stock(self, colour: str) -> bool:
        """Return whether the given colour has at least one variant with stock > 0."""
        return self.in_stock_by_colour.get(normalize_text(colour), False)


_CACHE: tuple[float, "OutfitGraph"] | None = None


def clear_outfit_graph_cache() -> None:
    """Clear the cached OutfitGraph instance (useful in tests)."""
    global _CACHE
    _CACHE = None


def _load_product_attrs(outfit_dir: Path) -> dict[str, ProductAttrs]:
    attrs_path = outfit_dir / "product_attrs.json"
    if not attrs_path.exists():
        return {}
    with attrs_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {slug: ProductAttrs.model_validate(item) for slug, item in raw.items()}


def _extract_colours(product: ProductSchema) -> tuple[list[str], dict[str, bool]]:
    colours: list[str] = []
    seen_norm: set[str] = set()
    in_stock_map: dict[str, bool] = {}

    for v in product.variants:
        col = (v.color or "").strip() or "Tiêu chuẩn"
        col_norm = normalize_text(col)
        if col_norm not in seen_norm:
            seen_norm.add(col_norm)
            colours.append(col)
            in_stock_map[col_norm] = False
        if v.stock > 0:
            in_stock_map[col_norm] = True

    return colours, in_stock_map


def _to_outfit_item(node: OutfitNode, colour: str) -> OutfitItem:
    p = node.product
    col_norm = normalize_text(colour)
    matching = [
        v
        for v in p.variants
        if normalize_text((v.color or "").strip() or "Tiêu chuẩn") == col_norm
    ]
    pool = matching or p.variants
    prices = [
        v.final_price if v.final_price is not None else v.selling_price for v in pool
    ]
    price = min(prices) if prices else 0.0

    img = next((v.image_url for v in matching if v.image_url), None)
    if not img and p.images:
        img = p.images[0].url
    if not img and p.variants:
        img = next((v.image_url for v in p.variants if v.image_url), None)

    return OutfitItem(
        product_id=p.id,
        name=p.name,
        slug=p.slug,
        category=node.attrs.category,
        color=colour,
        in_stock=node.is_in_stock(colour),
        image_url=img,
        price=price,
    )


def _merge_reasons(
    cat_reasons: list[str],
    colour_reasons: list[str],
    pattern_reasons: list[str],
    formality_reasons: list[str],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for r in [*cat_reasons, *colour_reasons, *pattern_reasons, *formality_reasons]:
        if r and r not in seen:
            seen.add(r)
            merged.append(r)
            if len(merged) >= 4:
                break
    return merged


@dataclass
class _ScoredCandidate:
    top_node: OutfitNode
    bottom_node: OutfitNode
    top_colour: str
    bottom_colour: str
    score: float
    both_in_stock: bool
    reasons: list[str]

    def to_suggestion(self) -> OutfitSuggestion:
        return OutfitSuggestion(
            top=_to_outfit_item(self.top_node, self.top_colour),
            bottom=_to_outfit_item(self.bottom_node, self.bottom_colour),
            score=self.score,
            reasons=self.reasons,
        )


class OutfitGraph:
    """In-memory bipartite outfit graph pairing tops and bottoms."""

    def __init__(
        self,
        nodes: dict[int, OutfitNode],
        out_of_scope_ids: set[int],
        rules: OutfitRules,
    ) -> None:
        self.nodes = nodes
        self.out_of_scope_ids = out_of_scope_ids
        self.rules = rules

    @classmethod
    def from_products(
        cls,
        products: list[ProductSchema],
        rules: OutfitRules,
        attrs_by_slug: dict[str, ProductAttrs],
    ) -> "OutfitGraph":
        """Build OutfitGraph by joining BE products with offline attrs by slug."""
        nodes: dict[int, OutfitNode] = {}
        out_of_scope_ids: set[int] = set()

        for p in products:
            attrs = attrs_by_slug.get(p.slug)
            if attrs is None:
                cat_name = p.category.name if p.category else None
                resolved = resolve_category_role(cat_name, rules)
                if resolved is None:
                    out_of_scope_ids.add(p.id)
                    continue
                canonical_cat, role = resolved
                attrs = ProductAttrs(
                    slug=p.slug,
                    role=role,
                    category=canonical_cat,
                    pattern_raw=None,
                    pattern_class="SOLID",
                    formality=compute_formality(canonical_cat, p.description, rules),
                )

            colours, in_stock_map = _extract_colours(p)
            if not colours:
                continue

            nodes[p.id] = OutfitNode(
                product=p,
                attrs=attrs,
                colours=colours,
                in_stock_by_colour=in_stock_map,
            )

        return cls(nodes=nodes, out_of_scope_ids=out_of_scope_ids, rules=rules)

    def _score_product_pair(
        self,
        top: OutfitNode,
        bottom: OutfitNode,
        top_colours: list[str],
        bottom_colours: list[str],
        effective_gender: GenderTarget | None,
        occasion_range: tuple[int, int] | None,
        favorite_norm: set[str],
        disliked: set[str],
    ) -> _ScoredCandidate | None:
        # Pre-check colour-independent constraints before iterating colours
        if not passes_hard_filters(
            top.attrs,
            bottom.attrs,
            top.product.target_gender,
            bottom.product.target_gender,
            effective_gender,
            occasion_range,
            set(),
            "",
            "",
        ):
            return None

        cat_s, cat_reasons = score_category(
            top.attrs.category, bottom.attrs.category, self.rules
        )
        pat_s, pat_reasons = score_pattern(
            top.attrs.pattern_class,
            top.attrs.pattern_raw,
            bottom.attrs.pattern_class,
            bottom.attrs.pattern_raw,
        )
        form_s, form_reasons = score_formality(
            top.attrs.formality, bottom.attrs.formality, occasion_range
        )

        best: _ScoredCandidate | None = None
        disliked_norm = {normalize_text(c) for c in disliked}

        for tc in top_colours:
            tc_norm = normalize_text(tc)
            if tc_norm in disliked_norm:
                continue
            for bc in bottom_colours:
                bc_norm = normalize_text(bc)
                if bc_norm in disliked_norm:
                    continue

                col_s, col_reasons = score_colour(tc, bc, self.rules)
                fav_hit = tc_norm in favorite_norm or bc_norm in favorite_norm
                total = combine_scores(
                    cat_s, col_s, pat_s, form_s, fav_hit, self.rules.weights
                )
                both_stock = top.is_in_stock(tc) and bottom.is_in_stock(bc)

                if best is None or (total, both_stock) > (
                    best.score,
                    best.both_in_stock,
                ):
                    reasons = _merge_reasons(
                        cat_reasons, col_reasons, pat_reasons, form_reasons
                    )
                    best = _ScoredCandidate(
                        top_node=top,
                        bottom_node=bottom,
                        top_colour=tc,
                        bottom_colour=bc,
                        score=total,
                        both_in_stock=both_stock,
                        reasons=reasons,
                    )

        return best

    def suggest_for_anchor(
        self,
        anchor_id: int,
        colour: str | None = None,
        occasion_range: tuple[int, int] | None = None,
        gender_hint: GenderTarget | None = None,
        favorite: set[str] | None = None,
        disliked: set[str] | None = None,
        limit: int = 3,
        target_category: str | None = None,
        target_color: str | None = None,
    ) -> OutfitResult:
        """Mode A: recommend matching complements for a specific product."""
        cap = max(1, min(5, limit))
        if anchor_id in self.out_of_scope_ids:
            return OutfitResult(mode="anchor", error="out_of_scope")
        anchor = self.nodes.get(anchor_id)
        if anchor is None:
            return OutfitResult(mode="anchor", error="not_found")

        eff_gender = (
            anchor.product.target_gender
            if anchor.product.target_gender != GenderTarget.UNISEX
            else gender_hint
        )

        notes: list[str] = []
        anchor_colours = anchor.colours
        if colour:
            c_norm = normalize_text(colour)
            matched = [c for c in anchor.colours if normalize_text(c) == c_norm]
            if matched:
                anchor_colours = matched
            else:
                notes.append(
                    f"Sản phẩm {anchor.product.name} không có màu '{colour}', "
                    "hệ thống gợi ý theo các màu hiện có."
                )

        favorite_norm = {normalize_text(c) for c in (favorite or set())}
        disliked_set = disliked or set()
        anchor_is_top = anchor.attrs.role in ("TOP", "JACKET")

        if anchor_is_top:
            candidates = [n for n in self.nodes.values() if n.attrs.role == "BOTTOM"]
        else:
            candidates = [n for n in self.nodes.values() if n.attrs.role == "TOP"]

        if target_category:
            resolved_target_cat, target_role = resolve_outfit_category(
                target_category, self.rules
            )
            if target_role == "OUT_OF_SCOPE":
                return OutfitResult(mode="anchor", error="out_of_scope")
            if resolved_target_cat is not None:
                if not anchor_is_top and target_role == "JACKET":
                    target_pool = [
                        n
                        for n in self.nodes.values()
                        if n.attrs.category == resolved_target_cat
                    ]
                elif (anchor_is_top and target_role == "BOTTOM") or (
                    not anchor_is_top and target_role == "TOP"
                ):
                    target_pool = [
                        n for n in candidates if n.attrs.category == resolved_target_cat
                    ]
                else:
                    target_pool = []

                if target_pool:
                    candidates = target_pool
                elif (anchor_is_top and target_role == "BOTTOM") or (
                    not anchor_is_top and target_role in ("TOP", "JACKET")
                ):
                    notes.append(
                        f"Hiện chưa có mẫu {resolved_target_cat} phù hợp, "
                        "hệ thống mở rộng sang các danh mục tương đương."
                    )

        cand_colour_map: dict[int, list[str]] = {
            c.product.id: c.colours for c in candidates
        }
        if target_color:
            tc_norm = normalize_text(target_color)
            matched_cands: list[OutfitNode] = []
            matched_col_map: dict[int, list[str]] = {}
            for cand in candidates:
                cols = [c for c in cand.colours if normalize_text(c) == tc_norm]
                if cols:
                    matched_cands.append(cand)
                    matched_col_map[cand.product.id] = cols
            if matched_cands:
                candidates = matched_cands
                cand_colour_map = matched_col_map
            else:
                notes.append(
                    f"Chưa có sản phẩm phối cùng màu '{target_color}', "
                    "hệ thống gợi ý theo các màu hiện có."
                )

        scored: list[_ScoredCandidate] = []
        for cand in candidates:
            if cand.product.id == anchor_id:
                continue
            cand_cols = cand_colour_map.get(cand.product.id, cand.colours)
            if anchor_is_top:
                pair = self._score_product_pair(
                    anchor,
                    cand,
                    anchor_colours,
                    cand_cols,
                    eff_gender,
                    occasion_range,
                    favorite_norm,
                    disliked_set,
                )
            else:
                pair = self._score_product_pair(
                    cand,
                    anchor,
                    cand_cols,
                    anchor_colours,
                    eff_gender,
                    occasion_range,
                    favorite_norm,
                    disliked_set,
                )
            if pair is not None:
                scored.append(pair)

        scored.sort(
            key=lambda x: (
                -x.score,
                -int(x.both_in_stock),
                x.top_node.product.id,
                x.bottom_node.product.id,
            )
        )
        outfits = [item.to_suggestion() for item in scored[:cap]]
        return OutfitResult(mode="anchor", outfits=outfits, notes=notes)

    @staticmethod
    def _filter_nodes_with_colour(
        nodes: list[OutfitNode],
        target_color: str | None,
        notes_out: list[str],
        label: str,
    ) -> list[tuple[OutfitNode, list[str]]]:
        if not target_color:
            return [(n, n.colours) for n in nodes]
        c_norm = normalize_text(target_color)
        matched: list[tuple[OutfitNode, list[str]]] = []
        for n in nodes:
            cols = [c for c in n.colours if normalize_text(c) == c_norm]
            if cols:
                matched.append((n, cols))
        if matched:
            return matched
        if nodes:
            notes_out.append(
                f"Chưa có mẫu {label} màu '{target_color}', "
                "hệ thống gợi ý theo các màu hiện có."
            )
        return [(n, n.colours) for n in nodes]

    def _rank_open_for_gender(
        self,
        eff_gender: GenderTarget | None,
        occasion_range: tuple[int, int] | None,
        favorite_norm: set[str],
        disliked_set: set[str],
        k: int,
        exclude_ids: set[int] | None = None,
        top_category: str | None = None,
        bottom_category: str | None = None,
        top_color: str | None = None,
        bottom_color: str | None = None,
        notes_out: list[str] | None = None,
    ) -> list[_ScoredCandidate]:
        if top_category == "Áo khoác":
            tops = [n for n in self.nodes.values() if n.attrs.role == "JACKET"]
        elif top_category is not None:
            tops = [n for n in self.nodes.values() if n.attrs.category == top_category]
        else:
            tops = [n for n in self.nodes.values() if n.attrs.role == "TOP"]

        if bottom_category is not None:
            bottoms = [
                n for n in self.nodes.values() if n.attrs.category == bottom_category
            ]
        else:
            bottoms = [n for n in self.nodes.values() if n.attrs.role == "BOTTOM"]

        blocked = exclude_ids or set()
        tops = [n for n in tops if n.product.id not in blocked]
        bottoms = [n for n in bottoms if n.product.id not in blocked]

        local_notes = notes_out if notes_out is not None else []
        top_entries = self._filter_nodes_with_colour(
            tops, top_color, local_notes, top_category or "áo"
        )
        bot_entries = self._filter_nodes_with_colour(
            bottoms, bottom_color, local_notes, bottom_category or "quần/váy"
        )

        candidates: list[_ScoredCandidate] = []
        for top, top_cols in top_entries:
            for bot, bot_cols in bot_entries:
                if bot.product.id == top.product.id:
                    continue
                pair = self._score_product_pair(
                    top,
                    bot,
                    top_cols,
                    bot_cols,
                    eff_gender,
                    occasion_range,
                    favorite_norm,
                    disliked_set,
                )
                if pair is not None:
                    candidates.append(pair)

        candidates.sort(
            key=lambda x: (
                -x.score,
                -int(x.both_in_stock),
                x.top_node.product.id,
                x.bottom_node.product.id,
            )
        )

        picked: list[_ScoredCandidate] = []
        used_ids: set[int] = set()
        for cand in candidates:
            t_id = cand.top_node.product.id
            b_id = cand.bottom_node.product.id
            if t_id in used_ids or b_id in used_ids:
                continue
            used_ids.add(t_id)
            used_ids.add(b_id)
            picked.append(cand)
            if len(picked) >= k:
                break
        return picked

    def _resolve_open_filters(
        self,
        top_category: str | None,
        bottom_category: str | None,
        top_color: str | None,
        bottom_color: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None, bool]:
        t_cat, t_role = resolve_outfit_category(top_category, self.rules)
        b_cat, b_role = resolve_outfit_category(bottom_category, self.rules)
        if t_role == "OUT_OF_SCOPE" or b_role == "OUT_OF_SCOPE":
            return None, None, None, None, True

        resolved_top: str | None = None
        resolved_bot: str | None = None
        eff_top_col = top_color
        eff_bot_col = bottom_color

        if t_role in ("TOP", "JACKET"):
            resolved_top = t_cat
        elif t_role == "BOTTOM" and b_role is None:
            resolved_bot = t_cat
            if eff_bot_col is None and eff_top_col is not None:
                eff_bot_col, eff_top_col = eff_top_col, None

        if b_role == "BOTTOM":
            resolved_bot = b_cat
        elif b_role in ("TOP", "JACKET") and resolved_top is None:
            resolved_top = b_cat
            if eff_top_col is None and eff_bot_col is not None:
                eff_top_col, eff_bot_col = eff_bot_col, None

        return resolved_top, resolved_bot, eff_top_col, eff_bot_col, False

    def suggest_open(
        self,
        occasion_range: tuple[int, int] | None = None,
        gender: GenderTarget | None = None,
        favorite: set[str] | None = None,
        disliked: set[str] | None = None,
        limit: int = 3,
        top_category: str | None = None,
        bottom_category: str | None = None,
        top_color: str | None = None,
        bottom_color: str | None = None,
    ) -> OutfitResult:
        """Mode B: recommend complete top+bottom sets without an anchor product."""
        cap = max(1, min(5, limit))
        resolved_top, resolved_bot, eff_top_col, eff_bot_col, is_oos = (
            self._resolve_open_filters(
                top_category, bottom_category, top_color, bottom_color
            )
        )
        if is_oos:
            return OutfitResult(mode="open", error="out_of_scope")

        eff_gender = gender
        if resolved_bot == "Chân váy" and eff_gender is None:
            eff_gender = GenderTarget.FEMALE

        favorite_norm = {normalize_text(c) for c in (favorite or set())}
        disliked_set = disliked or set()
        notes: list[str] = []

        if eff_gender is not None:
            picked = self._rank_open_for_gender(
                eff_gender,
                occasion_range,
                favorite_norm,
                disliked_set,
                cap,
                top_category=resolved_top,
                bottom_category=resolved_bot,
                top_color=eff_top_col,
                bottom_color=eff_bot_col,
                notes_out=notes,
            )
            return OutfitResult(
                mode="open",
                outfits=[c.to_suggestion() for c in picked],
                notes=list(dict.fromkeys(notes)),
            )

        half = math.ceil(cap / 2)
        female_picks = self._rank_open_for_gender(
            GenderTarget.FEMALE,
            occasion_range,
            favorite_norm,
            disliked_set,
            half,
            top_category=resolved_top,
            bottom_category=resolved_bot,
            top_color=eff_top_col,
            bottom_color=eff_bot_col,
            notes_out=notes,
        )
        used_female = {
            pid
            for c in female_picks
            for pid in (c.top_node.product.id, c.bottom_node.product.id)
        }
        male_picks = self._rank_open_for_gender(
            GenderTarget.MALE,
            occasion_range,
            favorite_norm,
            disliked_set,
            half,
            exclude_ids=used_female,
            top_category=resolved_top,
            bottom_category=resolved_bot,
            top_color=eff_top_col,
            bottom_color=eff_bot_col,
            notes_out=notes,
        )

        interleaved: list[_ScoredCandidate] = []
        for idx in range(max(len(female_picks), len(male_picks))):
            if idx < len(female_picks):
                interleaved.append(female_picks[idx])
            if idx < len(male_picks):
                interleaved.append(male_picks[idx])

        dedup_notes = list(dict.fromkeys(notes))
        dedup_notes.append("Chưa rõ giới tính, gợi ý cân bằng cho cả Nữ và Nam.")
        return OutfitResult(
            mode="open",
            outfits=[c.to_suggestion() for c in interleaved[:cap]],
            notes=dedup_notes,
        )


async def get_outfit_graph(client: ShopClient | None = None) -> OutfitGraph:
    """Lazily build and TTL-cache the OutfitGraph from Main BE products."""
    global _CACHE
    settings = get_settings()
    ttl = settings.OUTFIT_KG_CACHE_TTL_SECONDS
    now = time.monotonic()

    if _CACHE is not None:
        cached_at, cached_graph = _CACHE
        if (now - cached_at) < ttl and cached_graph.nodes:
            return cached_graph

    outfit_dir: Path = settings.DATA_DIR / "outfit_kg"
    rules = load_outfit_rules(outfit_dir)
    attrs_by_slug = _load_product_attrs(outfit_dir)

    filters = ProductFilterParams(page=1, limit=2000)
    if client is not None:
        products = await client.get_products(filters)
    else:
        async with ShopClient(settings=settings) as owned_client:
            products = await owned_client.get_products(filters)

    if products and all(not p.variants for p in products):
        logger.error(
            "ShopClient.get_products returned %d products with empty variants; "
            "cannot build variant-level OutfitGraph.",
            len(products),
        )
        return OutfitGraph(nodes={}, out_of_scope_ids=set(), rules=rules)

    graph = OutfitGraph.from_products(products, rules, attrs_by_slug)
    if graph.nodes:
        _CACHE = (now, graph)
    return graph
