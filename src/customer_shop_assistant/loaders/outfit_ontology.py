"""Loader, validator, and canonical ID resolvers for data/outfit_kg/ontology/."""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.outfit_kg import (
    CategoryNode,
    ColorNode,
    FitsStyleEdge,
    HarmonizesWithEdge,
    OccasionNode,
    OutfitOntology,
    PairsWithEdge,
    PrefersColorEdge,
    RecommendsCategoryEdge,
    StyleNode,
    SuitsCategoryEdge,
    SuitsColorEdge,
)

VALID_SLOT_CODES = frozenset({"office", "party", "casual", "sport"})


def normalize_key(text: str | None) -> str:
    """Normalize Vietnamese/ASCII label for deterministic alias matching."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).strip().lower().split())


def _strip_diacritics(text: str) -> str:
    """Return ASCII-folded Vietnamese string for accent-insensitive fallback."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "d"))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_outfit_ontology(ontology_dir: Path | None = None) -> OutfitOntology:
    """Load the 5 Git-tracked JSON files into a typed OutfitOntology model."""
    base_dir = ontology_dir or (get_settings().DATA_DIR / "outfit_kg" / "ontology")
    categories = [
        CategoryNode.model_validate(item)
        for item in _read_json(base_dir / "categories.json")
    ]
    styles = [
        StyleNode.model_validate(item) for item in _read_json(base_dir / "styles.json")
    ]
    occasions = [
        OccasionNode.model_validate(item)
        for item in _read_json(base_dir / "occasions.json")
    ]
    colors = [
        ColorNode.model_validate(item) for item in _read_json(base_dir / "colours.json")
    ]
    edges_raw = _read_json(base_dir / "edges.json")
    return OutfitOntology(
        categories=categories,
        styles=styles,
        occasions=occasions,
        colors=colors,
        fits_style=[
            FitsStyleEdge.model_validate(e) for e in edges_raw.get("fits_style", [])
        ],
        suits_category=[
            SuitsCategoryEdge.model_validate(e)
            for e in edges_raw.get("suits_category", [])
        ],
        recommends_category=[
            RecommendsCategoryEdge.model_validate(e)
            for e in edges_raw.get("recommends_category", [])
        ],
        pairs_with=[
            PairsWithEdge.model_validate(e) for e in edges_raw.get("pairs_with", [])
        ],
        harmonizes_with=[
            HarmonizesWithEdge.model_validate(e)
            for e in edges_raw.get("harmonizes_with", [])
        ],
        prefers_color=[
            PrefersColorEdge.model_validate(e)
            for e in edges_raw.get("prefers_color", [])
        ],
        suits_color=[
            SuitsColorEdge.model_validate(e) for e in edges_raw.get("suits_color", [])
        ],
    )


@lru_cache(maxsize=4)
def get_cached_ontology(ontology_dir_str: str | None = None) -> OutfitOntology:
    """Return cached OutfitOntology instance keyed by directory path."""
    path = Path(ontology_dir_str) if ontology_dir_str else None
    return load_outfit_ontology(path)


def clear_ontology_cache() -> None:
    """Clear the cached OutfitOntology (useful for tests)."""
    get_cached_ontology.cache_clear()


def _check_unique_ids_and_aliases(
    nodes: list[Any], label: str, errors: list[str]
) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    alias_owner: dict[str, str] = {}
    for node in nodes:
        if node.id in by_id:
            errors.append(f"Duplicate {label} id: '{node.id}'")
        by_id[node.id] = node

        keys = [node.id, node.name, *node.aliases]
        seen_local: set[str] = set()
        for raw_key in keys:
            norm = normalize_key(raw_key)
            if not norm or norm in seen_local:
                continue
            seen_local.add(norm)
            if norm in alias_owner and alias_owner[norm] != node.id:
                errors.append(
                    f"Duplicate {label} alias '{norm}' between "
                    f"'{alias_owner[norm]}' and '{node.id}'"
                )
            else:
                alias_owner[norm] = node.id
    return by_id


def validate_ontology(
    ontology: OutfitOntology,
    snapshot_products: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Validate ontology integrity and optional 1,000-product snapshot coverage."""
    errors: list[str] = []

    cat_by_id = _check_unique_ids_and_aliases(ontology.categories, "Category", errors)
    style_by_id = _check_unique_ids_and_aliases(ontology.styles, "Style", errors)
    occ_by_id = _check_unique_ids_and_aliases(ontology.occasions, "Occasion", errors)
    color_by_id = _check_unique_ids_and_aliases(ontology.colors, "Color", errors)

    for occ in ontology.occasions:
        if occ.slot_code not in VALID_SLOT_CODES:
            errors.append(
                f"Occasion '{occ.id}' has invalid slot_code '{occ.slot_code}'"
            )

    seen_rule_ids: set[str] = set()
    all_edges = [
        ("fits_style", ontology.fits_style, occ_by_id, style_by_id),
        ("suits_category", ontology.suits_category, occ_by_id, cat_by_id),
        ("recommends_category", ontology.recommends_category, style_by_id, cat_by_id),
        ("pairs_with", ontology.pairs_with, cat_by_id, cat_by_id),
        ("harmonizes_with", ontology.harmonizes_with, color_by_id, color_by_id),
        ("prefers_color", ontology.prefers_color, style_by_id, color_by_id),
        ("suits_color", ontology.suits_color, occ_by_id, color_by_id),
    ]
    for rel_name, edges, src_map, tgt_map in all_edges:
        for edge in edges:
            if edge.rule_id in seen_rule_ids:
                errors.append(f"Duplicate rule_id '{edge.rule_id}' in {rel_name}")
            seen_rule_ids.add(edge.rule_id)
            if edge.source_id not in src_map:
                errors.append(
                    f"{rel_name} edge '{edge.rule_id}' references missing "
                    f"source_id '{edge.source_id}'"
                )
            if edge.target_id not in tgt_map:
                errors.append(
                    f"{rel_name} edge '{edge.rule_id}' references missing "
                    f"target_id '{edge.target_id}'"
                )

    for pw in ontology.pairs_with:
        src_cat = cat_by_id.get(pw.source_id)
        tgt_cat = cat_by_id.get(pw.target_id)
        if src_cat and src_cat.role not in ("TOP", "OUTERWEAR"):
            errors.append(
                f"PAIRS_WITH '{pw.rule_id}' source '{pw.source_id}' "
                f"has invalid role '{src_cat.role}'"
            )
        if tgt_cat and tgt_cat.role != "BOTTOM":
            errors.append(
                f"PAIRS_WITH '{pw.rule_id}' target '{pw.target_id}' "
                f"has invalid role '{tgt_cat.role}'"
            )
        if src_cat and not src_cat.supports_top_bottom_blueprint:
            errors.append(
                f"PAIRS_WITH '{pw.rule_id}' source '{pw.source_id}' "
                "does not support top-bottom blueprint"
            )
        if tgt_cat and not tgt_cat.supports_top_bottom_blueprint:
            errors.append(
                f"PAIRS_WITH '{pw.rule_id}' target '{pw.target_id}' "
                "does not support top-bottom blueprint"
            )
        for sid in pw.style_ids:
            if sid not in style_by_id:
                errors.append(
                    f"PAIRS_WITH '{pw.rule_id}' references unknown style_id '{sid}'"
                )
        for oid in pw.occasion_ids:
            if oid not in occ_by_id:
                errors.append(
                    f"PAIRS_WITH '{pw.rule_id}' references unknown occasion_id '{oid}'"
                )

    seen_color_pairs: set[tuple[str, str]] = set()
    covered_colors: set[str] = set()
    for ch in ontology.harmonizes_with:
        if ch.source_id >= ch.target_id:
            errors.append(
                f"HARMONIZES_WITH '{ch.rule_id}' must enforce source_id < target_id "
                f"(got '{ch.source_id}' -> '{ch.target_id}')"
            )
        pair = (ch.source_id, ch.target_id)
        if pair in seen_color_pairs:
            errors.append(
                f"Duplicate HARMONIZES_WITH pair ({ch.source_id}, {ch.target_id})"
            )
        seen_color_pairs.add(pair)
        covered_colors.add(ch.source_id)
        covered_colors.add(ch.target_id)

    for col_id in sorted(set(color_by_id) - covered_colors):
        errors.append(f"Isolated Color node without HARMONIZES_WITH edge: '{col_id}'")

    seen_prefers: set[tuple[str, str]] = set()
    styles_with_colors: set[str] = set()
    for pc in ontology.prefers_color:
        pair = (pc.source_id, pc.target_id)
        if pair in seen_prefers:
            errors.append(
                f"Duplicate PREFERS_COLOR edge ({pc.source_id}, {pc.target_id})"
            )
        seen_prefers.add(pair)
        styles_with_colors.add(pc.source_id)

    for st_id in sorted(set(style_by_id) - styles_with_colors):
        errors.append(f"Style node missing PREFERS_COLOR edges: '{st_id}'")

    seen_suits_col: set[tuple[str, str]] = set()
    for sc in ontology.suits_color:
        pair = (sc.source_id, sc.target_id)
        if pair in seen_suits_col:
            errors.append(
                f"Duplicate SUITS_COLOR edge ({sc.source_id}, {sc.target_id})"
            )
        seen_suits_col.add(pair)

    if snapshot_products is not None:
        unmapped_cats: set[str] = set()
        unmapped_colors: set[str] = set()
        for prod in snapshot_products:
            cat_name = prod.get("categoryName")
            if resolve_category_id(cat_name, ontology) is None:
                unmapped_cats.add(str(cat_name))
            for variant in prod.get("variants") or []:
                col_name = variant.get("color")
                if resolve_color_id(col_name, ontology) is None:
                    unmapped_colors.add(str(col_name))
        for c in sorted(unmapped_cats):
            errors.append(f"Unmapped snapshot categoryName: '{c}'")
        for col in sorted(unmapped_colors):
            errors.append(f"Unmapped snapshot variant.color: '{col}'")

    return errors


def resolve_category_id(
    raw: str | None, ontology: OutfitOntology | None = None
) -> str | None:
    """Resolve raw category name/alias/slug to canonical Category.id."""
    key = normalize_key(raw)
    if not key:
        return None
    ont = ontology or get_cached_ontology()
    for cat in ont.categories:
        if key == normalize_key(cat.id) or key == normalize_key(cat.name):
            return cat.id
        if any(key == normalize_key(a) for a in cat.aliases):
            return cat.id
    return None


def resolve_style_id(
    raw: str | None, ontology: OutfitOntology | None = None
) -> str | None:
    """Resolve raw style name/alias/slug to canonical Style.id."""
    key = normalize_key(raw)
    if not key:
        return None
    ont = ontology or get_cached_ontology()
    for st in ont.styles:
        if key == normalize_key(st.id) or key == normalize_key(st.name):
            return st.id
        if any(key == normalize_key(a) for a in st.aliases):
            return st.id
    return None


def resolve_occasion_id(
    raw: str | None, ontology: OutfitOntology | None = None
) -> str | None:
    """Resolve raw occasion name/alias/slot_code to canonical Occasion.id."""
    key = normalize_key(raw)
    if not key:
        return None
    ont = ontology or get_cached_ontology()
    for occ in ont.occasions:
        if key == normalize_key(occ.id) or key == normalize_key(occ.name):
            return occ.id
        if any(key == normalize_key(a) for a in occ.aliases):
            return occ.id
    # Fallback by slot_code (e.g. 'casual' -> 'hangout')
    slot_defaults = {
        "office": "office",
        "party": "party",
        "casual": "hangout",
        "sport": "sport",
    }
    if key in slot_defaults and any(o.id == slot_defaults[key] for o in ont.occasions):
        return slot_defaults[key]
    return None


def resolve_occasions_by_slot_code(
    slot_code: str, ontology: OutfitOntology | None = None
) -> list[str]:
    """Return all Occasion IDs mapped to a slot_tracker slot_code."""
    ont = ontology or get_cached_ontology()
    norm = normalize_key(slot_code)
    return [o.id for o in ont.occasions if o.slot_code == norm]


def resolve_color_id(
    raw: str | None, ontology: OutfitOntology | None = None
) -> str | None:
    """Resolve raw color name/alias/slug to canonical Color.id."""
    key = normalize_key(raw)
    if not key:
        return None
    ont = ontology or get_cached_ontology()
    for col in ont.colors:
        if key == normalize_key(col.id) or key == normalize_key(col.name):
            return col.id
        if any(key == normalize_key(a) for a in col.aliases):
            return col.id

    folded = _strip_diacritics(key)
    for col in ont.colors:
        if folded == _strip_diacritics(normalize_key(col.name)):
            return col.id
        if any(folded == _strip_diacritics(normalize_key(a)) for a in col.aliases):
            return col.id
    return None
