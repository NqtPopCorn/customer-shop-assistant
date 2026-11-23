"""Pure outfit matching rules, attribute extractors, and scoring functions."""

import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from customer_shop_assistant.schemas.shop import GenderTarget


def normalize_text(value: str) -> str:
    """Normalize Vietnamese text to NFC and casefolded form for matching."""
    return unicodedata.normalize("NFC", value).strip().casefold()


class ColourAttr(BaseModel):
    """Semantic colour properties used by harmony scoring rules."""

    name: str = Field(..., description="Canonical Vietnamese colour name")
    family: str = Field(default="other", description="Colour wheel family")
    neutral: bool = Field(default=False, description="Whether colour acts as neutral")
    lightness: Literal["light", "mid", "dark"] = Field(
        default="mid", description="Perceived brightness tier"
    )
    loud: bool = Field(default=False, description="Whether colour is high-saturation")


class ProductAttrs(BaseModel):
    """Offline/runtime product attributes derived for outfit matching."""

    slug: str = Field(..., description="Unique product slug shared with BE")
    role: Literal["TOP", "JACKET", "BOTTOM"] = Field(
        ..., description="Outfit role classification"
    )
    category: str = Field(..., description="Canonical Vietnamese category name")
    pattern_raw: str | None = Field(
        default=None, description="Raw H&M graphical_appearance_name"
    )
    pattern_class: Literal["SOLID", "PATTERN"] = Field(
        ..., description="Normalized pattern class"
    )
    formality: int = Field(
        ..., ge=1, le=5, description="Formality score from 1 (casual) to 5 (formal)"
    )


class OutfitRules(BaseModel):
    """Loaded Knowledge Graph rules and colour ontology."""

    roles: dict[str, Literal["TOP", "JACKET", "BOTTOM"]] = Field(
        ..., description="Category name to outfit role map"
    )
    roles_norm: dict[str, tuple[str, Literal["TOP", "JACKET", "BOTTOM"]]] = Field(
        default_factory=dict,
        description="Normalized category name to (canonical_name, role)",
    )
    base_formality: dict[str, int] = Field(
        ..., description="Base formality score per category"
    )
    base_formality_norm: dict[str, int] = Field(
        default_factory=dict, description="Normalized category to base formality"
    )
    pairs_with: dict[str, dict[str, float]] = Field(
        ..., description="Category compatibility matrix (top -> bottom -> weight)"
    )
    pairs_with_norm: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Normalized category compatibility matrix"
    )
    formality_keywords_plus: list[str] = Field(
        default_factory=list, description="Keywords increasing formality by 1"
    )
    formality_keywords_minus: list[str] = Field(
        default_factory=list, description="Keywords decreasing formality by 1"
    )
    occasions: dict[str, tuple[int, int]] = Field(
        default_factory=dict, description="Occasion keyword to [min, max] formality"
    )
    occasions_sorted_norm: list[tuple[str, tuple[int, int]]] = Field(
        default_factory=list,
        description="Normalized occasions sorted longest-key-first",
    )
    solid_patterns: list[str] = Field(
        default_factory=list, description="Raw pattern values treated as SOLID"
    )
    solid_patterns_norm: frozenset[str] = Field(
        default_factory=frozenset, description="Normalized SOLID pattern values"
    )
    weights: dict[str, float] = Field(
        ..., description="Soft scoring weights (cat, colour, pattern, formality, fav)"
    )
    colours: dict[str, ColourAttr] = Field(
        default_factory=dict, description="Normalized colour name to ColourAttr"
    )
    classic_pairs_norm: frozenset[frozenset[str]] = Field(
        default_factory=frozenset,
        description="Unordered classic colour pairs (normalized)",
    )


_DEFAULT_COLOUR = ColourAttr(
    name="other", family="other", neutral=False, lightness="mid", loud=False
)


def _resolve_rules_dir(data_dir: Path) -> Path:
    if (data_dir / "rules.json").exists():
        return data_dir
    nested = data_dir / "outfit_kg"
    if (nested / "rules.json").exists():
        return nested
    return data_dir


@lru_cache(maxsize=8)
def _load_outfit_rules_cached(resolved_dir: Path) -> OutfitRules:
    rules_path = resolved_dir / "rules.json"
    colours_path = resolved_dir / "colours.json"

    with rules_path.open("r", encoding="utf-8") as f:
        raw_rules = json.load(f)
    with colours_path.open("r", encoding="utf-8") as f:
        raw_colours = json.load(f)

    roles: dict[str, Literal["TOP", "JACKET", "BOTTOM"]] = raw_rules["roles"]
    roles_norm: dict[str, tuple[str, Literal["TOP", "JACKET", "BOTTOM"]]] = {
        normalize_text(cat): (cat, role) for cat, role in roles.items()
    }

    base_formality: dict[str, int] = {
        k: int(v) for k, v in raw_rules["base_formality"].items()
    }
    base_formality_norm = {normalize_text(k): v for k, v in base_formality.items()}

    pairs_with: dict[str, dict[str, float]] = raw_rules["pairs_with"]
    pairs_with_norm = {
        normalize_text(top_k): {
            normalize_text(bot_k): float(score) for bot_k, score in bot_map.items()
        }
        for top_k, bot_map in pairs_with.items()
    }

    kw_cfg = raw_rules.get("formality_keywords", {})
    plus_kws = [normalize_text(k) for k in kw_cfg.get("plus", [])]
    minus_kws = [normalize_text(k) for k in kw_cfg.get("minus", [])]

    occasions: dict[str, tuple[int, int]] = {
        k: (int(v[0]), int(v[1])) for k, v in raw_rules.get("occasions", {}).items()
    }
    occasions_sorted_norm = sorted(
        ((normalize_text(k), v) for k, v in occasions.items()),
        key=lambda item: -len(item[0]),
    )

    solid_patterns: list[str] = list(raw_rules.get("solid_patterns", []))
    solid_patterns_norm = frozenset(normalize_text(p) for p in solid_patterns)

    weights = {k: float(v) for k, v in raw_rules["weights"].items()}

    colours: dict[str, ColourAttr] = {}
    for c_name, c_data in raw_colours.get("colours", {}).items():
        colours[normalize_text(c_name)] = ColourAttr(name=c_name, **c_data)

    classic_pairs_norm = frozenset(
        frozenset((normalize_text(pair[0]), normalize_text(pair[1])))
        for pair in raw_colours.get("classic_pairs", [])
        if len(pair) == 2 and normalize_text(pair[0]) != normalize_text(pair[1])
    )

    return OutfitRules(
        roles=roles,
        roles_norm=roles_norm,
        base_formality=base_formality,
        base_formality_norm=base_formality_norm,
        pairs_with=pairs_with,
        pairs_with_norm=pairs_with_norm,
        formality_keywords_plus=plus_kws,
        formality_keywords_minus=minus_kws,
        occasions=occasions,
        occasions_sorted_norm=occasions_sorted_norm,
        solid_patterns=solid_patterns,
        solid_patterns_norm=solid_patterns_norm,
        weights=weights,
        colours=colours,
        classic_pairs_norm=classic_pairs_norm,
    )


def load_outfit_rules(data_dir: Path) -> OutfitRules:
    """Load rules.json and colours.json from data_dir (cached by resolved path)."""
    return _load_outfit_rules_cached(_resolve_rules_dir(data_dir).resolve())


def resolve_category_role(
    category: str | None, rules: OutfitRules
) -> tuple[str, Literal["TOP", "JACKET", "BOTTOM"]] | None:
    """Resolve category name to (canonical_category, role) if inside outfit scope."""
    if not category:
        return None
    return rules.roles_norm.get(normalize_text(category))


def compute_formality(
    category: str, description: str | None, rules: OutfitRules
) -> int:
    """Compute formality score in [1, 5] from base category and description cues."""
    base = rules.base_formality_norm.get(normalize_text(category), 3)
    desc_norm = normalize_text(description) if description else ""
    has_plus = any(kw in desc_norm for kw in rules.formality_keywords_plus)
    has_minus = any(kw in desc_norm for kw in rules.formality_keywords_minus)

    delta = 0
    if has_plus and not has_minus:
        delta = 1
    elif has_minus and not has_plus:
        delta = -1

    return max(1, min(5, base + delta))


def classify_pattern(
    raw: str | None, rules: OutfitRules
) -> Literal["SOLID", "PATTERN"]:
    """Map raw H&M graphical_appearance_name into SOLID or PATTERN."""
    if not raw or not raw.strip():
        return "SOLID"
    if normalize_text(raw) in rules.solid_patterns_norm:
        return "SOLID"
    return "PATTERN"


def normalize_occasion(text: str | None, rules: OutfitRules) -> tuple[int, int] | None:
    """Map free-text Vietnamese occasion string to formality range [lo, hi]."""
    if not text or not text.strip():
        return None
    text_norm = normalize_text(text)
    for key_norm, f_range in rules.occasions_sorted_norm:
        if key_norm in text_norm:
            return f_range
    return None


def score_category(
    top_cat: str, bottom_cat: str, rules: OutfitRules
) -> tuple[float, list[str]]:
    """Score category pairing via PAIRS_WITH matrix."""
    resolved_top = resolve_category_role(top_cat, rules)
    row_key = "áo khoác" if resolved_top and resolved_top[1] == "JACKET" else top_cat
    row = rules.pairs_with_norm.get(normalize_text(row_key), {})
    score = row.get(normalize_text(bottom_cat), 0.5)

    reasons: list[str] = []
    if score >= 0.9:
        reasons.append(f"{top_cat} + {bottom_cat}: cặp kinh điển")
    elif score >= 0.6:
        reasons.append(f"{top_cat} + {bottom_cat}: phối được")
    return score, reasons


def _get_colour_attr(colour_name: str, rules: OutfitRules) -> ColourAttr:
    return rules.colours.get(normalize_text(colour_name), _DEFAULT_COLOUR)


def score_colour(c1: str, c2: str, rules: OutfitRules) -> tuple[float, list[str]]:
    """Score colour harmony between top and bottom variants using 4 rules + classics."""
    c1_norm = normalize_text(c1)
    c2_norm = normalize_text(c2)
    a = _get_colour_attr(c1, rules)
    b = _get_colour_attr(c2, rules)

    score = 0.5
    reasons: list[str] = []

    if c1_norm != c2_norm and frozenset((c1_norm, c2_norm)) in rules.classic_pairs_norm:
        score += 0.25
        reasons.append(f"{c1} – {c2}: cặp màu kinh điển")

    if a.neutral or b.neutral:
        score += 0.25
        reasons.append("Có màu trung tính làm nền")

    lightnesses = {a.lightness, b.lightness}
    if lightnesses == {"light", "dark"}:
        score += 0.15
        reasons.append("Tương phản sáng – tối hài hòa")
    elif lightnesses in ({"light", "mid"}, {"mid", "dark"}):
        score += 0.07

    both_loud = a.loud and b.loud
    if both_loud:
        score -= 0.35
        reasons.append("Hai màu rực dễ bị chỏi")

    if a.family == b.family and a.family != "other" and not both_loud:
        score += 0.10
        reasons.append("Phối cùng tông màu (tone-sur-tone)")

    clamped = max(0.0, min(1.0, score))
    return clamped, reasons


def score_pattern(
    p_top_class: Literal["SOLID", "PATTERN"],
    p_top_raw: str | None,
    p_bottom_class: Literal["SOLID", "PATTERN"],
    p_bottom_raw: str | None,
) -> tuple[float, list[str]]:
    """Score pattern compatibility between top and bottom."""
    top_denim = bool(p_top_raw and normalize_text(p_top_raw) == "denim")
    bot_denim = bool(p_bottom_raw and normalize_text(p_bottom_raw) == "denim")

    if top_denim and bot_denim:
        return 0.6, ["Phối denim-on-denim, cần khéo léo về sắc độ"]
    if p_top_class == "SOLID" and p_bottom_class == "SOLID":
        return 1.0, ["Cả hai đều trơn nên dễ phối"]
    if p_top_class != p_bottom_class:
        return 0.9, ["Một món họa tiết làm điểm nhấn trên nền trơn"]
    return 0.2, ["Hai món đều có họa tiết dễ gây rối mắt"]


def score_formality(
    f_top: int,
    f_bottom: int,
    occasion_range: tuple[int, int] | None,
) -> tuple[float, list[str]]:
    """Score formality alignment between items and optional target occasion."""
    diff = abs(f_top - f_bottom)
    gap_scores = {0: 1.0, 1: 0.8, 2: 0.4}
    gap_score = gap_scores.get(diff, 0.0)

    reasons: list[str] = []
    if occasion_range is None:
        if diff <= 1:
            reasons.append(f"Độ trang trọng {f_top}–{f_bottom} cân bằng, đồng điệu")
        return gap_score, reasons

    lo, hi = occasion_range
    if lo <= f_top <= hi and lo <= f_bottom <= hi:
        occ_score = 1.0
        reasons.append(f"Độ trang trọng {f_top}–{f_bottom} phù hợp với dịp đã chọn")
    elif (lo - 1) <= f_top <= (hi + 1) and (lo - 1) <= f_bottom <= (hi + 1):
        occ_score = 0.6
        reasons.append(f"Độ trang trọng {f_top}–{f_bottom} tương đối hợp với dịp")
    else:
        occ_score = 0.0

    return (gap_score + occ_score) / 2.0, reasons


def _gender_str(g: GenderTarget | str | None) -> str | None:
    if g is None:
        return None
    if isinstance(g, GenderTarget):
        return g.value
    return str(g).strip().upper() or None


def passes_hard_filters(
    top: ProductAttrs,
    bottom: ProductAttrs,
    top_gender: GenderTarget | str,
    bottom_gender: GenderTarget | str,
    effective_gender: GenderTarget | str | None,
    occasion_range: tuple[int, int] | None,
    disliked: set[str],
    top_colour: str,
    bottom_colour: str,
) -> bool:
    """Verify hard constraints: roles, gender, disliked colours, formality."""
    if top.role not in ("TOP", "JACKET") or bottom.role != "BOTTOM":
        return False

    eff_g = _gender_str(effective_gender)
    top_g = _gender_str(top_gender)
    bot_g = _gender_str(bottom_gender)

    if eff_g is not None:
        if top_g not in (eff_g, GenderTarget.UNISEX.value):
            return False
        if bot_g not in (eff_g, GenderTarget.UNISEX.value):
            return False
        if eff_g == GenderTarget.MALE.value and (
            normalize_text(bottom.category) == "chân váy"
        ):
            return False

    if disliked:
        disliked_norm = {normalize_text(c) for c in disliked}
        if (
            normalize_text(top_colour) in disliked_norm
            or normalize_text(bottom_colour) in disliked_norm
        ):
            return False

    if abs(top.formality - bottom.formality) > 2:
        return False

    if occasion_range is not None:
        lo, hi = occasion_range
        if not ((lo - 1) <= top.formality <= (hi + 1)):
            return False
        if not ((lo - 1) <= bottom.formality <= (hi + 1)):
            return False

    return True


def combine_scores(
    cat_score: float,
    colour_score: float,
    pattern_score: float,
    formality_score: float,
    fav_hit: bool,
    weights: dict[str, float],
) -> float:
    """Combine dimension scores using weighted sum from rules.json."""
    fav_score = 1.0 if fav_hit else 0.0
    total = (
        weights.get("cat", 0.30) * cat_score
        + weights.get("colour", 0.30) * colour_score
        + weights.get("pattern", 0.15) * pattern_score
        + weights.get("formality", 0.20) * formality_score
        + weights.get("fav", 0.05) * fav_score
    )
    return round(max(0.0, min(1.0, total)), 4)
