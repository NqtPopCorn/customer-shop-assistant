"""Pydantic v2 data contracts for Category-level Outfit KG & Counterfactual Policy."""

from typing import Any, Literal

from pydantic import BaseModel, Field

CategoryRole = Literal["TOP", "OUTERWEAR", "BOTTOM", "ONE_PIECE", "ACCESSORY"]
ColorHarmonyKind = Literal[
    "classic_contrast",
    "monochrome_tone",
    "neutral_balance",
    "analogous_soft",
]


class CategoryNode(BaseModel):
    """Canonical (:Category) node in the fashion ontology."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: CategoryRole
    base_formality: int = Field(..., ge=1, le=5)
    supports_top_bottom_blueprint: bool = True


class StyleNode(BaseModel):
    """Canonical (:Style) node in the fashion ontology."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    formality_min: int = Field(default=1, ge=1, le=5)
    formality_max: int = Field(default=5, ge=1, le=5)
    description: str = ""


class OccasionNode(BaseModel):
    """Canonical (:Occasion) node aligned 1-to-1 with slot_tracker slot_code."""

    id: str
    name: str
    slot_code: Literal["office", "party", "casual", "sport"]
    aliases: list[str] = Field(default_factory=list)
    formality_min: int = Field(default=1, ge=1, le=5)
    formality_max: int = Field(default=5, ge=1, le=5)


class ColorNode(BaseModel):
    """Canonical (:Color) node matching the 33 snapshot colors."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    family: str
    neutral: bool = False
    lightness: Literal["light", "mid", "dark"]


class BaseOntologyEdge(BaseModel):
    """Provenance fields required on every KG relationship."""

    source_id: str
    target_id: str
    rule_id: str
    compatibility: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    source: str


class FitsStyleEdge(BaseOntologyEdge):
    """(:Occasion)-[:FITS_STYLE]->(:Style) edge."""

    is_default: bool = False


class SuitsCategoryEdge(BaseOntologyEdge):
    """(:Occasion)-[:SUITS_CATEGORY]->(:Category) edge."""


class RecommendsCategoryEdge(BaseOntologyEdge):
    """(:Style)-[:RECOMMENDS_CATEGORY]->(:Category) edge."""


class PairsWithEdge(BaseOntologyEdge):
    """(:Category)-[:PAIRS_WITH]->(:Category) edge (TOP/OUTERWEAR -> BOTTOM)."""

    style_ids: list[str] = Field(default_factory=list)
    occasion_ids: list[str] = Field(default_factory=list)


class HarmonizesWithEdge(BaseOntologyEdge):
    """(:Color)-[:HARMONIZES_WITH]->(:Color) edge (c1.id < c2.id)."""

    kind: ColorHarmonyKind


class PrefersColorEdge(BaseOntologyEdge):
    """(:Style)-[:PREFERS_COLOR]->(:Color) edge."""


class SuitsColorEdge(BaseOntologyEdge):
    """(:Occasion)-[:SUITS_COLOR]->(:Color) edge."""


class OutfitOntology(BaseModel):
    """Complete Git-tracked Category-level Fashion KG ontology."""

    categories: list[CategoryNode] = Field(default_factory=list)
    styles: list[StyleNode] = Field(default_factory=list)
    occasions: list[OccasionNode] = Field(default_factory=list)
    colors: list[ColorNode] = Field(default_factory=list)
    fits_style: list[FitsStyleEdge] = Field(default_factory=list)
    suits_category: list[SuitsCategoryEdge] = Field(default_factory=list)
    recommends_category: list[RecommendsCategoryEdge] = Field(default_factory=list)
    pairs_with: list[PairsWithEdge] = Field(default_factory=list)
    harmonizes_with: list[HarmonizesWithEdge] = Field(default_factory=list)
    prefers_color: list[PrefersColorEdge] = Field(default_factory=list)
    suits_color: list[SuitsColorEdge] = Field(default_factory=list)


class OutfitBlueprint(BaseModel):
    """Category-level outfit blueprint returned by OutfitKGRepository."""

    top_category_id: str
    bottom_category_id: str
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    applied_rule_ids: list[str] = Field(default_factory=list)


class ColorHarmonyAdvice(BaseModel):
    """Decoupled color harmony advice returned by OutfitKGRepository."""

    anchor_color_id: str
    recommended_color_ids: list[str] = Field(default_factory=list)
    compatibility_by_color: dict[str, float] = Field(default_factory=dict)
    kind_by_color: dict[str, ColorHarmonyKind] = Field(default_factory=dict)
    rule_id_by_color: dict[str, str] = Field(default_factory=dict)
    reason_by_color: dict[str, str] = Field(default_factory=dict)
    applied_rule_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    available: bool = True


class ContextColorPalette(BaseModel):
    """70/30 normalized color palette from Style and Occasion preferences."""

    style_id: str | None = None
    occasion_id: str | None = None
    scores_by_color: dict[str, float] = Field(default_factory=dict)
    style_scores_by_color: dict[str, float] = Field(default_factory=dict)
    occasion_scores_by_color: dict[str, float] = Field(default_factory=dict)
    rule_ids_by_color: dict[str, list[str]] = Field(default_factory=dict)
    reasons_by_color: dict[str, list[str]] = Field(default_factory=dict)


class ColorScoreBreakdown(BaseModel):
    """Transparent debug breakdown of Color-aware Outfit Scoring (SPEC v1.3)."""

    color_score: float = Field(..., ge=0.0, le=1.0)
    pair_harmony: float = Field(..., ge=0.0, le=1.0)
    harmony_kind: ColorHarmonyKind | None = None
    context_color_fit: float = Field(..., ge=0.0, le=1.0)
    style_color_fit: float | None = None
    occasion_color_fit: float | None = None
    user_color_match: float | None = None
    top_color_id: str | None = None
    bottom_color_id: str | None = None
    color_rule_ids: list[str] = Field(default_factory=list)
    unmapped_colors: list[str] = Field(default_factory=list)


class CandidateOutcome(BaseModel):
    """Snapshot probing summary for a single counterfactual candidate value."""

    candidate_value: Any
    feasible_product_count: int = Field(..., ge=0)
    feasible_blueprint_count: int = Field(..., ge=0)
    top_k_ids: list[str] = Field(default_factory=list)
    category_pairs: list[tuple[str, str]] = Field(default_factory=list)
    applied_rule_ids: list[str] = Field(default_factory=list)
    mean_color_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SlotImpactSignals(BaseModel):
    """Counterfactual sensitivity signals computed across candidate values."""

    feasibility_change: bool
    category_shift: bool
    mean_top_k_jaccard: float = Field(..., ge=0.0, le=1.0)
    can_cover_with_alternatives: bool
    color_quality_gain: float = Field(default=0.0)


class SnapshotManifest(BaseModel):
    """Reproducibility manifest and coverage report for Fixed 1,000-Product Snapshot."""

    snapshot_file: str
    sha256: str
    total_products: int
    in_stock_products: int
    price_min: float
    price_max: float
    price_quantiles: dict[str, float] = Field(default_factory=dict)
    gender_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)
    blueprint_category_product_count: int
    non_blueprint_category_product_count: int
    color_count: int
    unmapped_categories: list[str] = Field(default_factory=list)
    unmapped_colors: list[str] = Field(default_factory=list)
    ontology_errors: list[str] = Field(default_factory=list)
