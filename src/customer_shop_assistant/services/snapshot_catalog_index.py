"""Read-only in-memory index of the Fixed 1,000-Product Research Snapshot."""

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.loaders.outfit_ontology import (
    get_cached_ontology,
    resolve_category_id,
    resolve_color_id,
)
from customer_shop_assistant.schemas.outfit_kg import (
    CandidateOutcome,
    ColorHarmonyAdvice,
    OutfitBlueprint,
    OutfitOntology,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotProductRecord:
    """Indexed product record with canonical category_id and variant color_ids."""

    deterministic_id: int
    slug: str
    hm_product_code: str
    name: str
    category_id: str
    category_name: str
    target_gender: str
    brand_name: str | None
    min_price: float
    max_price: float
    in_stock_color_ids: frozenset[str]
    all_color_ids: frozenset[str]
    color_price_map: dict[str, float] = field(default_factory=dict)
    color_name_by_id: dict[str, str] = field(default_factory=dict)
    color_image_by_id: dict[str, str | None] = field(default_factory=dict)
    color_variant_id_by_id: dict[str, int] = field(default_factory=dict)
    color_size_by_id: dict[str, str | None] = field(default_factory=dict)
    primary_image_url: str | None = None
    in_stock: bool = True

    def matches_gender(self, target_gender: str | None) -> bool:
        """Return True if product fits requested target_gender (UNISEX matches both)."""
        if not target_gender:
            return True
        g = target_gender.strip().upper()
        if g == "UNISEX":
            return True
        return self.target_gender in (g, "UNISEX")

    def lowest_price_for_colors(self, color_ids: set[str] | None = None) -> float:
        """Return lowest in-stock price matching allowed color_ids, else min_price."""
        if color_ids:
            matched = [
                p
                for cid, p in self.color_price_map.items()
                if cid in color_ids and cid in self.in_stock_color_ids
            ]
            if matched:
                return min(matched)
        return self.min_price


class SnapshotCatalogIndex:
    """Fast read-only RAM index over data/hm_seed_1000.json (< 30 ms probing)."""

    def __init__(
        self,
        records: list[SnapshotProductRecord],
        ontology: OutfitOntology,
    ) -> None:
        self.records = records
        self.ontology = ontology
        self.by_id: dict[int, SnapshotProductRecord] = {
            r.deterministic_id: r for r in records
        }
        self.by_slug: dict[str, SnapshotProductRecord] = {r.slug: r for r in records}
        self.by_category: dict[str, list[SnapshotProductRecord]] = {}
        for r in records:
            self.by_category.setdefault(r.category_id, []).append(r)

    @classmethod
    def from_json_path(
        cls,
        snapshot_path: Path,
        ontology: OutfitOntology | None = None,
    ) -> "SnapshotCatalogIndex":
        """Load snapshot JSON and build deterministic canonical indices."""
        ont = ontology or get_cached_ontology()
        with snapshot_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        items: list[dict[str, Any]] = (
            raw if isinstance(raw, list) else raw.get("items", raw.get("products", []))
        )

        records: list[SnapshotProductRecord] = []
        for idx, item in enumerate(items, start=1):
            cat_name = str(item.get("categoryName") or "")
            cat_id = resolve_category_id(cat_name, ont) or ""
            slug = str(item.get("slug") or f"prod-{idx}")
            gender = str(item.get("targetGender") or "UNISEX").upper()

            in_stock_cols: set[str] = set()
            all_cols: set[str] = set()
            color_prices: dict[str, float] = {}
            color_names: dict[str, str] = {}
            color_images: dict[str, str | None] = {}
            color_variant_ids: dict[str, int] = {}
            color_sizes: dict[str, str | None] = {}
            color_best_rank: dict[str, tuple[bool, bool, float, int]] = {}
            prices: list[float] = []

            for v_idx, variant in enumerate(item.get("variants") or []):
                raw_col = str(variant.get("color") or "").strip()
                cid = resolve_color_id(raw_col, ont)
                if cid is None:
                    continue
                all_cols.add(cid)
                color_names.setdefault(cid, raw_col)
                raw_img = variant.get("imageUrl")
                img_url = str(raw_img) if raw_img else None
                if img_url and cid not in color_images:
                    color_images[cid] = img_url

                raw_price = variant.get(
                    "finalPrice",
                    variant.get("sellingPrice", variant.get("listPrice", 0)),
                )
                price = float(raw_price or 0.0)
                prices.append(price)

                stock = int(variant.get("stock") or 0)
                if stock > 0:
                    in_stock_cols.add(cid)
                    if cid not in color_prices or price < color_prices[cid]:
                        color_prices[cid] = price

                rank = (not (stock > 0), not bool(img_url), price, v_idx)
                if cid not in color_best_rank or rank < color_best_rank[cid]:
                    color_best_rank[cid] = rank
                    raw_vid = variant.get("id")
                    if raw_vid is not None:
                        try:
                            color_variant_ids[cid] = int(raw_vid)
                        except (TypeError, ValueError):
                            pass
                    raw_size = variant.get("size")
                    color_sizes[cid] = str(raw_size) if raw_size else None
                    if img_url:
                        color_images[cid] = img_url

            images = item.get("images") or []
            primary_img = (
                str(images[0]["url"]) if images and images[0].get("url") else None
            )
            det_id = int(item.get("id") or idx)

            records.append(
                SnapshotProductRecord(
                    deterministic_id=det_id,
                    slug=slug,
                    hm_product_code=str(item.get("hmProductCode") or ""),
                    name=str(item.get("name") or slug),
                    category_id=cat_id,
                    category_name=cat_name,
                    target_gender=gender,
                    brand_name=(
                        str(item["brandName"]) if item.get("brandName") else None
                    ),
                    min_price=min(prices) if prices else 0.0,
                    max_price=max(prices) if prices else 0.0,
                    in_stock_color_ids=frozenset(in_stock_cols),
                    all_color_ids=frozenset(all_cols),
                    color_price_map=color_prices,
                    color_name_by_id=color_names,
                    color_image_by_id=color_images,
                    color_variant_id_by_id=color_variant_ids,
                    color_size_by_id=color_sizes,
                    primary_image_url=primary_img,
                    in_stock=bool(in_stock_cols),
                )
            )

        return cls(records=records, ontology=ont)

    def query_products(
        self,
        category_id: str,
        *,
        target_gender: str | None = None,
        color_ids: set[str] | None = None,
        disliked_color_ids: set[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        in_stock_only: bool = True,
    ) -> list[SnapshotProductRecord]:
        """Filter indexed snapshot products by hard constraints in RAM."""
        disliked = disliked_color_ids or set()
        results: list[SnapshotProductRecord] = []

        for rec in self.by_category.get(category_id, []):
            if in_stock_only and not rec.in_stock:
                continue
            if not rec.matches_gender(target_gender):
                continue

            avail_colors = (
                rec.in_stock_color_ids if in_stock_only else rec.all_color_ids
            )
            valid_colors = avail_colors - disliked
            if not valid_colors:
                continue
            if color_ids and not (valid_colors & color_ids):
                continue

            eff_price = rec.lowest_price_for_colors(
                (valid_colors & color_ids) if color_ids else valid_colors
            )
            if min_price is not None and eff_price < min_price:
                continue
            if max_price is not None and eff_price > max_price:
                continue
            results.append(rec)

        return results

    def probe_blueprints(
        self,
        blueprints: list[OutfitBlueprint],
        *,
        candidate_value: Any = None,
        target_gender: str | None = None,
        anchor_product_id: int | None = None,
        anchor_color_id: str | None = None,
        color_advice: ColorHarmonyAdvice | None = None,
        disliked_color_ids: set[str] | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        top_k: int = 5,
    ) -> CandidateOutcome:
        """Run fast counterfactual probing for a candidate value over KG blueprints."""
        disliked = disliked_color_ids or set()
        harmony_set = (
            set(color_advice.recommended_color_ids)
            if color_advice and color_advice.available
            else set()
        )
        anchor_rec = self.by_id.get(anchor_product_id) if anchor_product_id else None

        # Per-item price budget check when max_price is set for combo vs item
        per_item_max = max_price
        feasible_pairs: list[tuple[str, str]] = []
        top_k_ids: list[str] = []
        seen_top_ids: set[str] = set()
        feasible_product_ids: set[int] = set()
        applied_rules: list[str] = []
        seen_rules: set[str] = set()

        for bp in blueprints:
            if anchor_rec is not None and anchor_rec.category_id == bp.top_category_id:
                tops = [anchor_rec] if anchor_rec.matches_gender(target_gender) else []
            else:
                top_cols = {anchor_color_id} if anchor_color_id else None
                tops = self.query_products(
                    bp.top_category_id,
                    target_gender=target_gender,
                    color_ids=top_cols,
                    disliked_color_ids=disliked,
                    min_price=min_price,
                    max_price=per_item_max,
                )
                if not tops and top_cols:
                    # Retry without strict anchor color to match blueprint feasibility
                    tops = self.query_products(
                        bp.top_category_id,
                        target_gender=target_gender,
                        disliked_color_ids=disliked,
                        min_price=min_price,
                        max_price=per_item_max,
                    )

            if (
                anchor_rec is not None
                and anchor_rec.category_id == bp.bottom_category_id
            ):
                bots = [anchor_rec] if anchor_rec.matches_gender(target_gender) else []
            else:
                bot_cols = harmony_set if harmony_set else None
                bots = self.query_products(
                    bp.bottom_category_id,
                    target_gender=target_gender,
                    color_ids=bot_cols,
                    disliked_color_ids=disliked,
                    min_price=min_price,
                    max_price=per_item_max,
                )
                if not bots and bot_cols:
                    bots = self.query_products(
                        bp.bottom_category_id,
                        target_gender=target_gender,
                        disliked_color_ids=disliked,
                        min_price=min_price,
                        max_price=per_item_max,
                    )

            if not tops or not bots:
                continue

            feasible_pairs.append((bp.top_category_id, bp.bottom_category_id))
            for r_id in bp.applied_rule_ids:
                if r_id not in seen_rules:
                    seen_rules.add(r_id)
                    applied_rules.append(r_id)

            for t_rec in tops:
                feasible_product_ids.add(t_rec.deterministic_id)
            for b_rec in bots:
                feasible_product_ids.add(b_rec.deterministic_id)

            if len(top_k_ids) < top_k:
                pair_token = f"{tops[0].deterministic_id}+{bots[0].deterministic_id}"
                if pair_token not in seen_top_ids:
                    seen_top_ids.add(pair_token)
                    top_k_ids.append(pair_token)

        return CandidateOutcome(
            candidate_value=candidate_value,
            feasible_product_count=len(feasible_product_ids),
            feasible_blueprint_count=len(feasible_pairs),
            top_k_ids=top_k_ids[:top_k],
            category_pairs=feasible_pairs,
            applied_rule_ids=applied_rules,
        )


@lru_cache(maxsize=2)
def _cached_snapshot_index(
    snapshot_path_str: str, ontology_dir_str: str
) -> SnapshotCatalogIndex:
    ont = get_cached_ontology(ontology_dir_str)
    return SnapshotCatalogIndex.from_json_path(Path(snapshot_path_str), ontology=ont)


def get_snapshot_catalog_index(
    settings: Settings | None = None,
) -> SnapshotCatalogIndex:
    """Return cached SnapshotCatalogIndex configured by settings."""
    cfg = settings or get_settings()
    snap_path = (cfg.DATA_DIR / cfg.SNAPSHOT_CATALOG_FILENAME).resolve()
    ont_dir = (cfg.DATA_DIR / "outfit_kg" / "ontology").resolve()
    return _cached_snapshot_index(str(snap_path), str(ont_dir))


def clear_snapshot_index_cache() -> None:
    """Clear cached SnapshotCatalogIndex (useful for tests)."""
    _cached_snapshot_index.cache_clear()
