"""Build Fixed 1,000-Product Snapshot manifest & validate ontology coverage."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.loaders.outfit_ontology import (
    load_outfit_ontology,
    resolve_category_id,
    resolve_color_id,
    validate_ontology,
)
from customer_shop_assistant.schemas.outfit_kg import SnapshotManifest


def build_manifest(
    snapshot_path: Path,
    ontology_dir: Path,
    output_path: Path,
) -> SnapshotManifest:
    """Validate 1,000-product snapshot coverage and write snapshot_manifest.json."""
    sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    with snapshot_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    products: list[dict[str, Any]] = (
        raw if isinstance(raw, list) else raw.get("items", raw.get("products", []))
    )

    ontology = load_outfit_ontology(ontology_dir)
    ontology_errors = validate_ontology(ontology, snapshot_products=products)

    cat_by_id = {c.id: c for c in ontology.categories}
    gender_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    unmapped_cats: set[str] = set()
    unmapped_cols: set[str] = set()
    seen_colors: set[str] = set()
    all_prices: list[float] = []
    in_stock_count = 0
    blueprint_prod_count = 0
    non_blueprint_prod_count = 0

    for prod in products:
        gender = str(prod.get("targetGender") or "UNKNOWN").upper()
        gender_counts[gender] += 1

        cat_name = str(prod.get("categoryName") or "")
        category_counts[cat_name] += 1
        cat_id = resolve_category_id(cat_name, ontology)
        if cat_id is None:
            unmapped_cats.add(cat_name)
        else:
            node = cat_by_id[cat_id]
            if node.supports_top_bottom_blueprint:
                blueprint_prod_count += 1
            else:
                non_blueprint_prod_count += 1

        has_stock = False
        for variant in prod.get("variants") or []:
            raw_col = str(variant.get("color") or "").strip()
            col_id = resolve_color_id(raw_col, ontology)
            if col_id is None:
                unmapped_cols.add(raw_col)
            else:
                seen_colors.add(col_id)
            if int(variant.get("stock") or 0) > 0:
                has_stock = True
            price = float(
                variant.get(
                    "finalPrice",
                    variant.get("sellingPrice", variant.get("listPrice", 0)),
                )
                or 0.0
            )
            if price > 0:
                all_prices.append(price)
        if has_stock:
            in_stock_count += 1

    sorted_prices = sorted(all_prices)
    n_p = len(sorted_prices)
    quantiles = (
        {
            "p25": sorted_prices[int(0.25 * (n_p - 1))],
            "p50": sorted_prices[int(0.50 * (n_p - 1))],
            "p75": sorted_prices[int(0.75 * (n_p - 1))],
        }
        if n_p
        else {}
    )

    manifest = SnapshotManifest(
        snapshot_file=snapshot_path.name,
        sha256=sha256,
        total_products=len(products),
        in_stock_products=in_stock_count,
        price_min=min(sorted_prices) if sorted_prices else 0.0,
        price_max=max(sorted_prices) if sorted_prices else 0.0,
        price_quantiles=quantiles,
        gender_counts=dict(sorted(gender_counts.items())),
        category_counts=dict(sorted(category_counts.items())),
        blueprint_category_product_count=blueprint_prod_count,
        non_blueprint_category_product_count=non_blueprint_prod_count,
        color_count=len(seen_colors),
        unmapped_categories=sorted(unmapped_cats),
        unmapped_colors=sorted(unmapped_cols),
        ontology_errors=ontology_errors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, ensure_ascii=False, indent=2)

    return manifest


def main() -> int:
    """CLI entrypoint for building and validating the snapshot manifest."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Validate ontology coverage and build snapshot_manifest.json."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=settings.DATA_DIR / settings.SNAPSHOT_CATALOG_FILENAME,
    )
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=settings.DATA_DIR / "outfit_kg" / "ontology",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.DATA_DIR / "outfit_kg" / "snapshot_manifest.json",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.snapshot, args.ontology_dir, args.output)
    print(
        f"Snapshot '{manifest.snapshot_file}': {manifest.total_products} products, "
        f"{manifest.in_stock_products} in-stock, {manifest.color_count} colors, "
        f"blueprint={manifest.blueprint_category_product_count}, "
        f"non_blueprint={manifest.non_blueprint_category_product_count}."
    )
    print(f"Manifest saved to: {args.output}")

    if (
        manifest.unmapped_categories
        or manifest.unmapped_colors
        or manifest.ontology_errors
    ):
        print("VALIDATION ERRORS:", file=sys.stderr)
        for err in manifest.ontology_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
