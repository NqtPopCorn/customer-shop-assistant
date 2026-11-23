"""Offline script to build product_attrs.json for the Outfit Knowledge Graph."""

import json
from collections import Counter

import pandas as pd

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.services.outfit_rules import (
    ProductAttrs,
    classify_pattern,
    compute_formality,
    load_outfit_rules,
    resolve_category_role,
)


def build_product_attrs() -> dict[str, ProductAttrs]:
    """Join hm_seed_1000.json with enriched parquet and derive outfit attributes."""
    settings = get_settings()
    data_dir = settings.DATA_DIR
    outfit_dir = data_dir / "outfit_kg"
    rules = load_outfit_rules(outfit_dir)

    seed_path = data_dir / "hm_seed_1000.json"
    parquet_path = data_dir / "hm_ecommerce_products_enriched.parquet"

    with seed_path.open("r", encoding="utf-8") as f:
        seed_products = json.load(f)

    seed_codes = {
        str(item["hmProductCode"])
        for item in seed_products
        if item.get("hmProductCode")
    }
    df = pd.read_parquet(
        parquet_path, columns=["product_code", "graphical_appearance_name"]
    )
    df = df[df["product_code"].isin(seed_codes)]

    pattern_by_code: dict[str, str | None] = {}
    for code, group in df.groupby("product_code"):
        vals = [str(v) for v in group["graphical_appearance_name"].dropna().tolist()]
        pattern_by_code[str(code)] = (
            Counter(vals).most_common(1)[0][0] if vals else None
        )

    result: dict[str, ProductAttrs] = {}
    for item in seed_products:
        resolved = resolve_category_role(item.get("categoryName"), rules)
        if resolved is None:
            continue
        canonical_cat, role = resolved
        slug = str(item["slug"])
        code = str(item.get("hmProductCode") or "")
        pattern_raw = pattern_by_code.get(code)
        pattern_class = classify_pattern(pattern_raw, rules)
        formality = compute_formality(canonical_cat, item.get("description"), rules)

        result[slug] = ProductAttrs(
            slug=slug,
            role=role,
            category=canonical_cat,
            pattern_raw=pattern_raw,
            pattern_class=pattern_class,
            formality=formality,
        )

    out_path = outfit_dir / "product_attrs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {slug: attrs.model_dump() for slug, attrs in result.items()}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    return result


def main() -> None:
    """Run offline attribute build and print distribution summary."""
    attrs_by_slug = build_product_attrs()
    roles = Counter(a.role for a in attrs_by_slug.values())
    patterns = Counter(a.pattern_class for a in attrs_by_slug.values())
    formalities = Counter(a.formality for a in attrs_by_slug.values())

    print(f"Built product_attrs.json: {len(attrs_by_slug)} products")
    print(f"  By role         : {dict(sorted(roles.items()))}")
    print(f"  By pattern_class: {dict(sorted(patterns.items()))}")
    print(f"  By formality    : {dict(sorted(formalities.items()))}")


if __name__ == "__main__":
    main()
