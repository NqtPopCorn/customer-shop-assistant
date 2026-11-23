"""Extend the 300-product H&M dataset to 1,000 stratified products.

Preserves all 300 entries in `data/hm_seed_300.json` (including Golden Eval anchors)
and extracts 700 additional distinct products from H&M Parquet
with matched local archive photos copied to `fashion-shop-be/uploads/products/`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED_300_JSON_PATH = ROOT / "data" / "hm_seed_300.json"
OUTPUT_JSON_PATH = ROOT / "data" / "hm_seed_1000.json"

TARGET_1000_QUOTAS: dict[str, int] = {
    "Áo thun": 100,
    "Áo polo": 75,
    "Áo sơ mi": 100,
    "Áo khoác": 95,
    "Quần jeans": 80,
    "Quần kaki": 75,
    "Quần short": 65,
    "Quần âu": 75,
    "Váy liền": 95,
    "Chân váy": 75,
    "Đầm dạ hội": 70,
    "Mũ nón": 30,
    "Thắt lưng": 30,
    "Túi xách": 35,
}


def _load_seed_300_module() -> ModuleType:
    """Load sibling `prepare_hm_seed_300.py` without modifying sys.path."""
    module_path = Path(__file__).resolve().parent / "prepare_hm_seed_300.py"
    spec = importlib.util.spec_from_file_location("prepare_hm_seed_300", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_stratified_codes(
    cat_candidates: pd.DataFrame, cat_name: str, needed: int
) -> list[str]:
    """Select `needed` product codes balanced across index groups."""
    if needed <= 0:
        return []

    if cat_name not in ("Váy liền", "Chân váy", "Đầm dạ hội"):
        men_pool = cat_candidates[cat_candidates["index_group"] == "Menswear"]
        lady_pool = cat_candidates[cat_candidates["index_group"] == "Ladieswear"]
        other_pool = cat_candidates[
            ~cat_candidates["index_group"].isin(["Menswear", "Ladieswear"])
        ]

        men_target = needed // 3 + (1 if needed % 3 > 0 else 0)
        lady_target = needed // 3 + (1 if needed % 3 > 1 else 0)
        other_target = needed - men_target - lady_target

        picked = [str(c) for c in men_pool["product_code"].head(men_target)]
        picked += [str(c) for c in lady_pool["product_code"].head(lady_target)]
        picked += [str(c) for c in other_pool["product_code"].head(other_target)]

        if len(picked) < needed:
            seen = set(picked)
            remaining = [
                str(c) for c in cat_candidates["product_code"] if str(c) not in seen
            ]
            picked += remaining[: needed - len(picked)]
        return picked[:needed]

    return [str(c) for c in cat_candidates["product_code"].head(needed)]


def _next_unique_seq(
    p_code: str,
    c_idx: int,
    size_label: str,
    seq_state: list[int],
    existing_skus: set[str],
    existing_barcodes: set[str],
) -> tuple[str, str]:
    """Allocate a globally unique SKU and barcode pair."""
    while True:
        seq = seq_state[0]
        seq_state[0] += 1
        sku = f"HM{p_code}-C{c_idx + 1}-{size_label}-{seq}"
        barcode = f"8936{seq:09d}"
        if sku not in existing_skus and barcode not in existing_barcodes:
            existing_skus.add(sku)
            existing_barcodes.add(barcode)
            return sku, barcode


def _build_product_record(
    *,
    base: ModuleType,
    matcher: Any,
    rng: random.Random,
    cat_name: str,
    p_code: str,
    rows: pd.DataFrame,
    idx: int,
    seq_state: list[int],
    existing_slugs: set[str],
    existing_skus: set[str],
    existing_barcodes: set[str],
) -> dict[str, Any]:
    """Build a Vietnamese-enriched product record matching seed-300 schema."""
    first_row = rows.iloc[0]
    target_gender = base.map_gender(str(first_row["index_group_name"]), cat_name, idx)
    profiles = base.CATEGORY_PROFILES[cat_name]
    profile = profiles[idx % len(profiles)]

    tier_idx = idx % len(base.PRICE_TIERS)
    low_p, high_p = base.PRICE_TIERS[tier_idx]
    list_price = rng.randint(low_p // 1000, high_p // 1000) * 1000
    discount_ratio = [0.0, 0.0, 0.1, 0.15, 0.2][idx % 5]
    selling_price = int(round((list_price * (1.0 - discount_ratio)) / 1000) * 1000)

    ga = str(first_row["graphical_appearance_name"])
    raw_name = str(first_row["prod_name"])
    vi_name = base.build_vietnamese_name(
        cat_name,
        target_gender,
        profile["style"],
        ga,
        raw_name,
        p_code,
    )
    digest = hashlib.md5(f"{cat_name}-{p_code}".encode()).hexdigest()[:6]
    slug = f"hm-{p_code}-{digest}"
    while slug in existing_slugs:
        slug = f"{slug}-{idx}"
    existing_slugs.add(slug)

    color_entries: list[dict[str, str]] = []
    seen_vi_colors: set[str] = set()
    images: list[dict[str, Any]] = []

    for img_idx, (_, row) in enumerate(rows.iterrows()):
        raw_col = str(row["colour_group_name"])
        vi_col, hex_code = base.COLOR_MAP.get(raw_col, ("Đen", "#1A1A1A"))
        local_img_url = matcher.pick_and_copy_image(cat_name, target_gender, vi_col)

        images.append(
            {
                "url": local_img_url,
                "altText": f"{vi_name} - Màu {vi_col} ({img_idx + 1})",
                "sortOrder": img_idx,
            }
        )
        if vi_col not in seen_vi_colors:
            seen_vi_colors.add(vi_col)
            color_entries.append(
                {
                    "color": vi_col,
                    "colorHex": hex_code,
                    "imageUrl": local_img_url,
                }
            )

    if len(color_entries) < 2:
        for fb_col, fb_hex in base.FALLBACK_COLORS:
            if fb_col not in seen_vi_colors:
                seen_vi_colors.add(fb_col)
                fb_img_url = matcher.pick_and_copy_image(
                    cat_name, target_gender, fb_col
                )
                images.append(
                    {
                        "url": fb_img_url,
                        "altText": f"{vi_name} - Màu {fb_col} ({len(images) + 1})",
                        "sortOrder": len(images),
                    }
                )
                color_entries.append(
                    {
                        "color": fb_col,
                        "colorHex": fb_hex,
                        "imageUrl": fb_img_url,
                    }
                )
                break

    sizes = base.get_sizes_for_product(cat_name, target_gender)
    variants: list[dict[str, Any]] = []
    for c_idx, c_entry in enumerate(color_entries):
        for s_idx, size_label in enumerate(sizes):
            sku, barcode_num = _next_unique_seq(
                p_code=p_code,
                c_idx=c_idx,
                size_label=size_label,
                seq_state=seq_state,
                existing_skus=existing_skus,
                existing_barcodes=existing_barcodes,
            )
            is_oos = ((idx + c_idx * 3 + s_idx * 7) % 10) == 0
            stock = 0 if is_oos else rng.randint(12, 95)
            variants.append(
                {
                    "sku": sku,
                    "barcode": barcode_num,
                    "size": size_label,
                    "color": c_entry["color"],
                    "colorHex": c_entry["colorHex"],
                    "imageUrl": c_entry["imageUrl"],
                    "listPrice": list_price,
                    "sellingPrice": selling_price,
                    "stock": stock,
                    "weight": rng.randint(180, 650),
                }
            )

    gender_vi = {
        "MALE": "nam giới",
        "FEMALE": "nữ giới",
        "UNISEX": "cả nam và nữ (Unisex)",
    }[target_gender]

    description = (
        f"{vi_name} (Mã H&M #{p_code}) dành cho {gender_vi}, "
        f"được chế tác từ chất liệu {profile['material']}. "
        f"Thiết kế phù hợp để {profile['occasion']}. "
        f"Ưu điểm phom dáng: {profile['body_fit']}. "
        f"Mô tả chi tiết gốc: {first_row['detail_desc']}"
    )
    brand_name = "H&M" if idx % 3 != 2 else base.BRANDS[idx % len(base.BRANDS)]

    return {
        "hmProductCode": p_code,
        "name": vi_name,
        "slug": slug,
        "categoryName": cat_name,
        "brandName": brand_name,
        "targetGender": target_gender,
        "material": profile["material"],
        "careInstructions": profile["care"],
        "season": profile["season"],
        "provider": "H&M Global Catalog",
        "description": description,
        "images": images,
        "variants": variants,
    }


def main() -> None:
    """Generate `data/hm_seed_1000.json` preserving the initial 300 products."""
    base = _load_seed_300_module()
    rng = random.Random(20261123)
    matcher = base.LocalArchiveImageMatcher()

    existing_products: list[dict[str, Any]] = json.loads(
        SEED_300_JSON_PATH.read_text(encoding="utf-8")
    )
    if len(existing_products) != 300:
        raise ValueError(
            f"Expected 300 products in {SEED_300_JSON_PATH}, "
            f"got {len(existing_products)}"
        )

    existing_counts = Counter(str(p["categoryName"]) for p in existing_products)
    global_seen_codes = {str(p["hmProductCode"]) for p in existing_products}
    existing_slugs = {str(p["slug"]) for p in existing_products}
    existing_skus = {str(v["sku"]) for p in existing_products for v in p["variants"]}
    existing_barcodes = {
        str(v["barcode"]) for p in existing_products for v in p["variants"]
    }

    df = pd.read_parquet(base.PARQUET_PATH)
    adult = df[
        df["index_group_name"].isin(["Ladieswear", "Menswear", "Divided", "Sport"])
        & df["image_url"].notna()
        & df["detail_desc"].notna()
    ].copy()
    adult["mapped_cat"] = adult.apply(base.map_category, axis=1)
    adult = adult.dropna(subset=["mapped_cat"])
    adult["product_code_str"] = adult["product_code"].astype(str)

    grouped_meta = (
        adult.groupby(["mapped_cat", "product_code_str"])
        .agg(
            article_count=("article_id", "count"),
            index_group=("index_group_name", "first"),
        )
        .reset_index()
        .rename(columns={"product_code_str": "product_code"})
    )

    seq_state = [100_001]
    new_products: list[dict[str, Any]] = []

    for cat_name, target_quota in TARGET_1000_QUOTAS.items():
        base_count = existing_counts.get(cat_name, 0)
        needed = max(0, target_quota - base_count)
        if needed == 0:
            continue

        cat_candidates = grouped_meta[
            (grouped_meta["mapped_cat"] == cat_name)
            & (~grouped_meta["product_code"].isin(global_seen_codes))
        ].copy()
        cat_candidates["priority"] = cat_candidates["article_count"].clip(upper=4)
        cat_candidates = cat_candidates.sort_values(
            by=["priority", "product_code"], ascending=[False, True]
        )

        picked_codes = _select_stratified_codes(cat_candidates, cat_name, needed)
        if len(picked_codes) < needed:
            raise RuntimeError(
                f"Insufficient candidates for {cat_name!r}: "
                f"needed {needed}, got {len(picked_codes)}"
            )
        global_seen_codes.update(picked_codes)

        for offset, p_code in enumerate(picked_codes):
            rows = adult[
                (adult["mapped_cat"] == cat_name)
                & (adult["product_code_str"] == p_code)
            ].head(3)
            product_record = _build_product_record(
                base=base,
                matcher=matcher,
                rng=rng,
                cat_name=cat_name,
                p_code=p_code,
                rows=rows,
                idx=base_count + offset,
                seq_state=seq_state,
                existing_slugs=existing_slugs,
                existing_skus=existing_skus,
                existing_barcodes=existing_barcodes,
            )
            new_products.append(product_record)

    all_products = [*existing_products, *new_products]
    if len(all_products) != 1000:
        raise AssertionError(f"Expected 1000 products, got {len(all_products)}")
    if len({p["slug"] for p in all_products}) != 1000:
        raise AssertionError("Duplicate slug detected in 1000-product seed!")

    all_skus = [v["sku"] for p in all_products for v in p["variants"]]
    all_barcodes = [v["barcode"] for p in all_products for v in p["variants"]]
    if len(set(all_skus)) != len(all_skus):
        raise AssertionError("Duplicate SKU detected in 1000-product seed!")
    if len(set(all_barcodes)) != len(all_barcodes):
        raise AssertionError("Duplicate barcode detected in 1000-product seed!")

    OUTPUT_JSON_PATH.write_text(
        json.dumps(all_products, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cat_summary = dict(Counter(str(p["categoryName"]) for p in all_products))
    gender_summary = dict(Counter(str(p["targetGender"]) for p in all_products))
    print(
        f"Exported {len(all_products)} products "
        f"(+{len(new_products)} new) to {OUTPUT_JSON_PATH}"
    )
    print(f"Copied {len(matcher.copied_ids)} archive images to {base.BE_UPLOADS_DIR}")
    print(f"Categories: {cat_summary}")
    print(f"Genders: {gender_summary}")


if __name__ == "__main__":
    main()
