"""Tests for Git-tracked ontology, snapshot manifest, and <30ms RAM index."""

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.loaders.outfit_ontology import (
    load_outfit_ontology,
    resolve_category_id,
    resolve_color_id,
    resolve_occasion_id,
    resolve_style_id,
    validate_ontology,
)
from customer_shop_assistant.schemas.outfit_kg import SnapshotManifest
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)


def _load_build_manifest_fn() -> Any:
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "build_snapshot_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_snapshot_manifest", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_manifest


def test_ontology_and_snapshot_zero_unmapped() -> None:
    settings = get_settings()
    ont_dir = settings.DATA_DIR / "outfit_kg" / "ontology"
    snap_path = settings.DATA_DIR / settings.SNAPSHOT_CATALOG_FILENAME

    ontology = load_outfit_ontology(ont_dir)
    assert len(ontology.categories) == 14
    assert len(ontology.styles) == 6
    assert len(ontology.occasions) == 8
    assert len(ontology.colors) == 33
    assert len(ontology.harmonizes_with) == 54
    assert len(ontology.prefers_color) == 48
    assert len(ontology.suits_color) == 32

    allowed_kinds = {
        "classic_contrast",
        "monochrome_tone",
        "neutral_balance",
        "analogous_soft",
    }
    assert {e.kind for e in ontology.harmonizes_with}.issubset(allowed_kinds)
    covered_colors = {e.source_id for e in ontology.harmonizes_with} | {
        e.target_id for e in ontology.harmonizes_with
    }
    assert covered_colors == {c.id for c in ontology.colors}

    with snap_path.open("r", encoding="utf-8") as f:
        products = json.load(f)

    errors = validate_ontology(ontology, snapshot_products=products)
    assert errors == []

    # Check canonical resolvers
    assert resolve_category_id("Áo sơ mi", ontology) == "ao_so_mi"
    assert resolve_category_id("shirt", ontology) == "ao_so_mi"
    assert resolve_style_id("Smart-casual", ontology) == "smart_casual"
    assert resolve_occasion_id("đi đám cưới", ontology) is None
    assert resolve_occasion_id("đám cưới", ontology) == "party"
    assert resolve_color_id("Xanh navy", ontology) == "xanh_navy"
    assert resolve_color_id("xanh duong", ontology) == "xanh_duong"
    assert resolve_color_id("Nau be", ontology) == "nau_be"


def test_build_snapshot_manifest_and_latency(tmp_path: Path) -> None:
    settings = get_settings()
    ont_dir = settings.DATA_DIR / "outfit_kg" / "ontology"
    snap_path = settings.DATA_DIR / settings.SNAPSHOT_CATALOG_FILENAME
    out_path = tmp_path / "snapshot_manifest.json"

    build_manifest = _load_build_manifest_fn()
    manifest: SnapshotManifest = build_manifest(snap_path, ont_dir, out_path)
    assert out_path.exists()
    assert manifest.total_products == 1000
    assert manifest.in_stock_products == 1000
    assert manifest.blueprint_category_product_count == 740
    assert manifest.non_blueprint_category_product_count == 260
    assert manifest.color_count == 33
    assert manifest.unmapped_categories == []
    assert manifest.unmapped_colors == []
    assert manifest.ontology_errors == []

    # Benchmark SnapshotCatalogIndex counterfactual probing (< 30 ms)
    idx = get_snapshot_catalog_index(settings)
    repo = InMemoryOutfitKGRepository()
    blueprints = repo.get_outfit_blueprints(occasion_id="party", limit=8)

    start = time.perf_counter()
    outcome = idx.probe_blueprints(blueprints, target_gender="MALE", top_k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert outcome.feasible_blueprint_count >= 1
    assert outcome.feasible_product_count >= 2
    assert elapsed_ms < 30.0
