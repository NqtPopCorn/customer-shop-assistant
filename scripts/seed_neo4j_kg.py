"""Pre-seed validator & idempotent Neo4j seeder for Category-level Fashion KG."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.loaders.outfit_ontology import (
    load_outfit_ontology,
    validate_ontology,
)

_CONSTRAINTS = (
    "CREATE CONSTRAINT category_id IF NOT EXISTS "
    "FOR (n:Category) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT style_id IF NOT EXISTS FOR (n:Style) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT occasion_id IF NOT EXISTS "
    "FOR (n:Occasion) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT color_id IF NOT EXISTS FOR (n:Color) REQUIRE n.id IS UNIQUE",
)


def _load_snapshot_products(snapshot_path: Path) -> list[dict[str, Any]]:
    with snapshot_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("items", raw.get("products", []))


def seed_neo4j_kg(
    *,
    ontology_dir: Path,
    snapshot_path: Path,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    reset: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate ontology & snapshot coverage, then idempotently seed Neo4j."""
    ontology = load_outfit_ontology(ontology_dir)
    products = _load_snapshot_products(snapshot_path)
    errors = validate_ontology(ontology, snapshot_products=products)
    if errors:
        raise ValueError("Ontology validation failed:\n- " + "\n- ".join(errors))

    summary: dict[str, Any] = {
        "categories": len(ontology.categories),
        "styles": len(ontology.styles),
        "occasions": len(ontology.occasions),
        "colors": len(ontology.colors),
        "fits_style": len(ontology.fits_style),
        "suits_category": len(ontology.suits_category),
        "recommends_category": len(ontology.recommends_category),
        "pairs_with": len(ontology.pairs_with),
        "harmonizes_with": len(ontology.harmonizes_with),
        "prefers_color": len(ontology.prefers_color),
        "suits_color": len(ontology.suits_color),
        "validated": True,
        "seeded": False,
    }
    if validate_only:
        return summary

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            if reset:
                session.run(
                    "MATCH (n) WHERE n:Category OR n:Style OR n:Occasion OR n:Color "
                    "DETACH DELETE n"
                )

            for stmt in _CONSTRAINTS:
                session.run(stmt)

            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Category {id: row.id})
                SET n.name = row.name,
                    n.aliases = row.aliases,
                    n.role = row.role,
                    n.base_formality = row.base_formality,
                    n.supports_top_bottom_blueprint = row.supports_top_bottom_blueprint
                """,
                rows=[c.model_dump() for c in ontology.categories],
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Style {id: row.id})
                SET n.name = row.name,
                    n.aliases = row.aliases,
                    n.formality_min = row.formality_min,
                    n.formality_max = row.formality_max,
                    n.description = row.description
                """,
                rows=[s.model_dump() for s in ontology.styles],
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Occasion {id: row.id})
                SET n.name = row.name,
                    n.slot_code = row.slot_code,
                    n.aliases = row.aliases,
                    n.formality_min = row.formality_min,
                    n.formality_max = row.formality_max
                """,
                rows=[o.model_dump() for o in ontology.occasions],
            )
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Color {id: row.id})
                SET n.name = row.name,
                    n.aliases = row.aliases,
                    n.family = row.family,
                    n.neutral = row.neutral,
                    n.lightness = row.lightness
                """,
                rows=[c.model_dump() for c in ontology.colors],
            )

            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Occasion {id: row.source_id}), (b:Style {id: row.target_id})
                MERGE (a)-[r:FITS_STYLE {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.is_default = row.is_default,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.fits_style],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Occasion {id: row.source_id}), (b:Category {id: row.target_id})
                MERGE (a)-[r:SUITS_CATEGORY {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.suits_category],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Style {id: row.source_id}), (b:Category {id: row.target_id})
                MERGE (a)-[r:RECOMMENDS_CATEGORY {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.recommends_category],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Category {id: row.source_id}), (b:Category {id: row.target_id})
                MERGE (a)-[r:PAIRS_WITH {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.style_ids = row.style_ids,
                    r.occasion_ids = row.occasion_ids,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.pairs_with],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Color {id: row.source_id}), (b:Color {id: row.target_id})
                MERGE (a)-[r:HARMONIZES_WITH {rule_id: row.rule_id}]->(b)
                SET r.kind = row.kind,
                    r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.harmonizes_with],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Style {id: row.source_id}), (b:Color {id: row.target_id})
                MERGE (a)-[r:PREFERS_COLOR {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.prefers_color],
            )
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Occasion {id: row.source_id}), (b:Color {id: row.target_id})
                MERGE (a)-[r:SUITS_COLOR {rule_id: row.rule_id}]->(b)
                SET r.compatibility = row.compatibility,
                    r.confidence = row.confidence,
                    r.reason = row.reason,
                    r.source = row.source
                """,
                rows=[e.model_dump() for e in ontology.suits_color],
            )

            node_counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            ).data()
            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt"
            ).data()
            summary["neo4j_nodes"] = {r["label"]: r["cnt"] for r in node_counts}
            summary["neo4j_relationships"] = {r["rel"]: r["cnt"] for r in rel_counts}
            summary["seeded"] = True
    finally:
        driver.close()

    return summary


def main() -> int:
    """CLI entrypoint for validating and seeding Category-level Fashion KG."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Validate and idempotently seed Category-level Fashion KG."
    )
    parser.add_argument(
        "--ontology-dir",
        type=Path,
        default=settings.DATA_DIR / "outfit_kg" / "ontology",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=settings.DATA_DIR / settings.SNAPSHOT_CATALOG_FILENAME,
    )
    parser.add_argument("--reset", action="store_true", help="Clear core nodes first")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run ontology & snapshot validation without connecting to Neo4j",
    )
    args = parser.parse_args()

    try:
        summary = seed_neo4j_kg(
            ontology_dir=args.ontology_dir,
            snapshot_path=args.snapshot,
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
            reset=args.reset,
            validate_only=args.validate_only,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
