"""Conditional Neo4j parity integration test (runs when Neo4j is up)."""

import pytest

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
    Neo4jOutfitKGRepository,
)


def _try_connect_neo4j() -> Neo4jOutfitKGRepository | None:
    settings = get_settings()
    try:
        repo = Neo4jOutfitKGRepository(
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        repo.verify_connectivity()
        # Ensure ontology has been seeded before parity comparison
        if not repo.get_outfit_blueprints(limit=1):
            repo.close()
            return None
        return repo
    except Exception:
        return None


def test_neo4j_and_inmemory_repositories_match_100_percent() -> None:
    neo4j_repo = _try_connect_neo4j()
    if neo4j_repo is None:
        pytest.skip("Neo4j service not reachable or not seeded on configured Bolt URI.")

    try:
        mem_repo = InMemoryOutfitKGRepository()

        # Verify zero Product / ProductVariant nodes exist in Neo4j
        prod_count_rows, _, _ = neo4j_repo._driver.execute_query(
            "MATCH (n) WHERE n:Product OR n:ProductVariant RETURN count(n) AS cnt",
            database_=neo4j_repo._database,
        )
        assert prod_count_rows[0]["cnt"] == 0

        # Verify v1.3 color relationship counts in Neo4j
        rel_rows, _, _ = neo4j_repo._driver.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt",
            database_=neo4j_repo._database,
        )
        rel_counts = {str(r["rel_type"]): int(r["cnt"]) for r in rel_rows}
        assert rel_counts["HARMONIZES_WITH"] == 54
        assert rel_counts["PREFERS_COLOR"] == 48
        assert rel_counts["SUITS_COLOR"] == 32

        # Branch 1: Style + Occasion
        assert mem_repo.get_outfit_blueprints(
            style_id="smart_casual", occasion_id="office", limit=10
        ) == neo4j_repo.get_outfit_blueprints(
            style_id="smart_casual", occasion_id="office", limit=10
        )

        # Branch 2: Occasion only
        assert mem_repo.get_outfit_blueprints(
            occasion_id="party", limit=10
        ) == neo4j_repo.get_outfit_blueprints(occasion_id="party", limit=10)

        # Branch 3: Anchor only (bidirectional normalization)
        assert mem_repo.get_outfit_blueprints(
            anchor_category_id="chan_vay", limit=10
        ) == neo4j_repo.get_outfit_blueprints(anchor_category_id="chan_vay", limit=10)

        # Color harmony parity (includes kind, rule_id, and reason maps)
        assert mem_repo.get_color_harmony("xanh_navy") == neo4j_repo.get_color_harmony(
            "xanh_navy"
        )

        # Context color palette 70/30 blend parity across all 4 context modes
        for s_id, o_id in (
            ("minimalist", None),
            (None, "party"),
            ("minimalist", "party"),
            ("smart_casual", "cafe"),
        ):
            assert mem_repo.get_context_color_palette(
                style_id=s_id, occasion_id=o_id
            ) == neo4j_repo.get_context_color_palette(style_id=s_id, occasion_id=o_id)
    finally:
        neo4j_repo.close()
