"""Dual OutfitKGRepository (InMemory & Neo4j) implementing Method A scoring."""

import logging
from typing import Any, Protocol

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.loaders.outfit_ontology import get_cached_ontology
from customer_shop_assistant.schemas.outfit_kg import (
    CategoryNode,
    ColorHarmonyAdvice,
    OutfitBlueprint,
    OutfitOntology,
)

logger = logging.getLogger(__name__)


class OutfitKGRepository(Protocol):
    """Protocol for Category-level Fashion KG queries across backends."""

    def get_outfit_blueprints(
        self,
        style_id: str | None = None,
        occasion_id: str | None = None,
        anchor_category_id: str | None = None,
        limit: int = 10,
    ) -> list[OutfitBlueprint]:
        """Return ranked category-level OutfitBlueprints using Method A scoring."""

    def get_color_harmony(self, anchor_color_id: str | None) -> ColorHarmonyAdvice:
        """Return decoupled color harmony advice for an anchor canonical color_id."""

    def get_styles_for_occasion(self, occasion_id: str) -> list[str]:
        """Return ranked Style IDs linked to occasion via FITS_STYLE."""

    def get_occasions_for_category(self, category_id: str) -> list[str]:
        """Return ranked Occasion IDs linked to category via SUITS_CATEGORY."""

    def get_category(self, category_id: str) -> CategoryNode | None:
        """Return CategoryNode metadata for canonical category_id if known."""


def _dedupe_non_empty(items: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


class InMemoryOutfitKGRepository:
    """Pure-Python OutfitKGRepository operating directly on OutfitOntology."""

    def __init__(self, ontology: OutfitOntology | None = None) -> None:
        self.ontology = ontology or get_cached_ontology()
        self._categories: dict[str, CategoryNode] = {
            c.id: c for c in self.ontology.categories
        }
        self._colors_by_id = {c.id: c for c in self.ontology.colors}

    def get_category(self, category_id: str) -> CategoryNode | None:
        return self._categories.get(category_id)

    def get_styles_for_occasion(self, occasion_id: str) -> list[str]:
        matches = [e for e in self.ontology.fits_style if e.source_id == occasion_id]
        matches.sort(key=lambda e: (-int(e.is_default), -e.compatibility, e.target_id))
        return [e.target_id for e in matches]

    def get_occasions_for_category(self, category_id: str) -> list[str]:
        matches = [
            e for e in self.ontology.suits_category if e.target_id == category_id
        ]
        matches.sort(key=lambda e: (-e.compatibility, e.source_id))
        return [e.source_id for e in matches]

    def get_outfit_blueprints(
        self,
        style_id: str | None = None,
        occasion_id: str | None = None,
        anchor_category_id: str | None = None,
        limit: int = 10,
    ) -> list[OutfitBlueprint]:
        if anchor_category_id is not None:
            anchor_cat = self._categories.get(anchor_category_id)
            if anchor_cat is None or not anchor_cat.supports_top_bottom_blueprint:
                return []

        blueprints: list[OutfitBlueprint] = []

        # Branch 1: style_id known (RECOMMENDS_CATEGORY + PAIRS_WITH + opt FITS_STYLE)
        if style_id is not None:
            rec_edges = [
                e for e in self.ontology.recommends_category if e.source_id == style_id
            ]
            top_recs: dict[str, Any] = {}
            bot_recs: dict[str, Any] = {}
            for edge in rec_edges:
                cat = self._categories.get(edge.target_id)
                if not cat or not cat.supports_top_bottom_blueprint:
                    continue
                if cat.role in ("TOP", "OUTERWEAR"):
                    top_recs[cat.id] = edge
                elif cat.role == "BOTTOM":
                    bot_recs[cat.id] = edge

            occ_style_edge = None
            if occasion_id is not None:
                occ_style_edge = next(
                    (
                        e
                        for e in self.ontology.fits_style
                        if e.source_id == occasion_id and e.target_id == style_id
                    ),
                    None,
                )
            occ_style_score = (
                occ_style_edge.compatibility if occ_style_edge is not None else 0.5
            )

            for pair in self.ontology.pairs_with:
                if pair.source_id not in top_recs or pair.target_id not in bot_recs:
                    continue
                if anchor_category_id is not None and anchor_category_id not in (
                    pair.source_id,
                    pair.target_id,
                ):
                    continue
                st_top = top_recs[pair.source_id]
                st_bot = bot_recs[pair.target_id]
                total = round(
                    0.45 * pair.compatibility
                    + 0.25 * st_top.compatibility
                    + 0.25 * st_bot.compatibility
                    + 0.05 * occ_style_score,
                    4,
                )
                blueprints.append(
                    OutfitBlueprint(
                        top_category_id=pair.source_id,
                        bottom_category_id=pair.target_id,
                        score=total,
                        score_breakdown={
                            "pair": pair.compatibility,
                            "style_top": st_top.compatibility,
                            "style_bottom": st_bot.compatibility,
                            "occasion_style": occ_style_score,
                        },
                        applied_rule_ids=_dedupe_non_empty(
                            [
                                pair.rule_id,
                                st_top.rule_id,
                                st_bot.rule_id,
                                occ_style_edge.rule_id if occ_style_edge else None,
                            ]
                        ),
                        reasons=_dedupe_non_empty(
                            [
                                pair.reason,
                                st_top.reason,
                                st_bot.reason,
                                occ_style_edge.reason if occ_style_edge else None,
                            ]
                        ),
                    )
                )

        # Branch 2: style_id unknown, occasion_id known (SUITS_CATEGORY + PAIRS_WITH)
        elif occasion_id is not None:
            occ_edges = [
                e for e in self.ontology.suits_category if e.source_id == occasion_id
            ]
            top_occs: dict[str, Any] = {}
            bot_occs: dict[str, Any] = {}
            for edge in occ_edges:
                cat = self._categories.get(edge.target_id)
                if not cat or not cat.supports_top_bottom_blueprint:
                    continue
                if cat.role in ("TOP", "OUTERWEAR"):
                    top_occs[cat.id] = edge
                elif cat.role == "BOTTOM":
                    bot_occs[cat.id] = edge

            for pair in self.ontology.pairs_with:
                if pair.source_id not in top_occs or pair.target_id not in bot_occs:
                    continue
                if anchor_category_id is not None and anchor_category_id not in (
                    pair.source_id,
                    pair.target_id,
                ):
                    continue
                ot = top_occs[pair.source_id]
                ob = bot_occs[pair.target_id]
                total = round(
                    0.50 * pair.compatibility
                    + 0.25 * ot.compatibility
                    + 0.25 * ob.compatibility,
                    4,
                )
                blueprints.append(
                    OutfitBlueprint(
                        top_category_id=pair.source_id,
                        bottom_category_id=pair.target_id,
                        score=total,
                        score_breakdown={
                            "pair": pair.compatibility,
                            "occasion_top": ot.compatibility,
                            "occasion_bottom": ob.compatibility,
                        },
                        applied_rule_ids=_dedupe_non_empty(
                            [pair.rule_id, ot.rule_id, ob.rule_id]
                        ),
                        reasons=_dedupe_non_empty([pair.reason, ot.reason, ob.reason]),
                    )
                )

        # Branch 3: both style_id and occasion_id unknown (pure PAIRS_WITH)
        else:
            for pair in self.ontology.pairs_with:
                top_cat = self._categories.get(pair.source_id)
                bot_cat = self._categories.get(pair.target_id)
                if (
                    not top_cat
                    or not bot_cat
                    or not top_cat.supports_top_bottom_blueprint
                    or not bot_cat.supports_top_bottom_blueprint
                ):
                    continue
                if anchor_category_id is not None and anchor_category_id not in (
                    pair.source_id,
                    pair.target_id,
                ):
                    continue
                blueprints.append(
                    OutfitBlueprint(
                        top_category_id=pair.source_id,
                        bottom_category_id=pair.target_id,
                        score=round(pair.compatibility, 4),
                        score_breakdown={"pair": pair.compatibility},
                        applied_rule_ids=[pair.rule_id],
                        reasons=[pair.reason],
                    )
                )

        blueprints.sort(
            key=lambda bp: (-bp.score, bp.top_category_id, bp.bottom_category_id)
        )
        return blueprints[:limit]

    def get_color_harmony(self, anchor_color_id: str | None) -> ColorHarmonyAdvice:
        if not anchor_color_id or anchor_color_id not in self._colors_by_id:
            return ColorHarmonyAdvice(
                anchor_color_id=anchor_color_id or "",
                available=False,
            )

        matches: list[tuple[str, Any]] = []
        for edge in self.ontology.harmonizes_with:
            if edge.source_id == anchor_color_id:
                matches.append((edge.target_id, edge))
            elif edge.target_id == anchor_color_id:
                matches.append((edge.source_id, edge))

        matches.sort(key=lambda item: (-item[1].compatibility, item[0]))
        return ColorHarmonyAdvice(
            anchor_color_id=anchor_color_id,
            recommended_color_ids=[cid for cid, _ in matches],
            compatibility_by_color={cid: edge.compatibility for cid, edge in matches},
            applied_rule_ids=_dedupe_non_empty([edge.rule_id for _, edge in matches]),
            reasons=_dedupe_non_empty([edge.reason for _, edge in matches]),
            available=True,
        )


_BRANCH_1_CYPHER = """
MATCH (s:Style {id: $style_id})-[st:RECOMMENDS_CATEGORY]->(top:Category)
WHERE top.role IN ["TOP", "OUTERWEAR"] AND top.supports_top_bottom_blueprint = true
MATCH (s)-[sb:RECOMMENDS_CATEGORY]->(bot:Category)
WHERE bot.role = "BOTTOM" AND bot.supports_top_bottom_blueprint = true
MATCH (top)-[p:PAIRS_WITH]->(bot)
WHERE ($anchor_category_id IS NULL
       OR top.id = $anchor_category_id
       OR bot.id = $anchor_category_id)
OPTIONAL MATCH (o:Occasion {id: $occasion_id})-[os:FITS_STYLE]->(s)
WITH top, bot, p, st, sb, os,
     coalesce(os.compatibility, 0.5) AS occ_style_score
WITH top, bot, p, st, sb, os, occ_style_score,
     round(0.45 * p.compatibility
           + 0.25 * st.compatibility
           + 0.25 * sb.compatibility
           + 0.05 * occ_style_score, 4) AS total_score
ORDER BY total_score DESC, top.id ASC, bot.id ASC
LIMIT $limit
RETURN top.id AS top_category_id,
       bot.id AS bottom_category_id,
       total_score AS score,
       {
         pair: p.compatibility,
         style_top: st.compatibility,
         style_bottom: sb.compatibility,
         occasion_style: occ_style_score
       } AS score_breakdown,
       [x IN [p.rule_id, st.rule_id, sb.rule_id, os.rule_id] WHERE x IS NOT NULL]
         AS applied_rule_ids,
       [x IN [p.reason, st.reason, sb.reason, os.reason] WHERE x IS NOT NULL]
         AS reasons
"""

_BRANCH_2_CYPHER = """
MATCH (o:Occasion {id: $occasion_id})-[ot:SUITS_CATEGORY]->(top:Category)
WHERE top.role IN ["TOP", "OUTERWEAR"] AND top.supports_top_bottom_blueprint = true
MATCH (o)-[ob:SUITS_CATEGORY]->(bot:Category)
WHERE bot.role = "BOTTOM" AND bot.supports_top_bottom_blueprint = true
MATCH (top)-[p:PAIRS_WITH]->(bot)
WHERE ($anchor_category_id IS NULL
       OR top.id = $anchor_category_id
       OR bot.id = $anchor_category_id)
WITH top, bot, p, ot, ob,
     round(0.50 * p.compatibility
           + 0.25 * ot.compatibility
           + 0.25 * ob.compatibility, 4) AS total_score
ORDER BY total_score DESC, top.id ASC, bot.id ASC
LIMIT $limit
RETURN top.id AS top_category_id,
       bot.id AS bottom_category_id,
       total_score AS score,
       {
         pair: p.compatibility,
         occasion_top: ot.compatibility,
         occasion_bottom: ob.compatibility
       } AS score_breakdown,
       [x IN [p.rule_id, ot.rule_id, ob.rule_id] WHERE x IS NOT NULL]
         AS applied_rule_ids,
       [x IN [p.reason, ot.reason, ob.reason] WHERE x IS NOT NULL]
         AS reasons
"""

_BRANCH_3_CYPHER = """
MATCH (top:Category)-[p:PAIRS_WITH]->(bot:Category)
WHERE top.role IN ["TOP", "OUTERWEAR"] AND top.supports_top_bottom_blueprint = true
  AND bot.role = "BOTTOM" AND bot.supports_top_bottom_blueprint = true
  AND ($anchor_category_id IS NULL
       OR top.id = $anchor_category_id
       OR bot.id = $anchor_category_id)
WITH top, bot, p, round(p.compatibility, 4) AS total_score
ORDER BY total_score DESC, top.id ASC, bot.id ASC
LIMIT $limit
RETURN top.id AS top_category_id,
       bot.id AS bottom_category_id,
       total_score AS score,
       {pair: p.compatibility} AS score_breakdown,
       [p.rule_id] AS applied_rule_ids,
       [p.reason] AS reasons
"""

_COLOR_HARMONY_CYPHER = """
MATCH (c1:Color {id: $anchor_color_id})
OPTIONAL MATCH (c1)-[h:HARMONIZES_WITH]-(c2:Color)
WITH c1, c2, h
ORDER BY h.compatibility DESC, c2.id ASC
RETURN c1.id AS anchor_id,
       collect(
         CASE WHEN c2 IS NOT NULL THEN {
           color_id: c2.id,
           compatibility: h.compatibility,
           rule_id: h.rule_id,
           reason: h.reason
         } ELSE NULL END
       ) AS matches
"""


class Neo4jOutfitKGRepository:
    """Neo4j Bolt-backed OutfitKGRepository executing Method A Cypher queries."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def get_category(self, category_id: str) -> CategoryNode | None:
        cypher = (
            "MATCH (c:Category {id: $category_id}) "
            "RETURN c.id AS id, c.name AS name, c.aliases AS aliases, "
            "c.role AS role, c.base_formality AS base_formality, "
            "c.supports_top_bottom_blueprint AS supports_top_bottom_blueprint"
        )
        records, _, _ = self._driver.execute_query(
            cypher, category_id=category_id, database_=self._database
        )
        if not records:
            return None
        row = records[0]
        return CategoryNode(
            id=row["id"],
            name=row["name"],
            aliases=list(row["aliases"] or []),
            role=row["role"],
            base_formality=int(row["base_formality"]),
            supports_top_bottom_blueprint=bool(row["supports_top_bottom_blueprint"]),
        )

    def get_styles_for_occasion(self, occasion_id: str) -> list[str]:
        cypher = (
            "MATCH (:Occasion {id: $occasion_id})-[r:FITS_STYLE]->(s:Style) "
            "RETURN s.id AS style_id "
            "ORDER BY r.is_default DESC, r.compatibility DESC, s.id ASC"
        )
        records, _, _ = self._driver.execute_query(
            cypher, occasion_id=occasion_id, database_=self._database
        )
        return [str(r["style_id"]) for r in records]

    def get_occasions_for_category(self, category_id: str) -> list[str]:
        cypher = (
            "MATCH (o:Occasion)-[r:SUITS_CATEGORY]->(:Category {id: $category_id}) "
            "RETURN o.id AS occasion_id "
            "ORDER BY r.compatibility DESC, o.id ASC"
        )
        records, _, _ = self._driver.execute_query(
            cypher, category_id=category_id, database_=self._database
        )
        return [str(r["occasion_id"]) for r in records]

    def get_outfit_blueprints(
        self,
        style_id: str | None = None,
        occasion_id: str | None = None,
        anchor_category_id: str | None = None,
        limit: int = 10,
    ) -> list[OutfitBlueprint]:
        if anchor_category_id is not None:
            anchor_cat = self.get_category(anchor_category_id)
            if anchor_cat is None or not anchor_cat.supports_top_bottom_blueprint:
                return []

        if style_id is not None:
            cypher = _BRANCH_1_CYPHER
        elif occasion_id is not None:
            cypher = _BRANCH_2_CYPHER
        else:
            cypher = _BRANCH_3_CYPHER

        records, _, _ = self._driver.execute_query(
            cypher,
            style_id=style_id,
            occasion_id=occasion_id,
            anchor_category_id=anchor_category_id,
            limit=int(limit),
            database_=self._database,
        )
        out: list[OutfitBlueprint] = []
        for row in records:
            out.append(
                OutfitBlueprint(
                    top_category_id=row["top_category_id"],
                    bottom_category_id=row["bottom_category_id"],
                    score=float(row["score"]),
                    score_breakdown={
                        k: float(v) for k, v in (row["score_breakdown"] or {}).items()
                    },
                    applied_rule_ids=_dedupe_non_empty(
                        list(row["applied_rule_ids"] or [])
                    ),
                    reasons=_dedupe_non_empty(list(row["reasons"] or [])),
                )
            )
        return out

    def get_color_harmony(self, anchor_color_id: str | None) -> ColorHarmonyAdvice:
        if not anchor_color_id:
            return ColorHarmonyAdvice(anchor_color_id="", available=False)

        records, _, _ = self._driver.execute_query(
            _COLOR_HARMONY_CYPHER,
            anchor_color_id=anchor_color_id,
            database_=self._database,
        )
        if not records:
            return ColorHarmonyAdvice(anchor_color_id=anchor_color_id, available=False)

        matches = [m for m in (records[0]["matches"] or []) if m is not None]
        return ColorHarmonyAdvice(
            anchor_color_id=anchor_color_id,
            recommended_color_ids=[str(m["color_id"]) for m in matches],
            compatibility_by_color={
                str(m["color_id"]): float(m["compatibility"]) for m in matches
            },
            applied_rule_ids=_dedupe_non_empty([str(m["rule_id"]) for m in matches]),
            reasons=_dedupe_non_empty([str(m["reason"]) for m in matches]),
            available=True,
        )


def get_outfit_kg_repository(
    settings: Settings | None = None,
) -> OutfitKGRepository:
    """Construct configured OutfitKGRepository (inmemory default, or neo4j)."""
    cfg = settings or get_settings()
    ont_dir = (cfg.DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))

    if cfg.OUTFIT_KG_BACKEND == "neo4j":
        return Neo4jOutfitKGRepository(
            uri=cfg.NEO4J_URI,
            user=cfg.NEO4J_USER,
            password=cfg.NEO4J_PASSWORD,
            database=cfg.NEO4J_DATABASE,
        )

    return InMemoryOutfitKGRepository(ontology=ontology)
