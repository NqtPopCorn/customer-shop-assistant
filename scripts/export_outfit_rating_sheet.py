"""Export ~20 outfit suggestions (10 Mode A + 10 Mode B) for manual aesthetic rating."""

import asyncio
import csv
import random
from pathlib import Path

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import OutfitSuggestion
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.outfit_graph import get_outfit_graph
from customer_shop_assistant.services.outfit_rules import normalize_occasion


async def export_rating_sheet(output_path: Path) -> int:
    """Query OutfitGraph for Mode A and Mode B sets and write CSV with UTF-8 BOM."""
    graph = await get_outfit_graph()
    if not graph.nodes:
        raise RuntimeError(
            "OutfitGraph is empty. Ensure fashion-shop-be is running on "
            f"{get_settings().MAIN_BE_BASE_URL}."
        )

    rng = random.Random(42)
    all_ids = sorted(graph.nodes.keys())
    anchors = rng.sample(all_ids, min(10, len(all_ids)))

    rows: list[OutfitSuggestion] = []
    for pid in anchors:
        res = graph.suggest_for_anchor(anchor_id=pid, limit=1)
        if res.outfits:
            rows.append(res.outfits[0])

    mode_b_specs = [
        ("đi làm", GenderTarget.MALE),
        ("đi làm", GenderTarget.FEMALE),
        ("đi chơi", GenderTarget.MALE),
        ("đi chơi", GenderTarget.FEMALE),
        ("hẹn hò", GenderTarget.MALE),
        ("hẹn hò", GenderTarget.FEMALE),
        ("đi biển", GenderTarget.MALE),
        ("đi biển", GenderTarget.FEMALE),
        ("cà phê", GenderTarget.MALE),
        ("cà phê", GenderTarget.FEMALE),
    ]
    for occ_text, gender in mode_b_specs:
        occ_range = normalize_occasion(occ_text, graph.rules)
        res = graph.suggest_open(occasion_range=occ_range, gender=gender, limit=1)
        if res.outfits:
            rows.append(res.outfits[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "top",
                "top_color",
                "bottom",
                "bottom_color",
                "score",
                "reasons",
                "rating_1_5",
                "comment",
            ]
        )
        for o in rows:
            writer.writerow(
                [
                    f"[{o.top.product_id}] {o.top.name}",
                    o.top.color,
                    f"[{o.bottom.product_id}] {o.bottom.name}",
                    o.bottom.color,
                    f"{o.score:.4f}",
                    " | ".join(o.reasons),
                    "",
                    "",
                ]
            )
    return len(rows)


def main() -> None:
    """CLI entrypoint for exporting manual rating CSV."""
    out = Path("eval/outfit_manual_rating.csv")
    count = asyncio.run(export_rating_sheet(out))
    print(f"Exported {count} outfits to {out}")


if __name__ == "__main__":
    main()
