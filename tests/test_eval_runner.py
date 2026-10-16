from pathlib import Path

import pytest

from customer_shop_assistant.eval.dataset import load_golden_set
from customer_shop_assistant.eval.runner import EvalRunner


def test_load_golden_set() -> None:
    cases = load_golden_set(filepath="eval/questions.json")
    assert len(cases) >= 50
    assert any(c.category == "product_suitability" for c in cases)
    assert any(c.category == "size_fit" for c in cases)
    assert any(c.category == "clarification_strategy" for c in cases)


def test_load_golden_set_with_filter() -> None:
    cases = load_golden_set(
        filepath="eval/questions.json", category="size_fit", limit=5
    )
    assert len(cases) <= 5
    assert all(c.category == "size_fit" for c in cases)


@pytest.mark.anyio
async def test_eval_runner_dry_run() -> None:
    cases = load_golden_set(filepath="eval/questions.json", limit=3)
    runner = EvalRunner(dry_run=True)

    results = await runner.run_suite(
        cases, output_file=Path("logs/test_eval_run.jsonl")
    )
    assert len(results) == 3
    for r in results:
        assert r.latency_ms >= 0
        assert r.estimated_tokens > 0
