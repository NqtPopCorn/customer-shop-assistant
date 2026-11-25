from pathlib import Path

import pytest

from customer_shop_assistant.eval.dataset import load_golden_set
from customer_shop_assistant.eval.runner import EvalRunner
from customer_shop_assistant.schemas.chat import ClarificationMode

CANONICAL_INTENTS = {
    "PRODUCT_SEARCH",
    "OUTFIT_ADVICE",
    "SIZE_ADVICE",
    "BODY_FIT_ADVICE",
    "NON_SHOPPING",
}


def test_load_golden_set_v2_integrity() -> None:
    cases = load_golden_set(filepath="eval/questions.json")
    assert len(cases) >= 65
    assert len({c.question for c in cases}) == len(cases)
    assert all(
        c.clarification_mode != ClarificationMode.MINIMAL_CLARIFICATION for c in cases
    )
    assert all(
        c.ground_truth.expected_intent in CANONICAL_INTENTS
        for c in cases
        if c.ground_truth.expected_intent is not None
    )
    assert any(c.category == "product_suitability" for c in cases)
    assert any(c.category == "size_fit" for c in cases)
    assert any(c.category == "clarification_strategy" for c in cases)


def test_load_golden_set_with_filter() -> None:
    cases = load_golden_set(
        filepath="eval/questions.json", category="size_fit", limit=5
    )
    assert len(cases) == 5
    assert all(c.category == "size_fit" for c in cases)


@pytest.mark.anyio
async def test_eval_runner_dry_run(tmp_path: Path) -> None:
    cases = load_golden_set(filepath="eval/questions.json", limit=3)
    runner = EvalRunner(dry_run=True)

    out_path = tmp_path / "test_eval_run.jsonl"
    results = await runner.run_suite(cases, output_file=out_path)
    assert out_path.exists()
    assert len(results) == 3
    for r in results:
        assert r.latency_ms >= 0
        assert r.total_tokens > 0
        assert r.provider == "mock"
        assert r.model_name == "dry-run-fake"
