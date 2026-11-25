"""Unit tests for Tier 1 E1 benchmark, ablations, Tier 2 E2E, and unified report."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.eval.e1_benchmark import (
    aggregate_e1_results,
    calibrate_thresholds_on_dev,
    export_tier2_judge_sheet,
    load_e1_scenarios,
    load_frozen_thresholds,
    run_color_rerank_ablation,
    run_e1_policy_benchmark,
    write_unified_experiment_report,
)
from customer_shop_assistant.eval.e2e_benchmark import (
    aggregate_e2e_results,
    load_contrastive_scenarios,
    run_e2e_contrastive_benchmark,
    run_e2e_stratified_benchmark,
)
from customer_shop_assistant.loaders.outfit_ontology import get_cached_ontology
from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.outfit_kg_repository import (
    InMemoryOutfitKGRepository,
)
from customer_shop_assistant.services.slot_tracker import (
    extract_conversation_slots_with_lifecycle,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)


class _CyclingTestFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _infinite_test_messages() -> Any:
    while True:
        yield AIMessage(
            content=(
                "Dạ shop đã kiểm tra dữ liệu cửa hàng và gợi ý phương án "
                "phù hợp nhất cho bạn."
            )
        )


def test_e1_scenarios_dataset_integrity_and_intent_alignment() -> None:
    all_scenarios = load_e1_scenarios()
    assert len(all_scenarios) == 84
    assert len({s.id for s in all_scenarios}) == 84

    dev = load_e1_scenarios(split="dev")
    test = load_e1_scenarios(split="test")
    assert len(dev) == 28
    assert len(test) == 56

    for sc in all_scenarios:
        assert sc.gold_rationale.strip()
        _, intent, _ = extract_conversation_slots_with_lifecycle(
            [sc.question], user_profile=sc.user_profile
        )
        assert intent.value == sc.expected_intent, (
            f"{sc.id}: got {intent.value}, expected {sc.expected_intent}"
        )


def test_contrastive_s1_s10_schema_integrity() -> None:
    scenarios = load_contrastive_scenarios()
    assert len(scenarios) == 12
    ids = [s.id for s in scenarios]
    assert ids == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6A",
        "S6B",
        "S7",
        "S8",
        "S9",
        "S10A",
        "S10B",
    ]
    for sc in scenarios:
        assert set(sc.expected_by_mode.keys()) == {
            "IMMEDIATE_ANSWER",
            "ALWAYS_CLARIFY",
            "CONDITIONAL_CLARIFICATION",
        }


def test_dev_calibration_slot_metrics_and_ablations(tmp_path: Path) -> None:
    dev = load_e1_scenarios(split="dev")
    test = load_e1_scenarios(split="test")

    thresh_path = tmp_path / "frozen_thresholds.json"
    best_thresh, sweep = calibrate_thresholds_on_dev(
        dev, backend="inmemory", output_path=thresh_path
    )
    assert thresh_path.exists()
    assert len(sweep) == 10
    loaded_thresh, loaded_sweep, color_sweep = load_frozen_thresholds(thresh_path)
    assert loaded_thresh == best_thresh
    assert len(loaded_sweep) == 10
    assert len(color_sweep) == 9
    assert 0.2 <= best_thresh.jaccard_threshold <= 0.7

    jsonl_path = tmp_path / "e1_test_results.jsonl"
    aggregates, results = run_e1_policy_benchmark(
        test,
        backend="inmemory",
        thresholds=best_thresh,
        include_ablation=True,
        output_jsonl_path=jsonl_path,
    )
    assert jsonl_path.exists()
    assert len(aggregates) == 5

    mode_a = aggregates["IMMEDIATE_ANSWER"]
    mode_b = aggregates["ALWAYS_CLARIFY"]
    mode_c = aggregates["CONDITIONAL_CLARIFICATION (KG+Catalog)"]

    assert mode_a.clarification_rate == 0.0
    assert mode_a.policy_compliance_rate == 100.0
    assert mode_a.avg_turns_to_completion == 1.0

    assert mode_b.policy_compliance_rate == 100.0
    assert mode_c.policy_compliance_rate == 100.0
    assert mode_c.avg_probe_latency_ms < 30.0
    assert mode_c.outfit_rule_violation_rate == 0.0
    assert mode_c.mean_color_score >= 0.70
    assert mode_c.color_harmony_rate > 0.0
    assert mode_c.avg_slot_recall is not None and mode_c.avg_slot_recall >= 90.0

    # Color v1.2 vs v1.3 ablation
    cfg = get_settings()
    ont_dir = (cfg.DATA_DIR / "outfit_kg" / "ontology").resolve()
    ontology = get_cached_ontology(str(ont_dir))
    snapshot_index = get_snapshot_catalog_index(cfg)
    kg_repo = InMemoryOutfitKGRepository(ontology=ontology)
    color_ablation = run_color_rerank_ablation(
        test, kg_repo=kg_repo, snapshot_index=snapshot_index, ontology=ontology
    )
    assert color_ablation.v13.mean_color_score >= color_ablation.v12.mean_color_score

    # NON_SHOPPING strictly 0% clarification in all 3 modes
    non_shop = [
        r
        for r in results
        if r.variant == "kg_plus_catalog" and r.expected_intent == "NON_SHOPPING"
    ]
    assert non_shop and all(not r.clarification_asked for r in non_shop)

    # Mode C reduces unnecessary questions vs Mode B on anchor tasks
    b_anchor = aggregate_e1_results(
        "b_anchor",
        [
            r
            for r in results
            if r.mode == ClarificationMode.ALWAYS_CLARIFY.value
            and r.task_type == "anchor"
        ],
    )
    c_anchor = aggregate_e1_results(
        "c_anchor",
        [
            r
            for r in results
            if r.mode == ClarificationMode.CONDITIONAL_CLARIFICATION.value
            and r.variant == "kg_plus_catalog"
            and r.task_type == "anchor"
        ],
    )
    assert c_anchor.unnecessary_question_rate < b_anchor.unnecessary_question_rate

    # Mode C reduces missed necessary clarifications vs Mode A on open tasks
    a_open = aggregate_e1_results(
        "a_open",
        [
            r
            for r in results
            if r.mode == ClarificationMode.IMMEDIATE_ANSWER.value
            and r.task_type == "open"
        ],
    )
    c_open = aggregate_e1_results(
        "c_open",
        [
            r
            for r in results
            if r.mode == ClarificationMode.CONDITIONAL_CLARIFICATION.value
            and r.variant == "kg_plus_catalog"
            and r.task_type == "open"
        ],
    )
    assert c_open.missed_necessary_rate < a_open.missed_necessary_rate

    # Fallback Tier-1 CSV export
    csv_path = tmp_path / "judge_t1.csv"
    exported = export_tier2_judge_sheet(results, test, output_path=csv_path)
    assert exported == 30
    assert "tier1_offline" in csv_path.read_text(encoding="utf-8-sig")


@pytest.mark.anyio
async def test_e2e_mock_multiturn_and_unified_report(tmp_path: Path) -> None:
    cfg = get_settings()
    fake_agent = ShopAssistantAgent(
        chat_model=_CyclingTestFakeChatModel(messages=_infinite_test_messages()),
        settings=cfg,
    )
    contrastive = load_contrastive_scenarios()[:2]
    con_res = await run_e2e_contrastive_benchmark(
        contrastive, backend="inmemory", settings=cfg, agent=fake_agent
    )
    assert len(con_res) == 6
    assert any(r.turns_to_completion == 2 for r in con_res)

    test_scenarios = load_e1_scenarios(split="test")
    strat_res = await run_e2e_stratified_benchmark(
        test_scenarios,
        limit=2,
        backend="inmemory",
        settings=cfg,
        agent=fake_agent,
        output_jsonl_path=tmp_path / "e2e.jsonl",
    )
    assert len(strat_res) == 6
    assert (tmp_path / "e2e.jsonl").exists()

    e2e_agg = aggregate_e2e_results([*con_res, *strat_res])
    assert set(e2e_agg.keys()) == {
        "IMMEDIATE_ANSWER",
        "ALWAYS_CLARIFY",
        "CONDITIONAL_CLARIFICATION",
    }

    # Tier-2 CSV from E2E results
    csv_e2e = tmp_path / "judge_e2e.csv"
    cnt = export_tier2_judge_sheet(
        [], test_scenarios, output_path=csv_e2e, e2e_results=con_res
    )
    assert cnt == 6
    assert "tier2_e2e_llm" in csv_e2e.read_text(encoding="utf-8-sig")

    # Unified report generation
    thresh = load_frozen_thresholds(Path("eval/frozen_thresholds.json"))[0]
    aggregates, results = run_e1_policy_benchmark(
        test_scenarios[:6],
        backend="inmemory",
        thresholds=thresh,
        include_ablation=False,
    )
    report_path = tmp_path / "experiment_report.md"
    write_unified_experiment_report(
        aggregates,
        results,
        thresholds=thresh,
        e2e_contrastive=con_res,
        e2e_stratified=strat_res,
        settings=cfg,
        backend="inmemory",
        dev_count=28,
        test_count=len(test_scenarios),
        output_path=report_path,
    )
    agg = aggregate_e2e_results([*con_res, *strat_res])
    assert len(agg) == 3

    err_probe = strat_res[0].model_copy(
        update={
            "scenario_id": "ERR_1",
            "mode": "IMMEDIATE_ANSWER",
            "turns_to_completion": 0,
            "error": "TimeoutError",
            "passed": False,
        }
    )
    agg_with_err = aggregate_e2e_results([*strat_res, err_probe])
    imm_agg = agg_with_err["IMMEDIATE_ANSWER"]
    assert imm_agg.error_count == 1
    assert imm_agg.avg_turns >= 1.0

    content = report_path.read_text(encoding="utf-8")
    assert "Tầng 1 — So sánh 3 Chiến lược" in content
    assert "Tầng 2 — Kiểm thử End-to-End Đa lượt" in content
    assert "4.2 Tổng hợp E2E — Bộ Đối chứng S1–S10" in content
    assert "4.3 Tổng hợp E2E — Bộ Stratified Frozen Test" in content
    assert "MINIMAL_CLARIFICATION" not in content
