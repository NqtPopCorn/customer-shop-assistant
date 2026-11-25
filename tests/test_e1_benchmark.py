"""Unit tests for Experiment E1 dataset, Dev calibration, and InMemory benchmark."""

from pathlib import Path

from customer_shop_assistant.eval.e1_benchmark import (
    aggregate_e1_results,
    calibrate_thresholds_on_dev,
    export_tier2_judge_sheet,
    load_e1_scenarios,
    run_e1_policy_benchmark,
    write_e1_markdown_report,
)
from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.services.slot_tracker import (
    extract_conversation_slots_with_lifecycle,
)


def test_e1_scenarios_dataset_integrity_and_intent_alignment() -> None:
    all_scenarios = load_e1_scenarios()
    assert len(all_scenarios) == 72
    assert len({s.id for s in all_scenarios}) == 72

    dev = load_e1_scenarios(split="dev")
    test = load_e1_scenarios(split="test")
    assert len(dev) == 24
    assert len(test) == 48

    for sc in all_scenarios:
        assert sc.gold_rationale.strip()
        _, intent, _ = extract_conversation_slots_with_lifecycle(
            [sc.question], user_profile=sc.user_profile
        )
        assert intent.value == sc.expected_intent, (
            f"{sc.id}: got {intent.value}, expected {sc.expected_intent}"
        )


def test_dev_calibration_and_inmemory_e1_benchmark(tmp_path: Path) -> None:
    dev = load_e1_scenarios(split="dev")
    test = load_e1_scenarios(split="test")

    thresh_path = tmp_path / "frozen_thresholds.json"
    best_thresh, sweep = calibrate_thresholds_on_dev(
        dev, backend="inmemory", output_path=thresh_path
    )
    assert thresh_path.exists()
    assert len(sweep) == 10
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

    # Export report and Tier-2 sheet cleanly
    report_path = tmp_path / "report.md"
    write_e1_markdown_report(
        aggregates,
        results,
        thresholds=best_thresh,
        dev_sweep=sweep,
        backend="inmemory",
        dev_count=len(dev),
        test_count=len(test),
        output_path=report_path,
    )
    assert report_path.exists() and "IMMEDIATE_ANSWER" in report_path.read_text(
        encoding="utf-8"
    )

    csv_path = tmp_path / "judge.csv"
    exported = export_tier2_judge_sheet(results, test, output_path=csv_path)
    assert exported == 30
    assert csv_path.exists()
