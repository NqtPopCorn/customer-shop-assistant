"""Unified Research Experiments Runner (Tier 1 Policy/KG + Tier 2 Multi-turn E2E)."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.eval.dataset import load_golden_set
from customer_shop_assistant.eval.e1_benchmark import (
    _close_kg_repo,
    _open_kg_repo,
    calibrate_thresholds_on_dev,
    export_tier2_judge_sheet,
    load_e1_scenarios,
    load_frozen_thresholds,
    run_color_rerank_ablation,
    run_e1_policy_benchmark,
    write_unified_experiment_report,
)
from customer_shop_assistant.eval.e2e_benchmark import (
    E2EScenarioRunResult,
    ServiceHealthStatus,
    aggregate_e2e_results,
    check_eval_services,
    load_contrastive_scenarios,
    run_e2e_contrastive_benchmark,
    run_e2e_stratified_benchmark,
)
from customer_shop_assistant.eval.runner import EvalRunner
from customer_shop_assistant.loaders.outfit_ontology import get_cached_ontology
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class _CyclingFakeChatModel(GenericFakeChatModel):
    """Endless mock chat model for zero-GPU dry-runs of E2E pipelines."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _make_dry_run_agent(settings: Any) -> Any:
    from customer_shop_assistant.services.agent_service import ShopAssistantAgent

    def _infinite_replies() -> Any:
        while True:
            yield AIMessage(
                content=(
                    "Dạ shop đã ghi nhận yêu cầu của bạn và đề xuất các lựa chọn "
                    "phù hợp từ dữ liệu cửa hàng."
                )
            )

    return ShopAssistantAgent(
        chat_model=_CyclingFakeChatModel(messages=_infinite_replies()),
        settings=settings,
    )


def _print_service_health(health: ServiceHealthStatus) -> None:
    be_col = GREEN if health.main_be_ok else YELLOW
    be_state = "ONLINE" if health.main_be_ok else "OFFLINE"
    print(
        f"{DIM}  • Main BE  : {RESET}{be_col}{be_state}{RESET} "
        f"{DIM}({health.main_be_url} — {health.main_be_detail}){RESET}"
    )
    if health.neo4j_ok is not None:
        n_col = GREEN if health.neo4j_ok else RED
        n_state = "ONLINE" if health.neo4j_ok else "OFFLINE"
        print(
            f"{DIM}  • Neo4j KG : {RESET}{n_col}{n_state}{RESET} "
            f"{DIM}({health.neo4j_detail}){RESET}"
        )


async def run_unified_e1(
    *,
    backend: Literal["inmemory", "neo4j"] = "inmemory",
    calibrate: bool = False,
    ablation: bool = True,
    tier1_only: bool = False,
    run_contrastive: bool = True,
    e2e_limit: int = 12,
    e2e_all: bool = False,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> Path:
    """Execute unified Tier 1 + Tier 2 E1 research benchmark and write report."""
    settings = get_settings()
    dev_scenarios = load_e1_scenarios(split="dev")
    test_scenarios = load_e1_scenarios(split="test")

    print(f"\n{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(
        f"{CYAN}{BOLD}   UNIFIED RESEARCH EVALUATION — "
        f"EXPERIMENT E1 (TIER 1 + TIER 2){RESET}"
    )
    print(f"{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(
        f"{DIM}  • LLM      : {RESET}{YELLOW}{settings.LLM_PROVIDER}{RESET} "
        f"{DIM}| KG Backend: {RESET}{YELLOW}{backend}{RESET}"
    )

    health: ServiceHealthStatus | None = None
    if not dry_run:
        health = await check_eval_services(settings, backend=backend)
        _print_service_health(health)
    print()

    frozen_path = Path("eval/frozen_thresholds.json")
    if calibrate:
        thresholds, dev_sweep = calibrate_thresholds_on_dev(
            dev_scenarios,
            backend=backend,
            settings=settings,
            output_path=frozen_path,
        )
        _, _, color_sweep = load_frozen_thresholds(frozen_path)
    else:
        thresholds, dev_sweep, color_sweep = load_frozen_thresholds(frozen_path)

    aggregates, results = run_e1_policy_benchmark(
        test_scenarios,
        backend=backend,
        thresholds=thresholds,
        include_ablation=ablation,
        settings=settings,
        output_jsonl_path=Path("logs/e1_test_results.jsonl"),
    )

    color_ablation = None
    if ablation:
        ont_dir = (settings.DATA_DIR / "outfit_kg" / "ontology").resolve()
        ontology = get_cached_ontology(str(ont_dir))
        snapshot_index = get_snapshot_catalog_index(settings)
        kg_repo = _open_kg_repo(backend, settings)
        try:
            color_ablation = run_color_rerank_ablation(
                test_scenarios,
                kg_repo=kg_repo,
                snapshot_index=snapshot_index,
                ontology=ontology,
            )
        finally:
            _close_kg_repo(kg_repo)

    print(
        f"{BOLD}[Tier 1] Frozen Test Results ({len(test_scenarios)} scenarios):{RESET}"
    )
    for lbl, m in aggregates.items():
        slot_r = f"{m.avg_slot_recall:.1f}%" if m.avg_slot_recall is not None else "N/A"
        print(
            f"  {lbl:<42} | Clarify={m.clarification_rate:5.1f}% | "
            f"Policy={m.policy_compliance_rate:5.1f}% | "
            f"Gold={m.gold_alignment_rate:5.1f}% | "
            f"SlotR={slot_r:>5} | Probe={m.avg_probe_latency_ms:.2f}ms"
        )
    if color_ablation is not None:
        print(
            f"  {'Color Rerank (v1.2 -> v1.3)':<42} | "
            f"ColorScore={color_ablation.v12.mean_color_score:.3f}->"
            f"{color_ablation.v13.mean_color_score:.3f} "
            f"({color_ablation.delta_color_score:+.3f}) | "
            f"Harmony={color_ablation.v12.color_harmony_rate:.1f}%->"
            f"{color_ablation.v13.color_harmony_rate:.1f}%"
        )

    e2e_contrastive: list[E2EScenarioRunResult] = []
    e2e_stratified: list[E2EScenarioRunResult] = []

    if not tier1_only:
        mock_agent = _make_dry_run_agent(settings) if dry_run else None
        if run_contrastive:
            contrastive_cases = load_contrastive_scenarios()
            print(
                f"\n{YELLOW}{BOLD}[Tier 2A] Running Contrastive S1-S10 "
                f"({len(contrastive_cases)} scenarios x 3 modes)...{RESET}"
            )
            e2e_contrastive = await run_e2e_contrastive_benchmark(
                contrastive_cases,
                backend=backend,
                settings=settings,
                agent=mock_agent,
            )
        if e2e_limit > 0 or e2e_all:
            n_pick = len(test_scenarios) if e2e_all else e2e_limit
            print(
                f"{YELLOW}{BOLD}[Tier 2B] Running Stratified E2E "
                f"({n_pick} Test scenarios x 3 modes)...{RESET}"
            )
            e2e_stratified = await run_e2e_stratified_benchmark(
                test_scenarios,
                limit=e2e_limit,
                run_all=e2e_all,
                backend=backend,
                settings=settings,
                agent=mock_agent,
            )
        all_e2e = [*e2e_contrastive, *e2e_stratified]
        if all_e2e:
            e2e_log = Path("logs/e1_e2e_results.jsonl")
            e2e_log.parent.mkdir(parents=True, exist_ok=True)
            with e2e_log.open("w", encoding="utf-8") as f:
                for item in all_e2e:
                    f.write(item.model_dump_json() + "\n")
            print(f"\n{BOLD}[Tier 2] E2E Summary by Mode:{RESET}")
            for m_name, em in aggregate_e2e_results(all_e2e).items():
                print(
                    f"  {m_name:<28} | Pass={em.pass_rate:5.1f}% | "
                    f"ModeComp={em.mode_compliance_rate:5.1f}% | "
                    f"Ground={em.grounding_rate:5.1f}% | "
                    f"Tok={em.avg_total_tokens:.0f} | Lat={em.avg_latency_ms:.0f}ms"
                )

    report_path = output_path or Path("docs/research/experiment_report.md")
    write_unified_experiment_report(
        aggregates,
        results,
        thresholds=thresholds,
        dev_sweep=dev_sweep or None,
        color_sweep=color_sweep or None,
        color_ablation=color_ablation,
        e2e_contrastive=e2e_contrastive or None,
        e2e_stratified=e2e_stratified or None,
        service_status=health,
        settings=settings,
        backend=backend,
        dev_count=len(dev_scenarios),
        test_count=len(test_scenarios),
        output_path=report_path,
    )

    all_e2e_for_csv = [*e2e_contrastive, *e2e_stratified]
    export_tier2_judge_sheet(
        results,
        test_scenarios,
        e2e_results=all_e2e_for_csv or None,
    )

    print(f"\n{GREEN}{BOLD}✓ Báo cáo nghiên cứu thống nhất: {report_path}{RESET}")
    print(
        f"{GREEN}{BOLD}✓ Bảng chấm Tier-2 CSV      : "
        f"eval/e1_tier2_judge_sheet.csv{RESET}\n"
    )
    return report_path


async def run_experiment_outfit_kg(
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare baseline (OUTFIT_KG_ENABLED=False) vs KG tool on mix_and_match cases."""
    all_cases = load_golden_set()
    cases = [c for c in all_cases if c.sub_category == "mix_and_match"]
    if limit:
        cases = cases[:limit]

    summary: dict[str, dict[str, float]] = {}
    for enabled, label in ((False, "baseline_no_kg"), (True, "kg_tool")):
        base_settings = get_settings()
        cfg = base_settings.model_copy(update={"OUTFIT_KG_ENABLED": enabled})
        runner = EvalRunner(settings=cfg, dry_run=dry_run)
        print(f"\n{YELLOW}{BOLD}>>> Running Outfit Eval: {label} ...{RESET}")
        results = await runner.run_suite(cases)
        tot = len(results)
        ret_cnt = sum(1 for r in results if r.outfit_returned)
        viol_vals = [
            r.outfit_rule_violation_rate
            for r in results
            if r.outfit_rule_violation_rate is not None
        ]
        summary[label] = {
            "total": float(tot),
            "pass_rate": (sum(1 for r in results if r.passed) / tot * 100.0)
            if tot
            else 0.0,
            "outfit_returned_rate": (ret_cnt / tot * 100.0) if tot else 0.0,
            "rule_violation_rate": (sum(viol_vals) / len(viol_vals) * 100.0)
            if viol_vals
            else 0.0,
        }
    return summary


async def async_main() -> None:
    """Async entrypoint for unified research experiments CLI."""
    parser = argparse.ArgumentParser(
        description="Run research experiments for Customer Shop Assistant"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="e1",
        choices=["e1", "e1-kg", "outfit"],
        help=(
            "Experiment code (e1 runs unified Tier 1 + Tier 2; "
            "outfit compares KG on/off)"
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="inmemory",
        choices=["inmemory", "neo4j"],
        help="Outfit KG repository backend",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Tune and freeze counterfactual thresholds on Dev split",
    )
    parser.add_argument(
        "--ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Catalog-only, KG-only, and Color v1.2 vs v1.3 ablations",
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Run fast offline Tier 1 Policy/KG benchmark only (skip LLM E2E)",
    )
    parser.add_argument(
        "--e2e-contrastive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run S1-S10 multi-turn contrastive E2E suite in Tier 2",
    )
    parser.add_argument(
        "--e2e-limit",
        type=int,
        default=12,
        help="Number of stratified Frozen Test scenarios for Tier 2 E2E (0 to skip)",
    )
    parser.add_argument(
        "--e2e-all",
        action="store_true",
        help="Run all 56 Frozen Test scenarios in Tier 2 E2E",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case cap for outfit experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with cycling mock LLM (no GPU calls)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output markdown report path (default: docs/research/experiment_report.md)"
        ),
    )

    args = parser.parse_args()
    out_p = Path(args.output) if args.output else None

    if args.experiment in ("e1", "e1-kg"):
        await run_unified_e1(
            backend=args.backend,
            calibrate=args.calibrate,
            ablation=args.ablation,
            tier1_only=args.tier1_only,
            run_contrastive=args.e2e_contrastive,
            e2e_limit=args.e2e_limit,
            e2e_all=args.e2e_all,
            dry_run=args.dry_run,
            output_path=out_p,
        )
    elif args.experiment == "outfit":
        await run_experiment_outfit_kg(limit=args.limit, dry_run=args.dry_run)


def main() -> None:
    """CLI script entrypoint."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
