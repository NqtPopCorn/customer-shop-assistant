"""Evaluation package for benchmarking and researching the customer shop assistant."""

from customer_shop_assistant.eval.dataset import (
    E1Scenario,
    EvaluationCase,
    GoldClarificationLabel,
    GroundTruth,
    load_golden_set,
)
from customer_shop_assistant.eval.e2e_benchmark import (
    ContrastiveScenario,
    E2EAggregateMetrics,
    E2EScenarioRunResult,
    ServiceHealthStatus,
    aggregate_e2e_results,
    check_eval_services,
    load_contrastive_scenarios,
    run_e2e_contrastive_benchmark,
    run_e2e_stratified_benchmark,
)
from customer_shop_assistant.eval.metrics import (
    EvaluationMetricResult,
    SlotAccuracyResult,
    StructuralCheckResult,
    TokenUsageSummary,
    compute_slot_accuracy,
    evaluate_response,
    evaluate_turn_structure_and_grounding,
    extract_token_usage,
)
from customer_shop_assistant.eval.runner import EvalRunner

__all__ = [
    "ContrastiveScenario",
    "E1Scenario",
    "E2EAggregateMetrics",
    "E2EScenarioRunResult",
    "EvalRunner",
    "EvaluationCase",
    "EvaluationMetricResult",
    "GoldClarificationLabel",
    "GroundTruth",
    "ServiceHealthStatus",
    "SlotAccuracyResult",
    "StructuralCheckResult",
    "TokenUsageSummary",
    "aggregate_e2e_results",
    "check_eval_services",
    "compute_slot_accuracy",
    "evaluate_response",
    "evaluate_turn_structure_and_grounding",
    "extract_token_usage",
    "load_contrastive_scenarios",
    "load_golden_set",
    "run_e2e_contrastive_benchmark",
    "run_e2e_stratified_benchmark",
]
