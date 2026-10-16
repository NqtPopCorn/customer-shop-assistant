"""Evaluation package for benchmarking and researching the customer shop assistant."""

from customer_shop_assistant.eval.dataset import (
    EvaluationCase,
    GroundTruth,
    load_golden_set,
)
from customer_shop_assistant.eval.metrics import (
    EvaluationMetricResult,
    evaluate_response,
)
from customer_shop_assistant.eval.runner import EvalRunner

__all__ = [
    "EvaluationCase",
    "GroundTruth",
    "load_golden_set",
    "EvaluationMetricResult",
    "evaluate_response",
    "EvalRunner",
]
