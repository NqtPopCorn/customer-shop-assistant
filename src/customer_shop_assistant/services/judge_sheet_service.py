"""Service for reading, updating, and summarizing the Tier-2 judge CSV."""

import csv
import threading
from datetime import UTC, datetime
from pathlib import Path

from customer_shop_assistant.schemas.judge import (
    JudgeModeSummary,
    JudgeRow,
    JudgeRowStatus,
    JudgeScoreUpdate,
    JudgeSummaryResponse,
)

_CSV_LOCK = threading.Lock()

CANONICAL_MODES: tuple[str, ...] = (
    "IMMEDIATE_ANSWER",
    "ALWAYS_CLARIFY",
    "CONDITIONAL_CLARIFICATION",
)


class RowConflictError(Exception):
    """Raised when an optimistic-lock expected_updated_at check fails."""

    def __init__(self, server_row: JudgeRow) -> None:
        super().__init__("ROW_CONFLICT")
        self.server_row = server_row


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    return int(float(cleaned))


def _parse_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() == "true"


def _compute_row_status(
    relevance: int | None,
    unnecessary: int | None,
    missed: int | None,
    completion: int | None,
    faithfulness: int | None,
) -> JudgeRowStatus:
    scores = (relevance, unnecessary, missed, completion, faithfulness)
    filled = sum(1 for item in scores if item is not None)
    if filled == 0:
        return "unscored"
    if filled == len(scores):
        return "scored"
    return "partial"


def _row_from_raw(row_index: int, raw: dict[str, str]) -> JudgeRow:
    relevance = _parse_optional_int(raw.get("recommendation_relevance_1_5"))
    unnecessary = _parse_optional_int(raw.get("unnecessary_question_0_1"))
    missed = _parse_optional_int(raw.get("missed_clarification_0_1"))
    completion = _parse_optional_int(raw.get("task_completion_0_1"))
    faithfulness = _parse_optional_int(raw.get("explanation_faithfulness_1_5"))
    turns = _parse_optional_int(raw.get("turns_to_completion")) or 0
    return JudgeRow(
        row_index=row_index,
        source=raw.get("source", ""),
        suite=raw.get("suite", ""),
        scenario_id=raw.get("scenario_id", ""),
        task_type=raw.get("task_type", ""),
        mode=raw.get("mode", ""),
        turns_to_completion=turns,
        question=raw.get("question", ""),
        final_reply=raw.get("final_reply", ""),
        clarification_asked=_parse_bool(raw.get("clarification_asked")),
        asked_slots=raw.get("asked_slots", ""),
        decision_source=raw.get("decision_source", ""),
        reason_codes=raw.get("reason_codes", ""),
        tools_used=raw.get("tools_used", ""),
        applied_rule_ids=raw.get("applied_rule_ids", ""),
        products_summary=raw.get("products_summary", ""),
        outfits_summary=raw.get("outfits_summary", ""),
        grounding_ok=_parse_bool(raw.get("grounding_ok")),
        recommendation_relevance_1_5=relevance,
        unnecessary_question_0_1=unnecessary,
        missed_clarification_0_1=missed,
        task_completion_0_1=completion,
        explanation_faithfulness_1_5=faithfulness,
        judge_notes=raw.get("judge_notes", ""),
        judge_name=raw.get("judge_name", ""),
        updated_at=raw.get("updated_at", ""),
        status=_compute_row_status(
            relevance, unnecessary, missed, completion, faithfulness
        ),
    )


def load_judge_rows(csv_path: Path) -> list[JudgeRow]:
    """Load and parse all rows from a UTF-8-SIG judge sheet CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Judge sheet CSV not found: {csv_path}")

    with _CSV_LOCK:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [_row_from_raw(idx, raw) for idx, raw in enumerate(reader)]


def _opt_int_to_str(value: int | None) -> str:
    return "" if value is None else str(value)


def update_judge_row(
    csv_path: Path, row_index: int, update: JudgeScoreUpdate
) -> JudgeRow:
    """Update editable judge columns for one row and persist atomically."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Judge sheet CSV not found: {csv_path}")

    with _CSV_LOCK:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            raw_rows = list(reader)

        if row_index < 0 or row_index >= len(raw_rows):
            raise IndexError(
                f"Row index {row_index} out of range (total={len(raw_rows)})"
            )

        target = raw_rows[row_index]
        current_updated_at = target.get("updated_at", "")
        if (
            update.expected_updated_at is not None
            and not update.force
            and current_updated_at
            and current_updated_at != update.expected_updated_at
        ):
            raise RowConflictError(_row_from_raw(row_index, target))

        target["recommendation_relevance_1_5"] = _opt_int_to_str(
            update.recommendation_relevance_1_5
        )
        target["unnecessary_question_0_1"] = _opt_int_to_str(
            update.unnecessary_question_0_1
        )
        target["missed_clarification_0_1"] = _opt_int_to_str(
            update.missed_clarification_0_1
        )
        target["task_completion_0_1"] = _opt_int_to_str(update.task_completion_0_1)
        target["explanation_faithfulness_1_5"] = _opt_int_to_str(
            update.explanation_faithfulness_1_5
        )
        target["judge_notes"] = update.judge_notes

        if update.judge_name is not None:
            for col in ("judge_name", "updated_at"):
                if col not in fieldnames:
                    fieldnames.append(col)
            target["judge_name"] = update.judge_name.strip()
        elif update.expected_updated_at is not None and "updated_at" not in fieldnames:
            fieldnames.append("updated_at")

        if "updated_at" in fieldnames:
            target["updated_at"] = datetime.now(UTC).isoformat(timespec="milliseconds")

        tmp_path = csv_path.with_suffix(".csv.tmp")
        with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)
        tmp_path.replace(csv_path)

        return _row_from_raw(row_index, target)


def _avg_score(values: list[int | None], *, digits: int) -> float | None:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return round(sum(non_null) / len(non_null), digits)


def _pct_flag(values: list[int | None], *, digits: int) -> float | None:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    return round((sum(non_null) / len(non_null)) * 100.0, digits)


def _fmt_float(val: float | None, *, digits: int, suffix: str = "") -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}{suffix}"


def compute_judge_summary(
    rows: list[JudgeRow], suite: str = "all"
) -> JudgeSummaryResponse:
    """Compute per-mode human judge metrics and a Markdown summary table."""
    filtered = rows if suite == "all" else [r for r in rows if r.suite == suite]

    grouped: dict[str, list[JudgeRow]] = {}
    for row in filtered:
        grouped.setdefault(row.mode, []).append(row)

    ordered_modes = [m for m in CANONICAL_MODES if m in grouped] + sorted(
        m for m in grouped if m not in CANONICAL_MODES
    )

    by_mode: list[JudgeModeSummary] = []
    for mode_name in ordered_modes:
        items = grouped[mode_name]
        by_mode.append(
            JudgeModeSummary(
                mode=mode_name,
                total=len(items),
                scored=sum(1 for i in items if i.status == "scored"),
                partial=sum(1 for i in items if i.status == "partial"),
                unscored=sum(1 for i in items if i.status == "unscored"),
                avg_relevance=_avg_score(
                    [i.recommendation_relevance_1_5 for i in items], digits=2
                ),
                unnecessary_question_rate=_pct_flag(
                    [i.unnecessary_question_0_1 for i in items], digits=1
                ),
                missed_clarification_rate=_pct_flag(
                    [i.missed_clarification_0_1 for i in items], digits=1
                ),
                task_completion_rate=_pct_flag(
                    [i.task_completion_0_1 for i in items], digits=1
                ),
                avg_faithfulness=_avg_score(
                    [i.explanation_faithfulness_1_5 for i in items], digits=2
                ),
            )
        )

    md_lines = [
        (
            "| Chế độ | Đã chấm | Relevance (1-5) | Unnecessary (%) | "
            "Missed (%) | Task Comp (%) | Faithfulness (1-5) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in by_mode:
        md_lines.append(
            f"| `{item.mode}` | {item.scored}/{item.total} | "
            f"{_fmt_float(item.avg_relevance, digits=2)} | "
            f"{_fmt_float(item.unnecessary_question_rate, digits=1, suffix='%')} | "
            f"{_fmt_float(item.missed_clarification_rate, digits=1, suffix='%')} | "
            f"{_fmt_float(item.task_completion_rate, digits=1, suffix='%')} | "
            f"{_fmt_float(item.avg_faithfulness, digits=2)} |"
        )

    return JudgeSummaryResponse(
        suite=suite,
        total_rows=len(filtered),
        scored_rows=sum(1 for r in filtered if r.status == "scored"),
        partial_rows=sum(1 for r in filtered if r.status == "partial"),
        unscored_rows=sum(1 for r in filtered if r.status == "unscored"),
        by_mode=by_mode,
        markdown_table="\n".join(md_lines),
    )
