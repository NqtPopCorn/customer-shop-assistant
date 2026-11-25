import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from customer_shop_assistant.config import get_settings
from customer_shop_assistant.schemas.chat import LangChainUsageMetadata

logger = logging.getLogger(__name__)


def _write_log_line(log_path: Path, line: str) -> None:
    """Synchronously append line to log file, creating parent directories if needed."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as err:
        logger.warning("Failed writing research telemetry line: %s", err)


async def log_chat_turn(
    session_id: str,
    turn_id: int,
    user_message: str,
    assistant_message: str | None,
    latency_ms: float,
    token_usage: LangChainUsageMetadata | dict[str, Any] | None = None,
    error: str | None = None,
    system_prompt_version: str = "v1.0",
    *,
    clarification_mode: str | None = None,
    turn_intent: str | None = None,
    clarification_asked: bool = False,
    question_count: int = 0,
    asked_slots: list[str] | None = None,
    slot_states: dict[str, Any] | None = None,
    decision_source: str | None = None,
    decision_reason: str | None = None,
    tools_used: list[str] | None = None,
    reason_codes: list[str] | None = None,
    impact_signals: dict[str, Any] | None = None,
    evidence_rule_ids: list[str] | None = None,
) -> None:
    """Asynchronously log chat turn telemetry to logs/chat_turns.jsonl."""
    try:
        settings = get_settings()
        active_model = (
            settings.OLLAMA_MODEL
            if settings.LLM_PROVIDER == "ollama"
            else settings.LMSTUDIO_MODEL
        )

        formatted_usage: dict[str, int] | None = None
        if token_usage:
            if isinstance(token_usage, LangChainUsageMetadata):
                formatted_usage = {
                    "prompt_tokens": token_usage.input_tokens,
                    "completion_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.total_tokens,
                }
            elif isinstance(token_usage, dict):
                formatted_usage = {
                    "prompt_tokens": token_usage.get(
                        "input_tokens", token_usage.get("prompt_tokens", 0)
                    ),
                    "completion_tokens": token_usage.get(
                        "output_tokens", token_usage.get("completion_tokens", 0)
                    ),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "turn_id": turn_id,
            "provider": settings.LLM_PROVIDER,
            "model": active_model,
            "temperature": settings.TEMPERATURE,
            "system_prompt_version": system_prompt_version,
            "clarification_mode": clarification_mode,
            "turn_intent": turn_intent,
            "clarification_asked": clarification_asked,
            "question_count": question_count,
            "asked_slots": list(asked_slots or []),
            "slot_states": slot_states or {},
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "tools_used": list(tools_used or []),
            "user_message": user_message,
            "assistant_message": assistant_message,
            "latency_ms": round(float(latency_ms), 2),
            "token_usage": formatted_usage,
            "error": error,
        }
        if reason_codes is not None:
            record["reason_codes"] = list(reason_codes)
        if impact_signals is not None:
            record["impact_signals"] = impact_signals
        if evidence_rule_ids is not None:
            record["evidence_rule_ids"] = list(evidence_rule_ids)

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        log_file = project_root / "logs" / "chat_turns.jsonl"

        line = json.dumps(record, ensure_ascii=False)
        await asyncio.to_thread(_write_log_line, log_file, line)
    except Exception as e:
        logger.warning("Exception during log_chat_turn telemetry: %s", e)
