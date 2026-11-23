import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from customer_shop_assistant.schemas.chat import (
    LangChainRunTrace,
    LangChainUsageMetadata,
)

logger = logging.getLogger(__name__)


def serialize_base_message(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a LangChain BaseMessage into a JSON-compatible dict."""
    data: dict[str, Any] = {
        "type": getattr(msg, "type", "message"),
        "content": getattr(msg, "content", ""),
    }
    if getattr(msg, "name", None):
        data["name"] = msg.name
    if getattr(msg, "tool_calls", None):
        data["tool_calls"] = msg.tool_calls
    if getattr(msg, "tool_call_id", None):
        data["tool_call_id"] = msg.tool_call_id
    if getattr(msg, "usage_metadata", None):
        data["usage_metadata"] = dict(msg.usage_metadata)
    if getattr(msg, "response_metadata", None):
        data["response_metadata"] = dict(msg.response_metadata)
    return data


class LangChainDebugCallbackHandler(AsyncCallbackHandler):
    """Async LangChain callback handler collecting run traces and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._runs_by_id: dict[str, dict[str, Any]] = {}
        self._ordered_run_ids: list[str] = []

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        rid = str(run_id)
        name = (
            (metadata.get("ls_model_name") if metadata else None)
            or serialized.get("name")
            or "ChatModel"
        )
        serialized_messages = [
            serialize_base_message(msg) for sublist in messages for msg in sublist
        ]
        self._runs_by_id[rid] = {
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "name": name,
            "run_type": "chat_model",
            "inputs": {"messages": serialized_messages},
            "outputs": {},
            "metadata": metadata or {},
            "usage_metadata": None,
            "latency_ms": 0,
            "error": None,
            "_start_time": time.time(),
        }
        if rid not in self._ordered_run_ids:
            self._ordered_run_ids.append(rid)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        run = self._runs_by_id.get(rid)
        if not run:
            return

        elapsed = time.time() - run.get("_start_time", time.time())
        run["latency_ms"] = int(elapsed * 1000)

        # Extract outputs and usage
        generations_data: list[dict[str, Any]] = []
        token_usage: dict[str, Any] | None = None

        if response.generations:
            for sublist in response.generations:
                for gen in sublist:
                    gen_item: dict[str, Any] = {"text": gen.text}
                    msg = getattr(gen, "message", None)
                    if msg:
                        gen_item["message"] = serialize_base_message(msg)
                        if getattr(msg, "usage_metadata", None):
                            token_usage = dict(msg.usage_metadata)
                    generations_data.append(gen_item)

        if not token_usage and response.llm_output:
            raw_usage = response.llm_output.get("token_usage")
            if isinstance(raw_usage, dict):
                token_usage = {
                    "input_tokens": raw_usage.get(
                        "prompt_tokens", raw_usage.get("input_tokens", 0)
                    ),
                    "output_tokens": raw_usage.get(
                        "completion_tokens", raw_usage.get("output_tokens", 0)
                    ),
                    "total_tokens": raw_usage.get("total_tokens", 0),
                }

        run["outputs"] = {"generations": generations_data}
        if token_usage:
            run["usage_metadata"] = token_usage

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        run = self._runs_by_id.get(rid)
        if run:
            elapsed = time.time() - run.get("_start_time", time.time())
            run["latency_ms"] = int(elapsed * 1000)
            run["error"] = str(error)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        name = serialized.get("name") or "Tool"
        self._runs_by_id[rid] = {
            "run_id": rid,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "name": name,
            "run_type": "tool",
            "inputs": inputs or {"input": input_str},
            "outputs": {},
            "metadata": metadata or {},
            "usage_metadata": None,
            "latency_ms": 0,
            "error": None,
            "_start_time": time.time(),
        }
        if rid not in self._ordered_run_ids:
            self._ordered_run_ids.append(rid)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        run = self._runs_by_id.get(rid)
        if run:
            elapsed = time.time() - run.get("_start_time", time.time())
            run["latency_ms"] = int(elapsed * 1000)
            run["outputs"] = {"result": output}

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        run = self._runs_by_id.get(rid)
        if run:
            elapsed = time.time() - run.get("_start_time", time.time())
            run["latency_ms"] = int(elapsed * 1000)
            run["error"] = str(error)

    def get_runs(self) -> list[LangChainRunTrace]:
        """Return the collected run traces in execution order."""
        traces: list[LangChainRunTrace] = []
        for rid in self._ordered_run_ids:
            raw = dict(self._runs_by_id[rid])
            raw.pop("_start_time", None)
            traces.append(LangChainRunTrace(**raw))
        return traces

    def get_aggregated_usage(self) -> LangChainUsageMetadata | None:
        """Aggregate total token usage across all chat model runs."""
        total_in = 0
        total_out = 0
        has_usage = False

        for raw in self._runs_by_id.values():
            usage = raw.get("usage_metadata")
            if usage and isinstance(usage, dict):
                has_usage = True
                total_in += usage.get("input_tokens", 0)
                total_out += usage.get("output_tokens", 0)

        if not has_usage:
            return None

        return LangChainUsageMetadata(
            input_tokens=total_in,
            output_tokens=total_out,
            total_tokens=total_in + total_out,
        )

