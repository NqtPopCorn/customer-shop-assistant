---
trigger: model_decision
description: "Logging standards for conversation turns and research telemetry."
---

# Research Logging Rules

## 1. Research Telemetry Purpose
To evaluate conversational quality across models (Ollama vs LM Studio) and analyze response latency, all chatbot interaction turns must be logged as JSON Lines (JSONL).

## 2. File Location & Storage
- Log files must be appended to `logs/chat_turns.jsonl`.
- The `logs/` directory must be git-ignored except for `.gitkeep`.
- Writes should be asynchronous or safely buffered to prevent blocking request handlers.

## 3. Required JSONL Turn Schema
Every JSONL record must contain the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `string` | ISO-8601 UTC timestamp (e.g. `2026-10-11T10:45:00.123Z`) |
| `session_id` | `string` | Unique conversation session identifier |
| `turn_id` | `integer` | 1-based sequential index of the turn within the session |
| `provider` | `string` | `"ollama"` or `"lmstudio"` |
| `model` | `string` | Model name/tag (e.g. `"llama3.2"`, `"google/gemma-4-e2b"`) |
| `temperature` | `float` | Sampling temperature used for the invocation |
| `system_prompt_version` | `string` | Semantic prompt version identifier (e.g. `"v1.0"`) |
| `user_message` | `string` | Raw customer query string |
| `assistant_message` | `string \| null` | Model response text, or `null` if the turn resulted in an error |
| `latency_ms` | `float` | Execution duration in milliseconds |
| `token_usage` | `object \| null` | `{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}` if available |
| `error` | `string \| null` | Error message string if invocation failed, otherwise `null` |

## 4. Failure Handling
- If the LLM throws an exception or times out, the turn must still be recorded with `assistant_message: null` and the `error` field populated.
- Logging failures must never cause the chat API endpoint itself to crash.
