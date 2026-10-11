---
name: run-research-eval
description: "Procedure for evaluating test customer questions against Ollama and LM Studio models."
---

# Run Research Evaluation

Use this skill to benchmark and compare chatbot response quality between Ollama and LM Studio.

## Dataset
- Test questions: `eval/questions.json` (10 Vietnamese customer queries covering products, sizes, policies, and edge cases).

## Steps

### 1. Verify Prerequisites
Ensure the target LLM provider service is active:
- If Ollama: run `ollama list` and verify `llama3.2` is loaded.
- If LM Studio: ensure LM Studio server is running on `127.0.0.1:1235` (or configured port) with `google/gemma-4-e2b`.

### 2. Run Evaluation Script
Execute an evaluation run across both providers:
```powershell
# Run with Ollama
$env:LLM_PROVIDER="ollama"
uv run python -m customer_shop_assistant.eval.runner --output logs/eval_ollama.jsonl

# Run with LM Studio
$env:LLM_PROVIDER="lmstudio"
uv run python -m customer_shop_assistant.eval.runner --output logs/eval_lmstudio.jsonl
```

### 3. Compare Results
Inspect logs in `logs/` to compare:
- **Accuracy**: Did the model use correct catalog prices and sizes from `data/products.json`?
- **Policy adherence**: Did the model answer return policies accurately according to `data/policies.json`?
- **Latency**: Compare average `latency_ms` between providers.
- **Language fluidity**: Review tone and Vietnamese grammatical naturalness.
