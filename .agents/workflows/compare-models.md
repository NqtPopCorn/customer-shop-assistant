# Workflow: /compare-models

Workflow to evaluate Ollama and LM Studio side-by-side using the research evaluation dataset.

## Steps

1. **Verify Services**
   Ensure both local LLM endpoints are running:
   - Ollama service running on `http://localhost:11434`
   - LM Studio local server running on `http://127.0.0.1:1235`

2. **Run Evaluation on Ollama**
   ```powershell
   $env:LLM_PROVIDER="ollama"
   uv run python -m customer_shop_assistant.eval.runner --output logs/eval_ollama.jsonl
   ```

3. **Run Evaluation on LM Studio**
   ```powershell
   $env:LLM_PROVIDER="lmstudio"
   uv run python -m customer_shop_assistant.eval.runner --output logs/eval_lmstudio.jsonl
   ```

4. **Summarize Differences**
   Compare the generated outputs in `logs/` on:
   - Response latency (`latency_ms`)
   - Accuracy with respect to `data/products.json` and `data/policies.json`
   - Tone, helpfulness, and structure of answers in Vietnamese
