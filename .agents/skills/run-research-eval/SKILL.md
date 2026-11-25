---
name: run-research-eval
description: "Procedure for evaluating clarification strategies, Outfit KG ablations, and multi-turn E2E quality across Ollama and LM Studio."
---

# Run Research Evaluation

Use this skill to run the 3-tier evaluation suite and compare Ollama vs LM Studio.

## Datasets
- **Tier 1 (Controlled Policy & KG)**: `eval/e1_scenarios.json` (28 Dev / 56 Frozen Test scenarios across `OUTFIT_ADVICE`, `PRODUCT_SEARCH`, `SIZE_ADVICE`, `BODY_FIT_ADVICE`, `NON_SHOPPING`).
- **Tier 2 (Contrastive Multi-turn E2E)**: `eval/e1_contrastive_s1_s10.json` (12 branches covering S1–S10 across Modes A, B, C) + stratified sample from `eval/e1_scenarios.json`.
- **Tier 3 (Functional Regression)**: `eval/questions.json` v2.0 (72 unique cases across 7 categories).

## Steps

### 1. Verify Prerequisites
- **LLM Provider**:
  - Ollama: verify server is running and `OLLAMA_MODEL` is pulled (`ollama list`).
  - LM Studio: ensure server is active on `127.0.0.1:1235` with `LMSTUDIO_MODEL`.
- **Optional External Services (for Tier 2/3)**:
  - Main BE (`MAIN_BE_BASE_URL`) for live `search_products` and `check_voucher`.
  - Neo4j (`docker-compose up -d neo4j` in `H:\Code\PET\e-commerce-clothes`) when passing `--backend neo4j`.

### 2. Run Unified Research Benchmark (E1)
```powershell
# Tier 1 fast offline benchmark + Dev threshold calibration + Ablations (no GPU needed)
uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --tier1-only --calibrate --ablation

# Full E1 Research run (Tier 1 + Tier 2 Multi-turn E2E S1-S10 + 12 stratified Test cases)
$env:LLM_PROVIDER="lmstudio"
uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --ablation --e2e-limit 12

# Full E1 with Neo4j backend and all 56 Frozen Test cases in Tier 2
uv run python -m customer_shop_assistant.eval.experiments --experiment e1 --backend neo4j --ablation --e2e-all
```

### 3. Run Tier 3 Functional Regression
```powershell
# Quick dry-run
uv run python -m customer_shop_assistant.eval.runner --dry-run --limit 10

# Live comparison across providers
$env:LLM_PROVIDER="ollama"
uv run python -m customer_shop_assistant.eval.runner --output logs/eval_ollama.jsonl

$env:LLM_PROVIDER="lmstudio"
uv run python -m customer_shop_assistant.eval.runner --output logs/eval_lmstudio.jsonl
```

### 4. Inspect Outputs
- **Unified Research Report**: `docs/research/experiment_report.md` (Tier 1 metrics + Slot P/R/Exact, Policy & Color v1.2 vs v1.3 Ablation, Dev Calibration Sweeps, Tier 2 S1–S10 Matrix & Token/Latency/Grounding metrics, Auto Error Analysis).
- **Tier-2 Human Judge Sheet**: `eval/e1_tier2_judge_sheet.csv` (exported with UTF-8 BOM and real LLM replies when Tier 2 runs).
- **JSONL Telemetry**: `logs/e1_test_results.jsonl`, `logs/e1_e2e_results.jsonl`, `logs/eval_*.jsonl`.
