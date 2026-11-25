# Báo cáo Thực nghiệm E1: Clarification Strategy trên Outfit KG

- **Thời điểm chạy**: 2026-11-25 03:54:04 UTC
- **KG Backend**: `neo4j` (`data/outfit_kg/ontology/`)
- **Snapshot**: `data/hm_seed_1000.json` (1,000 SP, 14 categories, 33 màu)
- **Dataset**: `eval/e1_scenarios.json` (**24 Dev** / **48 Frozen Test**)
- **Ngưỡng đóng băng (`CounterfactualThresholds`)**: `jaccard_threshold=0.45`, `min_viable_outfits=1`, `alternatives_max_branches=3`

---

## 1. So sánh 3 Chiến lược trên Tập Frozen Test

| Chế độ | Clarify (%) | Qs/Req | Turns | Policy (%) | Gold Align (%) | Unnecessary (%) | Missed (%) | Constraint (%) | Rule Viol (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | 0.0% | 0.00 | 1.00 | 100.0% | 60.4% | 0.0% | 39.6% | 62.5% | 0.0% | 0.22 |
| `ALWAYS_CLARIFY` | 54.2% | 0.71 | 1.60 | 100.0% | 77.1% | 16.7% | 6.2% | 90.6% | 0.0% | 0.15 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 45.8% | 0.46 | 1.40 | 100.0% | 97.9% | 2.1% | 0.0% | 90.6% | 0.0% | 18.12 |

### 1.1 Phân rã theo nhóm tác vụ (`OPEN` vs `ANCHOR` vs `Other Intents`)

| Chế độ | Nhóm | Số ca | Clarify (%) | Gold Align (%) | Unnecessary (%) | Missed (%) |
|---|---|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | OPEN | 16 | 0.0% | 37.5% | 0.0% | 62.5% |
| `IMMEDIATE_ANSWER` | ANCHOR | 16 | 0.0% | 62.5% | 0.0% | 37.5% |
| `IMMEDIATE_ANSWER` | Other Intents | 16 | 0.0% | 81.2% | 0.0% | 18.8% |
| `ALWAYS_CLARIFY` | OPEN | 16 | 68.8% | 93.8% | 6.2% | 0.0% |
| `ALWAYS_CLARIFY` | ANCHOR | 16 | 62.5% | 37.5% | 43.8% | 18.8% |
| `ALWAYS_CLARIFY` | Other Intents | 16 | 31.2% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | OPEN | 16 | 68.8% | 93.8% | 6.2% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | ANCHOR | 16 | 37.5% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | Other Intents | 16 | 31.2% | 100.0% | 0.0% | 0.0% |

## 2. Nghiên cứu Ablation (`CONDITIONAL_CLARIFICATION`)

| Biến thể | Clarify (%) | Gold Align (%) | Unnecessary (%) | Missed (%) | Evidence Rule Cov (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `CONDITIONAL_CLARIFICATION (Catalog-only)` | 45.8% | 97.9% | 2.1% | 0.0% | 54.2% | 0.70 |
| `CONDITIONAL_CLARIFICATION (KG-only)` | 27.1% | 79.2% | 2.1% | 18.8% | 54.2% | 12.64 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 45.8% | 97.9% | 2.1% | 0.0% | 54.2% | 18.12 |

## 3. Hiệu chỉnh Ngưỡng trên Tập Dev (`calibrate_thresholds_on_dev`)

| `jaccard_threshold` | `min_viable_outfits` | Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 1 | 100.0% | 0.0% | 0.0% | 37.93 |
| 0.25 | 2 | 100.0% | 0.0% | 0.0% | 64.23 |
| 0.35 | 1 | 100.0% | 0.0% | 0.0% | 80.30 |
| 0.35 | 2 | 100.0% | 0.0% | 0.0% | 70.19 |
| 0.45 | 1 | 100.0% | 0.0% | 0.0% | 75.47 |
| 0.45 | 2 | 100.0% | 0.0% | 0.0% | 29.46 |
| 0.55 | 1 | 100.0% | 0.0% | 0.0% | 29.92 |
| 0.55 | 2 | 100.0% | 0.0% | 0.0% | 24.71 |
| 0.65 | 1 | 100.0% | 0.0% | 0.0% | 26.48 |
| 0.65 | 2 | 100.0% | 0.0% | 0.0% | 18.52 |

## 4. Mẫu Kiểm thử End-to-End (Local LLM + Live Neo4j)

| Scenario | Mode | Clarified | Source | Tools | Outfits | Latency (ms) | Total Tokens | Status |
|---|---|---:|---|---|---:|---:|---:|---|
| `E1_TEST_OPEN_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `suggest_outfit` | 0 | 23820.6 | 3353 | OK |
| `E1_TEST_OPEN_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 10686.8 | 328 | OK |
| `E1_TEST_OPEN_01` | `CONDITIONAL_CLARIFICATION` | True | `mode_c_hard_ask` | `-` | 0 | 9707.6 | 328 | OK |
| `E1_TEST_ANCHOR_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `suggest_outfit` | 3 | 39299.7 | 4251 | OK |
| `E1_TEST_ANCHOR_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 5629.5 | 292 | OK |
| `E1_TEST_ANCHOR_01` | `CONDITIONAL_CLARIFICATION` | False | `mode_c_counterfactual` | `suggest_outfit` | 3 | 7314.0 | 1443 | OK |
| `E1_TEST_SIZE_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `-` | 0 | 18068.8 | 2163 | OK |
| `E1_TEST_SIZE_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 3628.8 | 276 | OK |
| `E1_TEST_SIZE_01` | `CONDITIONAL_CLARIFICATION` | True | `mode_c_hard_ask` | `-` | 0 | 2724.0 | 276 | OK |

## 5. Phân tích Sai lệch (`CONDITIONAL_CLARIFICATION KG+Catalog`)

- `E1_TEST_OPEN_11` (gold=`proceed`, asked=True, source=`mode_c_hard_ask`, reason_codes=`['HARD_ASK_BROAD_EMPTY_INQUIRY']`)
