# Báo cáo Thực nghiệm E1: Clarification Strategy trên Outfit KG

- **Thời điểm chạy**: 2026-11-25 05:30:44 UTC
- **KG Backend**: `neo4j` (`data/outfit_kg/ontology/`)
- **Snapshot**: `data/hm_seed_1000.json` (1,000 SP, 14 categories, 33 màu)
- **Dataset**: `eval/e1_scenarios.json` (**28 Dev** / **56 Frozen Test**)
- **Ngưỡng đóng băng (`CounterfactualThresholds`)**: `jaccard_threshold=0.45`, `min_viable_outfits=1`, `alternatives_max_branches=3`

---

## 1. So sánh 3 Chiến lược trên Tập Frozen Test

| Chế độ | Clarify (%) | Qs/Req | Turns | Policy (%) | Gold Align (%) | Unnecessary (%) | Missed (%) | Constraint (%) | Color Score | Harmony (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | 0.0% | 0.00 | 1.00 | 100.0% | 57.1% | 0.0% | 42.9% | 70.0% | 0.875 | 97.1% | 0.05 |
| `ALWAYS_CLARIFY` | 48.2% | 0.62 | 1.54 | 100.0% | 73.2% | 14.3% | 12.5% | 92.5% | 0.885 | 97.1% | 0.04 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 48.2% | 0.45 | 1.39 | 100.0% | 98.2% | 1.8% | 0.0% | 92.5% | 0.886 | 97.1% | 11.53 |

### 1.1 Phân rã theo nhóm tác vụ (`OPEN` vs `ANCHOR` vs `Other Intents`)

| Chế độ | Nhóm | Số ca | Clarify (%) | Gold Align (%) | Unnecessary (%) | Missed (%) |
|---|---|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | OPEN | 16 | 0.0% | 37.5% | 0.0% | 62.5% |
| `IMMEDIATE_ANSWER` | ANCHOR | 16 | 0.0% | 62.5% | 0.0% | 37.5% |
| `IMMEDIATE_ANSWER` | COLOR | 8 | 0.0% | 37.5% | 0.0% | 62.5% |
| `IMMEDIATE_ANSWER` | Other Intents | 16 | 0.0% | 81.2% | 0.0% | 18.8% |
| `ALWAYS_CLARIFY` | OPEN | 16 | 68.8% | 93.8% | 6.2% | 0.0% |
| `ALWAYS_CLARIFY` | ANCHOR | 16 | 62.5% | 37.5% | 43.8% | 18.8% |
| `ALWAYS_CLARIFY` | COLOR | 8 | 12.5% | 50.0% | 0.0% | 50.0% |
| `ALWAYS_CLARIFY` | Other Intents | 16 | 31.2% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | OPEN | 16 | 68.8% | 93.8% | 6.2% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | ANCHOR | 16 | 37.5% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | COLOR | 8 | 62.5% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | Other Intents | 16 | 31.2% | 100.0% | 0.0% | 0.0% |

## 2. Nghiên cứu Ablation (`CONDITIONAL_CLARIFICATION`)

| Biến thể | Clarify (%) | Gold Align (%) | Unnecessary (%) | Missed (%) | Evidence Rule Cov (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `CONDITIONAL_CLARIFICATION (Catalog-only)` | 41.1% | 91.1% | 1.8% | 7.1% | 60.7% | 0.49 |
| `CONDITIONAL_CLARIFICATION (KG-only)` | 25.0% | 75.0% | 1.8% | 23.2% | 60.7% | 2.28 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 48.2% | 98.2% | 1.8% | 0.0% | 60.7% | 11.53 |

## 3. Hiệu chỉnh Ngưỡng trên Tập Dev (`calibrate_thresholds_on_dev`)

| `jaccard_threshold` | `min_viable_outfits` | Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 1 | 100.0% | 0.0% | 0.0% | 43.93 |
| 0.25 | 2 | 100.0% | 0.0% | 0.0% | 28.46 |
| 0.35 | 1 | 100.0% | 0.0% | 0.0% | 36.68 |
| 0.35 | 2 | 100.0% | 0.0% | 0.0% | 26.77 |
| 0.45 | 1 | 100.0% | 0.0% | 0.0% | 28.94 |
| 0.45 | 2 | 100.0% | 0.0% | 0.0% | 22.93 |
| 0.55 | 1 | 100.0% | 0.0% | 0.0% | 25.19 |
| 0.55 | 2 | 100.0% | 0.0% | 0.0% | 20.44 |
| 0.65 | 1 | 100.0% | 0.0% | 0.0% | 14.00 |
| 0.65 | 2 | 100.0% | 0.0% | 0.0% | 16.21 |

## 4. Mẫu Kiểm thử End-to-End (Local LLM + Live Neo4j)

| Scenario | Mode | Clarified | Source | Tools | Outfits | Latency (ms) | Total Tokens | Status |
|---|---|---:|---|---|---:|---:|---:|---|
| `E1_TEST_OPEN_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `suggest_outfit` | 0 | 25782.9 | 3353 | OK |
| `E1_TEST_OPEN_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 7138.7 | 328 | OK |
| `E1_TEST_OPEN_01` | `CONDITIONAL_CLARIFICATION` | True | `mode_c_hard_ask` | `-` | 0 | 7088.3 | 328 | OK |
| `E1_TEST_ANCHOR_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `suggest_outfit` | 3 | 26675.5 | 4174 | OK |
| `E1_TEST_ANCHOR_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 4129.2 | 292 | OK |
| `E1_TEST_ANCHOR_01` | `CONDITIONAL_CLARIFICATION` | False | `mode_c_counterfactual` | `suggest_outfit` | 3 | 5449.6 | 1443 | OK |
| `E1_TEST_SIZE_01` | `IMMEDIATE_ANSWER` | False | `mode_a_immediate` | `-` | 0 | 17918.0 | 2163 | OK |
| `E1_TEST_SIZE_01` | `ALWAYS_CLARIFY` | True | `mode_b_required_check` | `-` | 0 | 3015.6 | 276 | OK |
| `E1_TEST_SIZE_01` | `CONDITIONAL_CLARIFICATION` | True | `mode_c_hard_ask` | `-` | 0 | 2812.0 | 276 | OK |

## 5. Phân tích Sai lệch (`CONDITIONAL_CLARIFICATION KG+Catalog`)

- `E1_TEST_OPEN_11` (gold=`proceed`, asked=True, source=`mode_c_hard_ask`, reason_codes=`['HARD_ASK_BROAD_EMPTY_INQUIRY']`)
