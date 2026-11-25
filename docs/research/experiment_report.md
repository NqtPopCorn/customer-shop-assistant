# Báo cáo Thực nghiệm Nghiên cứu Trợ lý Chatbot Thời trang

- **Thời điểm chạy**: 2026-11-25 16:03:45 UTC
- **LLM Cấu hình**: `ollama` / `qwen3.5:2b`
- **KG Backend**: `neo4j` (Neo4j: `OK` (connected)) | **Main BE**: `OK` (HTTP 200)
- **Snapshot**: `data/hm_seed_1000.json` (1,000 SP, 14 categories, 33 màu)
- **Dataset**: `eval/e1_scenarios.json` (**28 Dev** / **56 Frozen Test**) + `eval/e1_contrastive_s1_s10.json` (12 nhánh S1–S10)
- **Ngưỡng đóng băng (`CounterfactualThresholds`)**: `jaccard_threshold=0.45`, `min_viable_outfits=1`, `min_color_quality=0.72`, `color_gain_threshold=0.12`, `alternatives_max_branches=3`

---

## 1. Tầng 1 — So sánh 3 Chiến lược trên Tập Frozen Test

| Chế độ | Clarify (%) | Qs/Req | Turns | Policy (%) | Gold Align (%) | Slot P/R/Exact | Unnecessary (%) | Missed (%) | Constraint (%) | Color Score | Harmony (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | 0.0% | 0.00 | 1.00 | 100.0% | 57.1% | 0%/0%/0% | 0.0% | 42.9% | 70.0% | 0.875 | 97.1% | 0.04 |
| `ALWAYS_CLARIFY` | 48.2% | 0.68 | 1.55 | 100.0% | 73.2% | 53%/71%/53% | 14.3% | 12.5% | 92.5% | 0.885 | 97.1% | 0.04 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 48.2% | 0.54 | 1.48 | 100.0% | 98.2% | 94%/98%/92% | 1.8% | 0.0% | 100.0% | 0.892 | 97.3% | 12.70 |

### 1.1 Phân rã theo nhóm tác vụ (`OPEN` vs `ANCHOR` vs `COLOR` vs `Other`)

| Chế độ | Nhóm | Số ca | Clarify (%) | Gold Align (%) | Slot Exact (%) | Unnecessary (%) | Missed (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | OPEN | 16 | 0.0% | 37.5% | 0.0% | 0.0% | 62.5% |
| `IMMEDIATE_ANSWER` | ANCHOR | 16 | 0.0% | 62.5% | 0.0% | 0.0% | 37.5% |
| `IMMEDIATE_ANSWER` | COLOR | 8 | 0.0% | 37.5% | 0.0% | 0.0% | 62.5% |
| `IMMEDIATE_ANSWER` | Other Intents | 16 | 0.0% | 81.2% | 0.0% | 0.0% | 18.8% |
| `ALWAYS_CLARIFY` | OPEN | 16 | 68.8% | 93.8% | 90.9% | 6.2% | 0.0% |
| `ALWAYS_CLARIFY` | ANCHOR | 16 | 62.5% | 37.5% | 23.1% | 43.8% | 18.8% |
| `ALWAYS_CLARIFY` | COLOR | 8 | 12.5% | 50.0% | 20.0% | 0.0% | 50.0% |
| `ALWAYS_CLARIFY` | Other Intents | 16 | 31.2% | 100.0% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | OPEN | 16 | 68.8% | 93.8% | 90.9% | 6.2% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | ANCHOR | 16 | 37.5% | 100.0% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | COLOR | 8 | 62.5% | 100.0% | 100.0% | 0.0% | 0.0% |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | Other Intents | 16 | 31.2% | 100.0% | 66.7% | 0.0% | 0.0% |

## 2. Nghiên cứu Ablation

### 2.1 Ablation Chiến lược Hỏi lại (`CONDITIONAL_CLARIFICATION`)

| Biến thể | Clarify (%) | Gold Align (%) | Slot Exact (%) | Unnecessary (%) | Missed (%) | Evidence Rule Cov (%) | Probe TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CONDITIONAL_CLARIFICATION (Catalog-only)` | 44.6% | 94.6% | 84.0% | 1.8% | 3.6% | 66.1% | 0.52 |
| `CONDITIONAL_CLARIFICATION (KG-only)` | 25.0% | 75.0% | 40.0% | 1.8% | 23.2% | 60.7% | 2.99 |
| `CONDITIONAL_CLARIFICATION (KG+Catalog)` | 48.2% | 98.2% | 92.0% | 1.8% | 0.0% | 66.1% | 12.70 |

### 2.2 Ablation Color-aware Reranking (v1.2 Blueprint-only vs v1.3)

| Phiên bản | Số ca | Outfit Returned (%) | Mean Color Score | Pair Harmony | Context Color Fit | Color Harmony Rate (%) |
|---|---:|---:|---:|---:|---:|---:|
| `v1.2 Blueprint-only` | 37 | 100.0% | 0.7737 | 0.7632 | 0.7345 | 61.3% |
| `v1.3 Color-aware (70/30)` | 37 | 100.0% | 0.8914 | 0.9386 | 0.8057 | 97.3% |

- **Chênh lệch (v1.3 − v1.2)**: `Δ Mean Color Score = +0.1177`, `Δ Color Harmony Rate = +36.0%`

## 3. Hiệu chỉnh Ngưỡng trên Tập Dev (`calibrate_thresholds_on_dev`)

### 3.1 Stage A — `jaccard_threshold` × `min_viable_outfits`

| `jaccard_threshold` | `min_viable_outfits` | Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 1 | 100.0% | 0.0% | 0.0% | 1.73 |
| 0.25 | 2 | 100.0% | 0.0% | 0.0% | 1.23 |
| 0.35 | 1 | 100.0% | 0.0% | 0.0% | 4.07 |
| 0.35 | 2 | 100.0% | 0.0% | 0.0% | 2.15 |
| 0.45 | 1 | 100.0% | 0.0% | 0.0% | 1.55 |
| 0.45 | 2 | 100.0% | 0.0% | 0.0% | 1.59 |
| 0.55 | 1 | 100.0% | 0.0% | 0.0% | 1.34 |
| 0.55 | 2 | 100.0% | 0.0% | 0.0% | 1.42 |
| 0.65 | 1 | 100.0% | 0.0% | 0.0% | 1.28 |
| 0.65 | 2 | 100.0% | 0.0% | 0.0% | 1.26 |

### 3.2 Stage B — `min_color_quality` × `color_gain_threshold`

| `min_color_quality` | `color_gain_threshold` | Gold Align (%) | Missed (%) | Unnecessary (%) | Probe TB (ms) |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 0.08 | 100.0% | 0.0% | 0.0% | 1.31 |
| 0.65 | 0.12 | 100.0% | 0.0% | 0.0% | 1.18 |
| 0.65 | 0.18 | 100.0% | 0.0% | 0.0% | 1.24 |
| 0.72 | 0.08 | 100.0% | 0.0% | 0.0% | 1.68 |
| 0.72 | 0.12 | 100.0% | 0.0% | 0.0% | 2.13 |
| 0.72 | 0.18 | 100.0% | 0.0% | 0.0% | 1.31 |
| 0.8 | 0.08 | 100.0% | 0.0% | 0.0% | 1.33 |
| 0.8 | 0.12 | 100.0% | 0.0% | 0.0% | 1.35 |
| 0.8 | 0.18 | 100.0% | 0.0% | 0.0% | 1.41 |

> *Ghi chú độ vững (Robustness)*: Toàn bộ lưới hiệu chuẩn Dev đạt **100.0%** `Gold Align`, cho thấy ngưỡng quyết định vững trên tập Dev; cơ chế tie-break chọn cấu hình mặc định trung tâm.

## 4. Tầng 2 — Kiểm thử End-to-End Đa lượt với Local LLM

### 4.1 Ma trận Đối chứng 3 Chế độ (S1–S10)

| Scenario | Mode A (`IMMEDIATE`) | Mode B (`ALWAYS`) | Mode C (`CONDITIONAL`) |
|---|---|---|---|
| `S1` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (PROCEED, `mode_c_counterfactual`, 1t) |
| `S2` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (PROCEED, `mode_c_hard_answer`, 1t) |
| `S3` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (PROCEED, `mode_b_required_check`, 1t) | PASS (ASK, `mode_c_counterfactual`, 2t) |
| `S4` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (PROCEED, `mode_b_required_check`, 1t) | PASS (ASK, `mode_c_counterfactual`, 2t) |
| `S5` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (PROCEED, `mode_b_required_check`, 1t) | PASS (ASK, `mode_c_counterfactual`, 2t) |
| `S6A` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (ASK, `mode_c_hard_ask`, 2t) |
| `S6B` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (ASK, `mode_c_hard_ask`, 2t) |
| `S7` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (ASK, `mode_c_counterfactual`, 2t) |
| `S8` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (ASK, `mode_c_hard_ask`, 2t) |
| `S9` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (ASK, `mode_b_required_check`, 2t) | PASS (PROCEED, `mode_c_llm`, 1t) |
| `S10A` | PASS (PROCEED, `mode_a_immediate`, 1t) | PASS (PROCEED, `mode_b_required_check`, 1t) | PASS (PROCEED, `mode_c_hard_answer`, 1t) |
| `S10B` | FAIL (PROCEED, `non_shopping_bypass`, 1t) | FAIL (PROCEED, `non_shopping_bypass`, 1t) | FAIL (PROCEED, `non_shopping_bypass`, 1t) |

### 4.2 Tổng hợp E2E — Bộ Đối chứng S1–S10 (Stress-test)

| Chế độ | Số ca | Errors | Pass (%) | Turns TB | Clarify (%) | Mode Comp (%) | Grounding (%) | Task Comp (%) | Tokens (In/Out/Tot) | Latency TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | 12 | 0 | 91.7% | 1.00 | 0.0% | 100.0% | 91.7% | 100.0% | 3995/440/4435 | 26931.2 |
| `ALWAYS_CLARIFY` | 12 | 0 | 91.7% | 1.58 | 58.3% | 100.0% | 91.7% | 100.0% | 2730/580/3310 | 33014.5 |
| `CONDITIONAL_CLARIFICATION` | 12 | 0 | 91.7% | 1.58 | 58.3% | 100.0% | 91.7% | 100.0% | 4698/630/5328 | 36159.2 |

### 4.3 Tổng hợp E2E — Bộ Stratified Frozen Test (Đại diện)

| Chế độ | Số ca | Errors | Pass (%) | Turns TB | Clarify (%) | Mode Comp (%) | Grounding (%) | Task Comp (%) | Tokens (In/Out/Tot) | Latency TB (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `IMMEDIATE_ANSWER` | 12 | 1 | 91.7% | 1.00 | 0.0% | 91.7% | 91.7% | 91.7% | 4285/475/4760 | 25230.9 |
| `ALWAYS_CLARIFY` | 12 | 0 | 100.0% | 1.75 | 66.7% | 100.0% | 100.0% | 100.0% | 3612/615/4227 | 31104.0 |
| `CONDITIONAL_CLARIFICATION` | 12 | 0 | 100.0% | 1.58 | 58.3% | 100.0% | 100.0% | 100.0% | 4219/622/4840 | 30996.1 |

## 5. Phân tích Sai lệch tự động (Error Analysis)

### 5.1 Sai lệch Nhãn Gold ở Tầng 1 (`CONDITIONAL_CLARIFICATION`)
- `E1_TEST_OPEN_11` (gold=`proceed`, asked=True, source=`mode_c_hard_ask`, reason_codes=`['HARD_ASK_BROAD_EMPTY_INQUIRY']`)

### 5.2 Sai lệch Slot hỏi ở Tầng 1 (`CONDITIONAL_CLARIFICATION`)
- `E1_TEST_OPEN_11`: asked=`['occasion']`, expected=`[]` (P=0.0, R=None)
- `E1_TEST_SEARCH_01`: asked=`['target_gender', 'occasion']`, expected=`['target_gender', 'category']` (P=0.5, R=0.5)

### 5.3 Lỗi Runtime / Grounding / Tool bắt được ở Tầng 2 E2E
- `[contrastive] S10B` (`IMMEDIATE_ANSWER`): Missing policy citation 'shipping.md' in reply.; Missing policy citation 'returns.md' in reply.
- `[contrastive] S10B` (`ALWAYS_CLARIFY`): Missing policy citation 'shipping.md' in reply.; Missing policy citation 'returns.md' in reply.
- `[contrastive] S10B` (`CONDITIONAL_CLARIFICATION`): Missing policy citation 'shipping.md' in reply.; Missing policy citation 'returns.md' in reply.
- `[stratified] E1_TEST_COL_IMP_01` (`IMMEDIATE_ANSWER`): TimeoutError: 
