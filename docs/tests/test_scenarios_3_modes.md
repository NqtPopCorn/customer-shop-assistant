# Tài Liệu Kịch Bản Test Đối Chứng 3 Chế Độ Làm Rõ (Mode A / Mode B / Mode C)

Tài liệu này tổng hợp bộ **10 kịch bản đối chứng (S1–S10)** được thiết kế để làm nổi bật tối đa sự khác biệt giữa 3 chế độ làm rõ của `customer-shop-assistant`, phục vụ kiểm thử thủ công trên UI, demo nghiên cứu và kiểm tra hồi quy (regression QA).

---

## 1. Tổng quan 3 chế độ làm rõ

| Chế độ | Giá trị API (`clarification_mode`) | Cơ chế cốt lõi | Giới hạn hỏi tối đa | Nguồn quyết định chính (`decision_source`) |
|---|---|---|---|---|
| **Mode A** (Ngay) | `IMMEDIATE_ANSWER` | Trả lời ngay lập tức, **0 câu hỏi làm rõ** ở mọi intent. Khi thiếu giới tính → chạy Balanced Multi-Gender Search; khi thiếu số đo size → trả bảng size chuẩn. | `0` lượt / `0` câu | `mode_a_immediate` |
| **Mode B** (Hỏi sâu) | `ALWAYS_CLARIFY` | Luôn kiểm tra Required Slots (và refine danh mục rộng `tops`/`pants`). Khóa tool tìm kiếm ở lượt phỏng vấn (`gate_search=true`). | `2` lượt / `3` câu mỗi form | `mode_b_required_check`, `fallback` |
| **Mode C** (Cân nhắc) | `CONDITIONAL_CLARIFICATION` | Kết hợp **Hard Guards** (mâu thuẫn, thiếu số đo size, câu quá trống) + **Counterfactual Policy trên Outfit KG & tồn kho** + **Gray Zone LLM**. | `1` lượt thường (`2` lượt nếu mâu thuẫn) / `2` câu mỗi form | `mode_c_counterfactual`, `mode_c_hard_ask`, `mode_c_hard_answer`, `mode_c_llm`, `fallback` |

---

## 2. Hướng dẫn chuẩn bị & cách soi kết quả

1. **Khởi động backend** với `DEBUG_CHAT_META=true` trong `.env` để hiển thị panel **LangChain Inspector** (`Runs Tree`, `State & Context`, `Raw JSON`).
2. **Tạo phiên hội thoại mới (Reset session)** trước khi chạy mỗi kịch bản S1–S10 và mỗi lần đổi Mode (A → B → C) để tránh slot lưu phiên (`target_gender`, `measurements`) ảnh hưởng chéo.
3. **Các trường cần đối chiếu trong `debugMeta`**:
   - `mode`, `phase`, `clarification_turns`
   - `decision_source` & `reason_codes`
   - `asked_slots` & số câu hỏi trong `clarificationForm.questions`
   - `tools_available` & danh sách tool trong `runs`
   - `impact_signals` (chỉ có ở Mode C Counterfactual: `feasibility_change`, `category_shift`, `mean_top_k_jaccard`, `can_cover_with_alternatives`, `color_quality_gain`)
   - `products` / `outfits` trả về để render product cards trên UI

---

## 3. Ma trận tổng quan 10 kịch bản đối chứng

| ID | Nhóm tương phản | Prompt test chính | Mode A (`IMMEDIATE_ANSWER`) | Mode B (`ALWAYS_CLARIFY`) | Mode C (`CONDITIONAL_CLARIFICATION`) |
|---|---|---|---|---|---|
| **S1** | **B hỏi thừa, C trả lời ngay** (Item anchor) | `Phối áo sơ mi trắng với quần gì đẹp?` | `PROCEED` | **`ASK`** (`target_gender`) | **`PROCEED`** (`COVERED_WITH_ALTERNATIVES`) |
| **S2** | **B hỏi thừa, C trả lời ngay** (Danh mục rộng) | `Tìm áo cho nam` | `PROCEED` | **`ASK` 3 câu** (`category_subtype`, `occasion`, `budget`) | **`PROCEED`** (`HARD_ANSWER_REQUIRED_COMPLETE`) |
| **S3** | **B bỏ sót, C bắt lỗi tồn kho** (Hết màu) | `Phối áo sơ mi nam màu xanh mint với quần âu đi làm.` | `PROCEED` | **`PROCEED`** (đủ slot, tìm ra 0 mẫu khớp) | **`ASK` 1 câu** (`color` — `UNFULFILLABLE_COLOR_CONSTRAINT`) |
| **S4** | **B bỏ sót, C bắt lỗi ngân sách** (Bất khả thi) | `Phối đồ đi tiệc cho nam dưới 100k` | `PROCEED` | **`PROCEED`** (đủ slot, không có bộ khớp) | **`ASK` 1 câu** (`budget` — `UNFULFILLABLE_HARD_CONSTRAINT`) |
| **S5** | **B bỏ sót, C bắt phân vân màu** | `Phối đồ đi làm cho nam nhưng chưa biết chọn màu sáng hay màu tối?` | `PROCEED` | **`PROCEED`** (đủ slot, tự chọn màu) | **`ASK` 1 câu** (`color` — `AMBIGUOUS_COLOR_PREFERENCE`) |
| **S6** | **Khác biệt cả 3 Mode (Đa lượt Size)** | T1: `Tư vấn chọn size giúp mình với`<br>T2-A: Submit form (Nam / Áo / Khung Size L)<br>T2-B: `Mình là nam, muốn chọn size áo sơ mi` | T1: `PROCEED` (bảng size)<br>T2-B: `PROCEED` (bảng size) | T1: `ASK` **3 câu**<br>T2-A: `PROCEED` (Size L + cards)<br>T2-B: **`ASK` lượt 2** (`measurements`) | T1: `ASK` **2 câu**<br>T2-A: `PROCEED` (Size L + cards)<br>T2-B: **`PROCEED` fallback** (bảng size) |
| **S7** | **A vs (B & C): Thiếu thông tin chí mạng** | T1: `Tư vấn giúp mình một bộ đồ đi tiệc tối nay với.`<br>T2 (tuỳ chọn): `Nam nữ gì cũng được, bạn cứ gợi ý giúp mình` | T1: `PROCEED` (đa giới tính + giả định) | T1: **`ASK`** (`target_gender`)<br>T2: `PROCEED` (`DECLINED`) | T1: **`ASK`** (`target_gender`, `LOW_TOP_K_JACCARD`)<br>T2: `PROCEED` (`DECLINED`) |
| **S8** | **A vs (B & C): Mâu thuẫn trực diện** | `Phối áo sơ mi đen cho nam đi làm nhưng không thích màu đen.` | `PROCEED` (bỏ qua mâu thuẫn) | **`ASK`** (`color`) | **`ASK`** (`color`, `HARD_ASK_CONFLICTING_SLOTS`) |
| **S9** | **Nhánh Gray Zone LLM ở Mode C** | `Tìm quần đi làm` | `PROCEED` (đa giới tính) | **`ASK`** (`target_gender`, khóa tool) | **`GRAY_ZONE`** (`mode_c_llm` tự quyết) |
| **S10** | **Đối chứng (Đủ slot & Phi mua sắm)** | 10a: `Phối đồ đi làm cho nam phong cách minimalist.`<br>10b: `Phí ship của shop bao nhiêu và có được đổi size không?` | `PROCEED` | `PROCEED` | `PROCEED` |

---

## 4. Chi tiết từng kịch bản kiểm thử

### Nhóm 1 — Tách rõ Mode B (hỏi cứng nhắc) vs Mode C (không hỏi thừa)
> **Tổ hợp kỳ vọng:** `Mode A = PROCEED` | `Mode B = ASK` | `Mode C = PROCEED`

#### S1 — Phối đồ có sẵn item anchor nhưng thiếu giới tính
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Prompt:**
  ```text
  Phối áo sơ mi trắng với quần gì đẹp?
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`, `MODE_A_STRICT_ZERO`). Gọi `suggest_outfit`, trả combo gợi ý ngay kèm giả định.
  - **Mode B:** `ASK` (`mode_b_required_check`, `MISSING_REQUIRED_SLOTS`). Khóa tool (`tools_available: []`), hiện form hỏi `target_gender` (Nam / Nữ / Unisex).
  - **Mode C:** `PROCEED` (`mode_c_counterfactual`, `COUNTERFACTUAL_PROCEED`, `COVERED_WITH_ALTERNATIVES`). Không hỏi lại vì đã có món đồ neo (`áo sơ mi trắng`) và có thể bao phủ gọn cả Nam & Nữ trong một lượt trả lời.
- **Checklist:**
  - [ ] Mode A trả combo ngay, `clarificationAsked = false`
  - [ ] Mode B hiện form hỏi `target_gender`, không gọi tool
  - [ ] Mode C trả combo ngay cho cả Nam & Nữ, `impact_signals.target_gender.can_cover_with_alternatives = true`

#### S2 — Tìm danh mục rộng đã có giới tính
- **Intent kỳ vọng:** `PRODUCT_SEARCH`
- **Prompt:**
  ```text
  Tìm áo cho nam
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). Gọi `search_products`, trả danh sách áo nam.
  - **Mode B:** `ASK` (`mode_b_required_check`, `BROAD_CATEGORY_REFINEMENT`). Hiện form hỏi **3 câu**: `category_subtype` (Sơ mi / Thun-Polo / Khoác-Blazer), `occasion`, `budget`.
  - **Mode C:** `PROCEED` (`mode_c_hard_answer`, `HARD_ANSWER_REQUIRED_COMPLETE`). Đã đủ `target_gender=MALE` và `category=tops` nên tìm luôn, không hỏi thêm 3 câu.
- **Checklist:**
  - [ ] Mode A trả product cards áo nam ngay
  - [ ] Mode B hiện form đủ 3 câu (`category_subtype`, `occasion`, `budget`)
  - [ ] Mode C trả product cards áo nam ngay, `clarificationAsked = false`

---

### Nhóm 2 — Tách rõ Mode C (thông minh theo tồn kho & KG) vượt trội Mode B
> **Tổ hợp kỳ vọng:** `Mode A = PROCEED` | `Mode B = PROCEED` | `Mode C = ASK`
>
> *Đây là nhóm kịch bản quan trọng nhất chứng minh Mode C không chỉ "hỏi ít hơn Mode B" mà còn phát hiện các ràng buộc bất khả thi / phân vân từ dữ liệu thật mà Mode B bỏ sót.*

#### S3 — Yêu cầu màu hết hàng / không tồn tại trong catalog
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Prompt:**
  ```text
  Phối áo sơ mi nam màu xanh mint với quần âu đi làm.
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). Không hỏi, tìm trực tiếp (báo hết màu hoặc trả fallback).
  - **Mode B:** `PROCEED` (`mode_b_required_check`, `REQUIRED_SLOTS_COMPLETE`). Do đã có `target_gender=MALE` và `top_category=shirt`, Mode B không hỏi và lao đi gọi tool.
  - **Mode C:** `ASK` (`mode_c_counterfactual`, `UNFULFILLABLE_COLOR_CONSTRAINT`). Counterfactual probe catalog phát hiện 0 áo sơ mi nam màu `xanh_mint` → dừng lại hỏi **1 câu** (`color`) để khách chọn tông màu thay thế đang có hàng.
- **Turn 2 mở rộng (cho Mode C):** Chọn hoặc nhập `Màu trắng` → Mode C chuyển sang `PROCEED` và trả các combo áo sơ mi trắng + quần âu nam.
- **Checklist:**
  - [ ] Mode A & Mode B đều `clarificationAsked = false`
  - [ ] Mode C `clarificationAsked = true`, `asked_slots = ["color"]`, `reason_codes = ["UNFULFILLABLE_COLOR_CONSTRAINT"]`
  - [ ] Turn 2 Mode C sau khi chọn `Màu trắng` trả combo hợp lệ kèm product cards

#### S4 — Ngân sách bất khả thi với catalog hiện tại
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Prompt:**
  ```text
  Phối đồ đi tiệc cho nam dưới 100k
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`).
  - **Mode B:** `PROCEED` (`mode_b_required_check`, `REQUIRED_SLOTS_COMPLETE`). Đã đủ `target_gender=MALE` và `occasion=party`.
  - **Mode C:** `ASK` (`mode_c_counterfactual`, `UNFULFILLABLE_HARD_CONSTRAINT`). Probe thấy 0 bộ đồ khả thi dưới 100k → hỏi **1 câu** (`budget`) để điều chỉnh ngân sách.
- **Checklist:**
  - [ ] Mode A & Mode B đều `clarificationAsked = false`
  - [ ] Mode C `clarificationAsked = true`, `asked_slots = ["budget"]`, `reason_codes = ["UNFULFILLABLE_HARD_CONSTRAINT"]`

#### S5 — Khách chủ động phân vân chọn màu sắc
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Prompt:**
  ```text
  Phối đồ đi làm cho nam nhưng chưa biết chọn màu sáng hay màu tối?
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`).
  - **Mode B:** `PROCEED` (`mode_b_required_check`, `REQUIRED_SLOTS_COMPLETE`). Bỏ qua trạng thái `color = AMBIGUOUS` vì chỉ kiểm tra Required Slots.
  - **Mode C:** `ASK` (`mode_c_counterfactual`, `AMBIGUOUS_COLOR_PREFERENCE`). Nhận diện `color = AMBIGUOUS`, chạy tone counterfactual → hỏi **1 câu** (`color`) chọn tông màu ưu tiên (Trung tính / Tông đất / Điểm nhấn).
- **Checklist:**
  - [ ] Mode A & Mode B đều `clarificationAsked = false`
  - [ ] Mode C `clarificationAsked = true`, `asked_slots = ["color"]`, `reason_codes = ["AMBIGUOUS_COLOR_PREFERENCE"]`

---

### Nhóm 3 — Hội thoại đa lượt: Cả 3 Mode thể hiện 3 hành vi tách biệt hoàn toàn

#### S6 — Tư vấn chọn size thiếu toàn bộ thông tin (2 nhánh Turn 2)
- **Intent kỳ vọng:** `SIZE_ADVICE` (xuyên suốt cả Turn 1 và Turn 2)
- **Turn 1 Prompt:**
  ```text
  Tư vấn chọn size giúp mình với
  ```
- **Kỳ vọng Turn 1:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). **0 câu hỏi**, trả ngay **bảng size chuẩn** để khách tự đối chiếu.
  - **Mode B:** `ASK` (`mode_b_required_check`). Hiện form **3 câu**: `target_gender`, `category`, `measurements`.
  - **Mode C:** `ASK` (`mode_c_hard_ask`, `HARD_ASK_SIZE_MISSING_MEASUREMENTS`). Hiện form **tối đa 2 câu**, ưu tiên số đo trước: `measurements`, `target_gender`.

- **Nhánh Turn 2-A (Test submit form đầy đủ — chạy cho Mode B & Mode C):**
  - Tích chọn trên UI (hoặc gửi):
    ```text
    Mình chọn:
    1. Nam
    2. Áo (Sơ mi / Thun / Polo / Khoác)
    3. Cân đối / Đầy đặn – Khung Size L (63–72kg | 1m68–1m75)
    ```
  - **Kỳ vọng Turn 2-A:**
    - `turn_intent` giữ nguyên **`SIZE_ADVICE`** (KHÔNG bị lật sang `BODY_FIT_ADVICE`).
    - `budget = UNKNOWN` (KHÔNG ăn nhầm `63–72kg` thành `63k–72k`), `body_concern = NOT_APPLICABLE`.
    - `measurements = { height_cm: 171.5, weight_kg: 67.5 }`.
    - Node `prefetch` chạy tuần tự **`recommend_size`** → **`search_products(size="L", target_gender="MALE")`** → trả **tư vấn Size L** (size phụ **M nếu muốn mặc ôm gọn hơn**, KHÔNG ghi "rộng hơn") **kèm card sản phẩm Áo Nam size L** (`products` không rỗng).

- **Nhánh Turn 2-B (Test trả lời thiếu số đo — reset session, chạy lại Turn 1 rồi gửi Turn 2-B bằng text):**
  ```text
  Mình là nam, muốn chọn size áo sơ mi
  ```
  - **Kỳ vọng Turn 2-B:**
    - **Mode A:** `PROCEED` — trả bảng size chuẩn Áo Nam.
    - **Mode B:** **`ASK` tiếp lượt 2** (`clarification_turns: 1 < MAX_MODE_B_TURNS: 2`), hỏi riêng 1 câu `measurements`.
    - **Mode C:** **`PROCEED` fallback** (`fallback`, `MODE_C_TURN_CAP_FALLBACK` do đã chạm trần `MAX_MODE_C_TURNS = 1`) → ngừng hỏi, trả ngay **bảng size chuẩn Áo Nam** để khách tự đối chiếu.
- **Checklist:**
  - [ ] T1: Mode A trả bảng size (0 câu); Mode B hỏi 3 câu; Mode C hỏi 2 câu (`measurements` đầu tiên)
  - [ ] T2-A (Mode B): `intent = SIZE_ADVICE`, không có `budget` ảo, lời khuyên ghi M là "ôm gọn hơn", có `products` size L hiển thị dưới chat
  - [ ] T2-B: Mode B hỏi tiếp lượt 2 (`measurements`); Mode C dừng hỏi và trả bảng size chuẩn

---

### Nhóm 4 — Tách rõ Mode A vs (Mode B & Mode C)
> **Tổ hợp kỳ vọng:** `Mode A = PROCEED` | `Mode B = ASK` | `Mode C = ASK`

#### S7 — Phối đồ mở cho dịp phân nhánh mạnh (thiếu giới tính, không có anchor)
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Turn 1 Prompt:**
  ```text
  Tư vấn giúp mình một bộ đồ đi tiệc tối nay với.
  ```
- **Hành vi kỳ vọng Turn 1:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). Trả ngay gợi ý cân bằng đa giới tính (Nữ / Nam) kèm câu giả định.
  - **Mode B:** `ASK` (`mode_b_required_check`, `MISSING_REQUIRED_SLOTS`) → hỏi `target_gender`.
  - **Mode C:** `ASK` (`mode_c_counterfactual`, `COUNTERFACTUAL_HIGH_IMPACT_ASK`, `LOW_TOP_K_JACCARD`) → hỏi `target_gender` (vì không có item anchor nên đồ dự tiệc Nam và Nữ khác biệt hoàn toàn).
- **Turn 2 Prompt (tuỳ chọn — test khách từ chối cung cấp giới tính cho Mode B & C):**
  ```text
  Nam nữ gì cũng được, bạn cứ gợi ý giúp mình
  ```
  - **Kỳ vọng Turn 2:** Cả Mode B và Mode C nhận trạng thái `DECLINED` cho `target_gender` (hoặc resolve `UNISEX`), lập tức `PROCEED` gợi ý bộ đồ mà **không hỏi lặp lại**.
- **Checklist:**
  - [ ] T1 Mode A trả combo ngay, `clarificationAsked = false`
  - [ ] T1 Mode B hỏi `target_gender` (`mode_b_required_check`)
  - [ ] T1 Mode C hỏi `target_gender` (`mode_c_counterfactual`, có `impact_signals.target_gender`)
  - [ ] T2 sau khi khách từ chối, cả B & C trả gợi ý ngay không lặp form

#### S8 — Yêu cầu có mâu thuẫn trực diện (`CONFLICTING`)
- **Intent kỳ vọng:** `OUTFIT_ADVICE`
- **Prompt:**
  ```text
  Phối áo sơ mi đen cho nam đi làm nhưng không thích màu đen.
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). Bỏ qua mâu thuẫn, gợi ý luôn.
  - **Mode B:** `ASK` (`mode_b_required_check`, `MISSING_REQUIRED_SLOTS`) → hỏi xác nhận lại `color`.
  - **Mode C:** `ASK` (`mode_c_hard_ask`, `HARD_ASK_CONFLICTING_SLOTS`) → kích hoạt Hard Ask Guard hỏi xác nhận lại `color` (dạng câu hỏi xử lý mâu thuẫn).
- **Checklist:**
  - [ ] Mode A trả lời ngay (`clarificationAsked = false`)
  - [ ] Mode B & Mode C đều hiện form xác nhận lại tông màu (`asked_slots = ["color"]`)
  - [ ] Mode C có `reason_codes = ["HARD_ASK_CONFLICTING_SLOTS"]`

---

### Nhóm 5 — Nhánh Gray Zone LLM (Mode C) & Kịch bản Đối chứng (Control)

#### S9 — Tìm sản phẩm đơn lẻ thiếu giới tính (Vùng Xám LLM)
- **Intent kỳ vọng:** `PRODUCT_SEARCH`
- **Prompt:**
  ```text
  Tìm quần đi làm
  ```
- **Hành vi kỳ vọng:**
  - **Mode A:** `PROCEED` (`mode_a_immediate`). Tự chạy Balanced Multi-Gender Search (chia nhóm Nữ / Nam / Unisex) + trả product cards.
  - **Mode B:** `ASK` (`mode_b_required_check`). Khóa tool (`gate_search = true`), ép hỏi `target_gender`.
  - **Mode C:** `GRAY_ZONE` (`mode_c_llm`, `GRAY_ZONE_LLM_EVAL`). Không khóa tool (`search_products` sẵn sàng) và mở đánh giá 4 tiêu chí để LLM tự quyết nên hỏi `target_gender` hay trả lời đa phương án (nếu hỏi, tối đa 2 câu, không hỏi lại `occasion`).
- **Checklist:**
  - [ ] Mode A trả sản phẩm đa giới tính ngay
  - [ ] Mode B khóa tool (`tools_available = []`), hỏi `target_gender`
  - [ ] Mode C có `decision_source = "mode_c_llm"` và không hỏi lại `occasion` (đã biết = `office`)

#### S10 — Đối chứng (Sanity Check): Đủ thông tin & Phi mua sắm
> **Tổ hợp kỳ vọng:** `Mode A = PROCEED` | `Mode B = PROCEED` | `Mode C = PROCEED`

- **S10a — Yêu cầu phối đồ đã đủ thông tin:**
  ```text
  Phối đồ đi làm cho nam phong cách minimalist.
  ```
  - **Kỳ vọng:** Cả 3 mode đều `PROCEED` ngay, gọi `suggest_outfit` trả combo công sở nam minimalist kèm product cards. Mode B không hỏi thừa; Mode C tự chọn bảng màu minimalist mà **không hỏi màu**.
- **S10b — Câu hỏi chính sách (Phi mua sắm):**
  ```text
  Phí ship của shop bao nhiêu và có được đổi size không?
  ```
  - **Kỳ vọng:** Cả 3 mode đều nhận diện `NON_SHOPPING` (`non_shopping_bypass`), gọi `search_policy` trả lời trực tiếp có trích nguồn `[tên_tệp: tiêu_đề_mục]`, tuyệt đối không hiện form làm rõ.
- **Checklist:**
  - [ ] S10a: Cả 3 mode đều `clarificationAsked = false`, có `outfits` & `products`
  - [ ] S10b: Cả 3 mode đều `intent = NON_SHOPPING`, `tools_used` chứa `search_policy`, có trích nguồn

---

## 5. Bảng ghi nhận kết quả kiểm thử (Test Log Template)

- **Ngày test:** `YYYY-MM-DD`
- **Provider / Model:** `ollama / qwen3.5:2b` *(hoặc cập nhật model thực tế)*
- **Commit / Branch:** `...`

| Kịch bản | Mode A | Mode B | Mode C | Ghi chú / Bug phát hiện |
|---|---|---|---|---|
| **S1** (Anchor outfit) | ⬜ | ⬜ | ⬜ | |
| **S2** (Broad category) | ⬜ | ⬜ | ⬜ | |
| **S3** (Out-of-stock color) | ⬜ | ⬜ | ⬜ | |
| **S4** (Unfulfillable budget) | ⬜ | ⬜ | ⬜ | |
| **S5** (Ambiguous color) | ⬜ | ⬜ | ⬜ | |
| **S6** (Size multi-turn A/B) | ⬜ | ⬜ | ⬜ | |
| **S7** (Open party + decline) | ⬜ | ⬜ | ⬜ | |
| **S8** (Conflicting color) | ⬜ | ⬜ | ⬜ | |
| **S9** (Gray zone search) | ⬜ | ⬜ | ⬜ | |
| **S10a / S10b** (Control) | ⬜ | ⬜ | ⬜ | |

