"""Pure functional clarification policy and prompt generation module."""

from dataclasses import dataclass

from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.services.session_manager import ClarifyPhase

MAX_GATED_TURNS = 1

PROMPT_A = """\
CHẾ ĐỘ TRẢ LỜI NGAY (IMMEDIATE_ANSWER):
- Khi khách hỏi tìm đồ hoặc gợi ý trang phục: BẮT BUỘC gọi ngay tool `search_products`
  với thông tin hiện có và giới thiệu ngay 2-5 sản phẩm kèm mã [ID].
- Nếu thiếu thông tin như giới tính, loại đồ hay ngân sách: TỰ ĐƯA RA GIẢ ĐỊNH HỢP LÝ
  và nêu ngắn gọn giả định (1 câu ngắn) trong câu trả lời.
- KHI KẾT QUẢ `search_products` CHIA NHÓM ĐA GIỚI TÍNH (`GỢI Ý CHO NỮ`,
  `GỢI Ý CHO NAM`, `GỢI Ý UNISEX`): BẮT BUỘC nêu rõ bạn chưa chọn giới tính và
  giới thiệu ĐẦY ĐỦ sản phẩm của TẤT CẢ các nhóm có trong kết quả (kèm `[ID]` và giá),
  tuyệt đối không chỉ chọn riêng 1 giới tính.
- TUYỆT ĐỐI KHÔNG đặt câu hỏi làm rõ về giới tính, dịp mặc, gu, màu sắc hay ngân sách.
- Chỉ được hỏi lại khi khách hỏi trực tiếp về SIZE mà thiếu số đo bắt buộc cho tool
  `recommend_size`.
- Luôn đặt `needs_clarification = False` và `clarification_form = None` (trừ trường
  hợp hỏi size thiếu số đo bắt buộc).
"""

PROMPT_B_INTERVIEW = """\
CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn tìm hiểu):
- Khách đang có nhu cầu tìm/tư vấn trang phục hoặc chọn size: CHƯA gợi ý sản phẩm ngay.
  Đặt needs_clarification=True.
- Hệ thống đã tạo sẵn bảng câu hỏi trắc nghiệm ưu tiên (P0 -> P1 -> P2) bên dưới
  lời dẫn của bạn.
- Phần reply CHỈ ĐƯỢC viết đúng 1 câu chào và dẫn nhập ngắn gọn (15-30 từ) khớp với
  ngữ cảnh khách vừa nêu để mời khách tích chọn vào bảng bên dưới.
- TUYỆT ĐỐI KHÔNG tự liệt kê gạch đầu dòng câu hỏi và KHÔNG đòi chiều cao/cân nặng/số đo
  3 vòng (trừ khi khách hỏi size hoặc vóc dáng).
"""

PROMPT_B_OPEN = """\
CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn chốt đồ):
- Khách đã cung cấp thông tin làm rõ. BẮT BUỘC dùng đầy đủ thông tin trong phần
  TÓM TẮT NGỮ CẢNH (đặc biệt `target_gender`, `category`, `budget`) để gọi tool
  `search_products` (hoặc `recommend_size` nếu hỏi size) và giới thiệu 2-4 sản phẩm
  phù hợp kèm mã [ID] và giá tiền.
- TUYỆT ĐỐI KHÔNG hỏi lại hay lặp lại phỏng vấn. Đặt needs_clarification=False.
- LƯU Ý VỀ GIÁ: Đơn vị là VNĐ. Nếu nói 300k, 400k thì min_price=300000,
  max_price=400000.
"""

PROMPT_C = """\
CHẾ ĐỘ TỰ CÂN NHẮC (CONDITIONAL_CLARIFICATION):
Hãy đọc TÓM TẮT NGỮ CẢNH (P0 -> P1 -> P2) và tự quyết định theo 3 tiêu chí:
(1) Ảnh hưởng đến kết quả gợi ý, (2) Mức độ không chắc chắn, (3) Chi phí làm rõ.
- KHI NÊN HỎI LẠI (needs_clarification=True):
  * Thiếu thông tin P0 quan trọng làm kết quả dễ sai lệch (đặc biệt thiếu Giới tính
    Nam/Nữ khi hỏi đồ dùng chung như áo, quần, đồ đi làm/đi chơi; hoặc yêu cầu quá
    chung chung chưa rõ dịp/danh mục; hoặc khách hỏi size nhưng thiếu số đo).
  * Chỉ hỏi 1-2 câu có tác động lớn nhất qua `clarification_form` (mỗi câu 2-4 lựa
    chọn viết theo góc nhìn khách hàng) và viết đúng 1 câu dẫn ngắn ở `reply`.
  * TUYỆT ĐỐI KHÔNG hỏi lại thông tin đã có trong TÓM TẮT NGỮ CẢNH, và KHÔNG hỏi
    số đo cơ thể (P3) khi khách chỉ tìm sản phẩm.
- KHI NÊN GỢI Ý NGAY (needs_clarification=False):
  * Các thông tin P0 cốt lõi (giới tính hoặc loại đồ đặc thù + ngữ cảnh tìm kiếm) đã
    đủ rõ để lọc catalog đúng hướng: gọi ngay `search_products` với đúng
    `target_gender` và tiêu chí đã biết, giới thiệu 2-4 sản phẩm kèm mã [ID].
"""

PROMPT_NON_SHOPPING = """\
YÊU CẦU PHI MUA SẮM (CHÍNH SÁCH / VOUCHER / CHI TIẾT MÃ / XÃ GIAO):
- Khách không đang nhờ tìm kiếm hay chọn lọc danh sách sản phẩm chung.
- Hãy trả lời trực tiếp bằng công cụ tương ứng (search_policy, check_voucher,
  get_product_detail) hoặc đáp lại lịch sự ngắn gọn.
- Đặt needs_clarification=False.
"""


@dataclass(frozen=True)
class TurnPolicy:
    """Policy resolved for the current turn."""

    gate_search: bool
    prompt_block: str


def _normalize_mode(mode: ClarificationMode | str) -> ClarificationMode:
    if isinstance(mode, str):
        mode_clean = mode.strip().upper()
        if mode_clean in (
            ClarificationMode.CONDITIONAL_CLARIFICATION,
            "MINIMAL_CLARIFICATION",
        ):
            return ClarificationMode.CONDITIONAL_CLARIFICATION
        if mode_clean in ClarificationMode.__members__:
            return ClarificationMode[mode_clean]
        return ClarificationMode.CONDITIONAL_CLARIFICATION
    if mode == ClarificationMode.MINIMAL_CLARIFICATION:
        return ClarificationMode.CONDITIONAL_CLARIFICATION
    return mode


def resolve_turn_policy(
    mode: ClarificationMode | str,
    phase: ClarifyPhase,
    is_shopping_query: bool = True,
) -> TurnPolicy:
    """Resolve tool gating and prompt instructions deterministically."""
    norm_mode = _normalize_mode(mode)

    if norm_mode == ClarificationMode.ALWAYS_CLARIFY:
        if phase == ClarifyPhase.INTERVIEW:
            if is_shopping_query:
                return TurnPolicy(gate_search=True, prompt_block=PROMPT_B_INTERVIEW)
            return TurnPolicy(gate_search=False, prompt_block=PROMPT_NON_SHOPPING)
        return TurnPolicy(gate_search=False, prompt_block=PROMPT_B_OPEN)

    if norm_mode == ClarificationMode.IMMEDIATE_ANSWER:
        return TurnPolicy(gate_search=False, prompt_block=PROMPT_A)

    return TurnPolicy(gate_search=False, prompt_block=PROMPT_C)


def next_phase(
    mode: ClarificationMode | str,
    current_phase: ClarifyPhase,
    gated_turns: int,
    needs_clarification: bool,
    is_shopping_query: bool = True,
) -> ClarifyPhase:
    """Transition phase one-way from INTERVIEW to OPEN on shopping turns."""
    norm_mode = _normalize_mode(mode)

    if (
        norm_mode != ClarificationMode.ALWAYS_CLARIFY
        or current_phase == ClarifyPhase.OPEN
        or not is_shopping_query
    ):
        return current_phase
    if needs_clarification or gated_turns >= MAX_GATED_TURNS:
        return ClarifyPhase.OPEN
    return ClarifyPhase.INTERVIEW
