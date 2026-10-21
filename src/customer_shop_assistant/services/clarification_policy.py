"""Pure functional clarification policy and prompt generation module."""

from dataclasses import dataclass

from customer_shop_assistant.schemas.chat import ClarificationMode
from customer_shop_assistant.services.session_manager import ClarifyPhase

MAX_GATED_TURNS = 1

PROMPT_A = """\
CHẾ ĐỘ TRẢ LỜI NGAY (IMMEDIATE_ANSWER):
- Khi khách cần gợi ý sản phẩm, hãy tìm ngay bằng thông tin hiện có và nêu rõ giả định.
- Tuyệt đối KHÔNG hỏi về gu, dịp mặc hay phong cách.
- Chỉ hỏi lại khi thiếu tham số bắt buộc của công cụ (ví dụ số đo khi tính size).
- Đặt needs_clarification=False trừ trường hợp thiếu tham số bắt buộc đó.
"""

PROMPT_B_INTERVIEW = """\
CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn tìm hiểu):
- Nếu khách yêu cầu tìm hoặc gợi ý sản phẩm: CHƯA gợi ý ngay. Hãy hỏi MỘT câu về \
dịp mặc, phong cách, vóc dáng hoặc ngân sách, kèm 2-4 quick_replies và đặt \
needs_clarification=True.
- Nếu khách hỏi việc khác (chính sách, voucher, bảng size chuẩn, xem chi tiết một \
mẫu cụ thể): hãy trả lời bình thường bằng công cụ tương ứng.
"""

PROMPT_B_OPEN = """\
CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn chốt đồ):
- Khách đã cung cấp thông tin. BẮT BUỘC gọi tool search_products để tìm và giới \
thiệu 2-4 sản phẩm phù hợp.
- BẮT BUỘC giới thiệu cụ thể các mẫu tìm được kèm mã [ID] và giá tiền.
- TUYỆT ĐỐI KHÔNG hỏi thêm hay lặp lại phỏng vấn. Đặt needs_clarification=False.
- LƯU Ý VỀ GIÁ: Đơn vị là VNĐ. Nếu nói 300k, 400k thì min_price=300000, \
max_price=400000.
"""

PROMPT_C = """\
CHẾ ĐỘ TỰ CÂN NHẮC (CONDITIONAL_CLARIFICATION):
- Nếu khách nêu loại đồ kèm ít nhất một ràng buộc (dịp, ngân sách, giới tính, \
màu sắc): tìm và gợi ý ngay.
- Nếu yêu cầu quá chung (ví dụ 'gợi ý đồ', 'tìm áo') hoặc hỏi size mà thiếu số đo: \
hỏi MỘT câu ngắn kèm quick_replies và đặt needs_clarification=True.
- Chỉ hỏi khi câu trả lời thực sự làm kết quả tốt hơn.
"""


@dataclass(frozen=True)
class TurnPolicy:
    """Policy resolved for the current turn."""

    gate_search: bool
    prompt_block: str


def resolve_turn_policy(
    mode: ClarificationMode | str, phase: ClarifyPhase
) -> TurnPolicy:
    """Resolve tool gating and prompt instructions deterministically."""
    if isinstance(mode, str):
        mode_clean = mode.strip().upper()
        if mode_clean in (
            ClarificationMode.CONDITIONAL_CLARIFICATION,
            "MINIMAL_CLARIFICATION",
        ):
            mode = ClarificationMode.CONDITIONAL_CLARIFICATION
        elif mode_clean in ClarificationMode.__members__:
            mode = ClarificationMode[mode_clean]
        else:
            mode = ClarificationMode.CONDITIONAL_CLARIFICATION

    if mode == ClarificationMode.ALWAYS_CLARIFY:
        if phase == ClarifyPhase.INTERVIEW:
            return TurnPolicy(gate_search=True, prompt_block=PROMPT_B_INTERVIEW)
        return TurnPolicy(gate_search=False, prompt_block=PROMPT_B_OPEN)
    if mode == ClarificationMode.IMMEDIATE_ANSWER:
        return TurnPolicy(gate_search=False, prompt_block=PROMPT_A)
    return TurnPolicy(gate_search=False, prompt_block=PROMPT_C)


def next_phase(
    mode: ClarificationMode | str,
    current_phase: ClarifyPhase,
    gated_turns: int,
    needs_clarification: bool,
) -> ClarifyPhase:
    """Transition clarification state machine phase one-way from INTERVIEW to OPEN."""
    if isinstance(mode, str):
        mode_clean = mode.strip().upper()
        if mode_clean in ClarificationMode.__members__:
            mode = ClarificationMode[mode_clean]

    if mode != ClarificationMode.ALWAYS_CLARIFY or current_phase == ClarifyPhase.OPEN:
        return current_phase
    if needs_clarification or gated_turns >= MAX_GATED_TURNS:
        return ClarifyPhase.OPEN
    return ClarifyPhase.INTERVIEW
