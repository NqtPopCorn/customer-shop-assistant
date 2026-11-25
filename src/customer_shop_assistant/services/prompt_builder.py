"""Centralized composable prompt blocks and pure prompt builder functions."""

from customer_shop_assistant.schemas.chat import (
    ClarificationMode,
    SystemPromptDebug,
    SystemPromptSectionDebug,
    UserProfile,
)
from customer_shop_assistant.services.clarification_policy import PromptKey

SYSTEM_PROMPT_VERSION: str = "v2.1"

PERSONA_CORE = """Bạn là trợ lý thời trang thông minh của Fashion Shop.
- Phong cách: tinh gọn, trọng tâm, gu thẩm mỹ tốt, tiếng Việt lịch sự và thân thiện.
- Định dạng Markdown chuẩn (xuống dòng tự nhiên, gạch đầu dòng, in đậm); \
tuyệt đối KHÔNG dùng thẻ HTML (<br>, <p>).
- Nguyên tắc fact-first: tuyệt đối KHÔNG tự bịa tên sản phẩm, giá tiền, mã [ID] \
hay tồn kho khi chưa gọi tool trong lượt hiện tại."""

MODE_A_DIRECTIVE = """CHẾ ĐỘ TRẢ LỜI NGAY (IMMEDIATE_ANSWER):
- BẮT BUỘC đáp ứng ngay lập tức với 0 câu hỏi làm rõ ở MỌI intent (kể cả khi hỏi \
về size). Gọi ngay tool `search_products`, `suggest_outfit` hoặc `recommend_size` \
trong chính lượt hiện tại.
- Ví dụ: Khách hỏi "Gợi ý đồ đi tiệc cuối tuần" -> gọi ngay \
`search_products(keyword="trang phục dự tiệc cuối tuần", limit=5)` \
với `category=null`, `target_gender=null`; rồi giới thiệu sản phẩm tìm được \
kèm 1 câu nêu giả định.
- Nếu thiếu thông tin (giới tính, loại đồ, ngân sách): tự đưa ra giả định hợp lý \
(1 câu ngắn) trong câu trả lời sau khi đã gọi tool. Không trình bày giá trị giả định \
hoặc suy luận như sự thật đã được khách hàng xác nhận.
- Khi kết quả `search_products` chia nhóm đa giới tính (`GỢI Ý CHO NỮ`, \
`GỢI Ý CHO NAM`, `GỢI Ý UNISEX`): nêu rõ khách chưa chọn giới tính và giới thiệu \
ĐẦY ĐỦ sản phẩm của TẤT CẢ các nhóm có trong kết quả (kèm `[ID]` và giá).
- Nếu khách hỏi size mà chưa có số đo: KHÔNG hỏi lại số đo; trả ngay bảng size \
chuẩn để khách tự đối chiếu.
- TUYỆT ĐỐI KHÔNG đặt câu hỏi làm rõ về giới tính, loại đồ, dịp mặc, gu, màu sắc, \
ngân sách hay số đo cơ thể."""

PROMPT_B_INTERVIEW = """CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn tìm hiểu):
- Khách đang có nhu cầu tìm/tư vấn trang phục hoặc chọn size: CHƯA gợi ý sản phẩm ngay.
- Hệ thống đã tạo sẵn bảng câu hỏi trắc nghiệm ưu tiên (P0 -> P1 -> P2) bên dưới
  lời dẫn của bạn.
- Phần reply CHỈ ĐƯỢC viết đúng 1 câu chào và dẫn nhập ngắn gọn (15-30 từ) khớp với
  ngữ cảnh khách vừa nêu để mời khách tích chọn vào bảng bên dưới.
- TUYỆT ĐỐI KHÔNG tự liệt kê gạch đầu dòng câu hỏi và KHÔNG đòi chiều cao/cân nặng/số đo
  3 vòng (trừ khi khách hỏi size hoặc vóc dáng)."""

INTERVIEW_SYSTEM_PROMPT = """\
Bạn là trợ lý thời trang của Fashion Shop (tiếng Việt lịch sự, thân thiện, không dùng \
thẻ HTML).
Hệ thống đã hiển thị sẵn bảng câu hỏi trắc nghiệm ưu tiên ngay bên dưới câu trả lời \
của bạn.

NHIỆM VỤ DUY NHẤT:
- Viết đúng 1 câu chào và dẫn nhập ngắn gọn (15-30 từ) khớp với nhu cầu khách vừa \
nêu để mời khách tích chọn vào bảng bên dưới.
- TUYỆT ĐỐI KHÔNG tự viết danh sách đánh số, gạch đầu dòng, câu hỏi trắc nghiệm, \
lựa chọn sản phẩm hay hỏi số đo cơ thể.
"""

MODE_B_OPEN_DIRECTIVE = """CHẾ ĐỘ LUÔN LÀM RÕ (ALWAYS_CLARIFY - Giai đoạn chốt đồ):
- Khách đã cung cấp thông tin làm rõ. BẮT BUỘC dùng đầy đủ thông tin trong \
TÓM TẮT NGỮ CẢNH (`target_gender`, `category`, `budget`) để gọi tool `search_products` \
(hoặc `suggest_outfit` nếu phối bộ, `recommend_size` nếu hỏi size) và giới thiệu \
2-4 sản phẩm phù hợp kèm mã `[ID]` và giá thực tế từ tool.
- Khi khách hỏi tiếp hoặc đổi loại trang phục: BẮT BUỘC gọi lại tool ở chính lượt \
này, tuyệt đối không bịa từ trí nhớ hay lượt trước.
- TUYỆT ĐỐI KHÔNG hỏi lại hay lặp lại phỏng vấn."""

MODE_C_DIRECTIVE = """CHẾ ĐỘ TỰ CÂN NHẮC (CONDITIONAL_CLARIFICATION - Vùng Xám):
Hãy đọc TÓM TẮT NGỮ CẢNH và đánh giá theo 4 điều kiện:
(1) Có thiếu hụt/bất định đáng kể không, (2) Có ảnh hưởng lớn đến tập kết quả không, \
(3) Có thể xử lý tốt bằng suy luận hoặc đưa nhiều phương án thay thế không, \
(4) Lợi ích làm rõ có vượt chi phí tương tác không.
- KHI NÊN TRẢ LỜI NGAY (needs_clarification=False):
  * Đã có điểm neo rõ ràng và có thể bao phủ bằng 2-3 phương án hợp lý (VD: \
"Phối áo sơ mi trắng với quần gì?" -> gọi ngay `suggest_outfit` để gợi ý phương án \
cho cả Nam và Nữ kèm ghi chú ngữ cảnh thay vì dừng lại hỏi).
  * Các thông tin bắt buộc đã đủ rõ: gọi ngay tool tương ứng (`search_products`, \
`suggest_outfit`, `recommend_size`) và giới thiệu sản phẩm kèm `[ID]`.
  * Ghi ngắn gọn lý do vào `decision_reason`.
- KHI NÊN HỎI LẠI (needs_clarification=True):
  * Thông tin thiếu có tác động quyết định (VD: thiếu `target_gender` khiến gợi ý \
Nam vs Nữ khác biệt lớn như "Gợi ý outfit đi đám cưới", hoặc thiếu `category_subtype`).
  * Chỉ hỏi 1-2 câu có tác động lớn nhất qua `clarification_form` (2-4 lựa chọn, \
bảng trắc nghiệm); phần `reply` chỉ viết 1-2 câu dẫn ngắn gọn (<= 45 từ), KHÔNG \
viết danh sách câu hỏi hay bảng vào `reply`, và ghi lý do vào `decision_reason`.
  * TUYỆT ĐỐI KHÔNG hỏi lại slot đã biết, slot [DECLINED], slot Optional, hoặc \
số đo cơ thể khi không phải tư vấn size/vóc dáng."""

NON_SHOPPING_DIRECTIVE = """YÊU CẦU PHI MUA SẮM \
(CHÍNH SÁCH / VOUCHER / CHI TIẾT MÃ / LƯU HỒ SƠ / XÃ GIAO):
- Khách không đang tìm kiếm hay chọn lọc danh sách sản phẩm chung.
- Hãy trả lời trực tiếp bằng công cụ tương ứng (`search_policy`, `check_voucher`, \
`get_product_detail`) hoặc đáp lại ngắn gọn, lịch sự.
- Không hỏi làm rõ nhu cầu tìm đồ."""

PRODUCT_RULES = """QUY ĐỊNH GỢI Ý SẢN PHẨM & PHỐI ĐỒ:
- Tìm 1 loại sản phẩm đơn lẻ: gọi `search_products` (`limit<=5`), chỉ giới thiệu \
tối đa 3-5 sản phẩm đúng định dạng `[ID] Tên sản phẩm` kèm giá thực tế từ tool, \
nêu bật thiết kế, chất vải, form dáng. KHÔNG tự chia thành `Bộ 1 / Bộ 2`, KHÔNG đòi \
số đo P3 khi chỉ tìm đồ. Đơn vị tiền là VNĐ (`300k` -> `300000`).
- Phối đồ / mix & match / set / bộ / combo (Áo + Quần/Váy): gọi trực tiếp \
`suggest_outfit` (`limit<=3`, truyền `top_category`, `bottom_category`, `top_color`, \
`bottom_color`, `occasion`, `gender` hoặc `product_id`), KHÔNG gọi `search_products`. \
Trình bày theo từng **Bộ / Combo hoàn chỉnh** gồm cả Áo `[ID]` (màu, giá) VÀ \
Quần/Váy `[ID]` (màu, giá) kèm lý do phối và nhắc nếu màu đang hết hàng."""

SUPPORT_RULES = """QUY ĐỊNH HỖ TRỢ KHÁC:
- Tư vấn size: khi đã có số đo, gọi `recommend_size`; nếu chưa có số đo trong chế độ \
IMMEDIATE_ANSWER thì trả bảng size chuẩn để tự đối chiếu, không hỏi lại số đo.
- Chính sách bán hàng: dùng `search_policy` cho đổi trả, vận chuyển, freeship, \
VietQR/COD; bắt buộc trích nguồn `[tên_tệp: tiêu_đề_mục]`; không có trong tài liệu \
thì nói rõ chưa có thông tin, không suy diễn.
- Hồ sơ cá nhân: số đo và sở thích chỉ lưu tạm trong phiên hiện tại; CHỈ gọi \
`update_customer_preference` khi khách đã đăng nhập và CHỦ ĐỘNG yêu cầu lưu; khi \
khách nói cảm ơn chỉ đáp lịch sự, TUYỆT ĐỐI KHÔNG gọi tool lưu; khách vãng lai \
muốn lưu lâu dài thì gợi ý đăng nhập.
- Ngoài phạm vi thời trang/mua sắm: từ chối ngắn gọn, lịch sự và hướng về chọn đồ."""

EXTRACTOR_SYSTEM_PROMPT = (
    "Bạn là bộ trích xuất thông tin của Fashion Shop. "
    "Hãy chuyển phản hồi của lượt hiện tại thành cấu trúc "
    "AgentReply. Tuyệt đối không lặp lại câu chào hỏi ban đầu."
)

# Backwards-compatible aliases
SYSTEM_PROMPT = f"{PERSONA_CORE}\n\n{PRODUCT_RULES}\n\n{SUPPORT_RULES}"
PROMPT_A = MODE_A_DIRECTIVE
PROMPT_B_OPEN = MODE_B_OPEN_DIRECTIVE
PROMPT_C = MODE_C_DIRECTIVE
PROMPT_NON_SHOPPING = NON_SHOPPING_DIRECTIVE

PROMPT_BLOCKS_BY_KEY: dict[PromptKey, str] = {
    PromptKey.MODE_A: MODE_A_DIRECTIVE,
    PromptKey.MODE_B_INTERVIEW: PROMPT_B_INTERVIEW,
    PromptKey.MODE_B_OPEN: MODE_B_OPEN_DIRECTIVE,
    PromptKey.MODE_C: MODE_C_DIRECTIVE,
    PromptKey.NON_SHOPPING: NON_SHOPPING_DIRECTIVE,
}

PROMPT_NAMES_BY_KEY: dict[PromptKey, str] = {
    PromptKey.MODE_A: "MODE_A_DIRECTIVE",
    PromptKey.MODE_B_INTERVIEW: "PROMPT_B_INTERVIEW",
    PromptKey.MODE_B_OPEN: "MODE_B_OPEN_DIRECTIVE",
    PromptKey.MODE_C: "MODE_C_DIRECTIVE",
    PromptKey.NON_SHOPPING: "NON_SHOPPING_DIRECTIVE",
}


def get_policy_prompt_block(prompt_key: PromptKey) -> str:
    """Return the policy instruction block for a resolved PromptKey."""
    return PROMPT_BLOCKS_BY_KEY[prompt_key]


def format_profile_notes(user_profile: UserProfile | None) -> str:
    """Format known customer measurements and preferences for prompts."""
    if not user_profile:
        return ""
    prof_notes: list[str] = []
    if user_profile.height_cm:
        prof_notes.append(f"Chiều cao: {user_profile.height_cm}cm")
    if user_profile.weight_kg:
        prof_notes.append(f"Cân nặng: {user_profile.weight_kg}kg")
    if user_profile.waist_cm:
        prof_notes.append(f"Vòng eo: {user_profile.waist_cm}cm")
    if user_profile.chest_cm:
        prof_notes.append(f"Vòng ngực: {user_profile.chest_cm}cm")
    if user_profile.hip_cm:
        prof_notes.append(f"Vòng mông: {user_profile.hip_cm}cm")
    if user_profile.gender:
        prof_notes.append(f"Giới tính: {user_profile.gender.value}")
    if user_profile.fit_preference:
        prof_notes.append(f"Gu mặc: {user_profile.fit_preference}")
    if not prof_notes:
        return ""
    return (
        "\nThông tin khách hàng đã biết (áp dụng để gợi ý sản phẩm "
        f"& chọn size): {', '.join(prof_notes)}"
    )


def _resolve_policy_block_name_and_key(
    prompt_key: PromptKey | None = None,
    clarification_mode: ClarificationMode | None = None,
) -> tuple[str, str, PromptKey]:
    if prompt_key is not None:
        return (
            get_policy_prompt_block(prompt_key),
            PROMPT_NAMES_BY_KEY[prompt_key],
            prompt_key,
        )
    if clarification_mode == ClarificationMode.IMMEDIATE_ANSWER:
        return MODE_A_DIRECTIVE, "MODE_A_DIRECTIVE", PromptKey.MODE_A
    if clarification_mode == ClarificationMode.ALWAYS_CLARIFY:
        return PROMPT_B_INTERVIEW, "PROMPT_B_INTERVIEW", PromptKey.MODE_B_INTERVIEW
    return MODE_C_DIRECTIVE, "MODE_C_DIRECTIVE", PromptKey.MODE_C


def build_system_prompt_with_debug(
    user_profile: UserProfile | None,
    prompt_key: PromptKey | None = None,
    slot_summary: str = "",
    clarification_mode: ClarificationMode | None = None,
) -> tuple[str, SystemPromptDebug]:
    """Assemble composable prompt blocks with mode directive placed first."""
    persona_clean = PERSONA_CORE.strip()
    profile_notes = format_profile_notes(user_profile)
    clean_profile = profile_notes.lstrip("\n").strip()
    clean_slots = slot_summary.strip()

    policy_block, policy_name, resolved_key = _resolve_policy_block_name_and_key(
        prompt_key=prompt_key,
        clarification_mode=clarification_mode,
    )
    policy_clean = policy_block.strip()
    product_clean = PRODUCT_RULES.strip()
    support_clean = SUPPORT_RULES.strip()

    if resolved_key == PromptKey.NON_SHOPPING:
        sections = [
            SystemPromptSectionDebug(
                id="base",
                name="PERSONA_CORE",
                included=True,
                char_count=len(persona_clean),
                content=persona_clean,
            ),
            SystemPromptSectionDebug(
                id="policy",
                name=policy_name,
                included=True,
                char_count=len(policy_clean),
                content=policy_clean,
            ),
            SystemPromptSectionDebug(
                id="support_rules",
                name="SUPPORT_RULES",
                included=True,
                char_count=len(support_clean),
                content=support_clean,
            ),
            SystemPromptSectionDebug(
                id="profile",
                name="PROFILE_NOTES",
                included=bool(clean_profile),
                char_count=len(clean_profile),
                content=clean_profile,
            ),
        ]
    else:
        sections = [
            SystemPromptSectionDebug(
                id="base",
                name="PERSONA_CORE",
                included=True,
                char_count=len(persona_clean),
                content=persona_clean,
            ),
            SystemPromptSectionDebug(
                id="policy",
                name=policy_name,
                included=True,
                char_count=len(policy_clean),
                content=policy_clean,
            ),
            SystemPromptSectionDebug(
                id="product_rules",
                name="PRODUCT_RULES",
                included=True,
                char_count=len(product_clean),
                content=product_clean,
            ),
            SystemPromptSectionDebug(
                id="support_rules",
                name="SUPPORT_RULES",
                included=True,
                char_count=len(support_clean),
                content=support_clean,
            ),
            SystemPromptSectionDebug(
                id="profile",
                name="PROFILE_NOTES",
                included=bool(clean_profile),
                char_count=len(clean_profile),
                content=clean_profile,
            ),
            SystemPromptSectionDebug(
                id="slot_summary",
                name="SLOT_SUMMARY",
                included=bool(clean_slots),
                char_count=len(clean_slots),
                content=clean_slots,
            ),
        ]

    full_prompt = "\n\n".join(s.content for s in sections if s.included) + "\n"
    formula = " + ".join(s.name for s in sections if s.included)
    return full_prompt, SystemPromptDebug(
        version=SYSTEM_PROMPT_VERSION,
        formula=formula,
        total_chars=len(full_prompt),
        sections=sections,
    )


def build_system_prompt(
    user_profile: UserProfile | None,
    prompt_key: PromptKey | None = None,
    slot_summary: str = "",
    clarification_mode: ClarificationMode | None = None,
) -> str:
    """Construct system prompt from composable blocks."""
    full_prompt, _ = build_system_prompt_with_debug(
        user_profile=user_profile,
        prompt_key=prompt_key,
        slot_summary=slot_summary,
        clarification_mode=clarification_mode,
    )
    return full_prompt


def build_interview_system_prompt_with_debug(
    user_profile: UserProfile | None,
    slot_summary: str = "",
) -> tuple[str, SystemPromptDebug]:
    """Build compact ALWAYS_CLARIFY INTERVIEW prompt and its breakdown."""
    profile_notes = format_profile_notes(user_profile)
    sys_content = INTERVIEW_SYSTEM_PROMPT + profile_notes
    if slot_summary:
        sys_content += f"\n{slot_summary}\n"

    clean_profile = profile_notes.lstrip("\n")
    sections = [
        SystemPromptSectionDebug(
            id="base",
            name="INTERVIEW_SYSTEM_PROMPT",
            included=True,
            char_count=len(INTERVIEW_SYSTEM_PROMPT),
            content=INTERVIEW_SYSTEM_PROMPT,
        ),
        SystemPromptSectionDebug(
            id="profile",
            name="PROFILE_NOTES",
            included=bool(clean_profile),
            char_count=len(clean_profile),
            content=clean_profile,
        ),
        SystemPromptSectionDebug(
            id="slot_summary",
            name="SLOT_SUMMARY",
            included=bool(slot_summary),
            char_count=len(slot_summary),
            content=slot_summary,
        ),
    ]
    formula = " + ".join(s.name for s in sections if s.included)
    return sys_content, SystemPromptDebug(
        version=SYSTEM_PROMPT_VERSION,
        formula=formula,
        total_chars=len(sys_content),
        sections=sections,
    )


def build_interview_system_prompt(
    user_profile: UserProfile | None,
    slot_summary: str = "",
) -> str:
    """Construct compact system prompt for ALWAYS_CLARIFY INTERVIEW turns."""
    sys_content, _ = build_interview_system_prompt_with_debug(
        user_profile=user_profile,
        slot_summary=slot_summary,
    )
    return sys_content


def build_extractor_system_prompt_with_debug() -> tuple[str, SystemPromptDebug]:
    """Build structured extractor system prompt along with its debug breakdown."""
    section = SystemPromptSectionDebug(
        id="base",
        name="EXTRACTOR_SYSTEM_PROMPT",
        included=True,
        char_count=len(EXTRACTOR_SYSTEM_PROMPT),
        content=EXTRACTOR_SYSTEM_PROMPT,
    )
    return EXTRACTOR_SYSTEM_PROMPT, SystemPromptDebug(
        version=SYSTEM_PROMPT_VERSION,
        formula="EXTRACTOR_SYSTEM_PROMPT",
        total_chars=len(EXTRACTOR_SYSTEM_PROMPT),
        sections=[section],
    )
