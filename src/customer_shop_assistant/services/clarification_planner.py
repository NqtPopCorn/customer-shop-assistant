"""Dynamic clarification form planner, strict validator, and reply aligner."""

import re

from customer_shop_assistant.schemas.chat import (
    ClarificationForm,
    ClarificationQuestion,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.slot_tracker import (
    ConversationSlots,
    TurnIntent,
    get_missing_required_slots,
    get_not_applicable_slots,
    get_optional_slots,
)

_CASUAL_GARMENTS = {"jeans", "tshirt", "short", "polo"}
_MEASUREMENT_INTENTS = {TurnIntent.SIZE_ADVICE, TurnIntent.BODY_FIT_ADVICE}

_OCCASION_LABEL_MAP = {
    "party": "cho buổi tiệc sắp tới",
    "office": "đi làm công sở",
    "casual": "đi chơi, dạo phố",
    "sport": "vận động, thể thao",
}

_FORM_TITLE_MAP = {
    TurnIntent.SIZE_ADVICE: "Xác định vóc dáng và chọn size chuẩn",
    TurnIntent.BODY_FIT_ADVICE: "Xác định vóc dáng và nhu cầu tôn dáng",
    TurnIntent.OUTFIT_ADVICE: "Xác định nhu cầu phối đồ của bạn",
}

_OCCASION_TITLE_MAP = {
    "party": "Xác định nhu cầu trang phục dự tiệc của bạn",
    "office": "Xác định nhu cầu trang phục công sở của bạn",
    "casual": "Xác định nhu cầu trang phục dạo phố của bạn",
    "sport": "Xác định nhu cầu trang phục thể thao của bạn",
}

_SLOT_ID_ALIASES = {
    "gender": "target_gender",
    "body_measurements": "measurements",
}


def _build_gender_question(intent: TurnIntent) -> ClarificationQuestion:
    if intent in _MEASUREMENT_INTENTS:
        options = ["Nam", "Nữ"]
    else:
        options = ["Nam", "Nữ", "Unisex (Nam nữ đều mặc được)"]
    return ClarificationQuestion(
        id="target_gender",
        question="Bạn đang tìm trang phục cho đối tượng nào?",
        type="single",
        options=options,
        allow_custom=False,
    )


def _build_occasion_question(slots: ConversationSlots) -> ClarificationQuestion:
    category = slots.value_of("category")
    if category in _CASUAL_GARMENTS:
        options = [
            "Đi chơi / Dạo phố – Thoải mái, trẻ trung",
            "Đi làm thoải mái (Smart-casual)",
            "Mặc hằng ngày / Du lịch",
        ]
    else:
        options = [
            "Đi làm công sở – Lịch sự, chỉn chu",
            "Đi chơi / Dạo phố – Thoải mái, trẻ trung",
            "Dự tiệc / Sự kiện – Nổi bật, sang trọng",
        ]
    return ClarificationQuestion(
        id="occasion",
        question="Bạn đang tìm trang phục cho dịp nào?",
        type="single",
        options=options,
        allow_custom=False,
    )


def _build_category_question(
    slots: ConversationSlots, intent: TurnIntent
) -> ClarificationQuestion:
    gender = slots.value_of("target_gender")
    occasion = slots.value_of("occasion")

    if intent == TurnIntent.SIZE_ADVICE:
        if gender == GenderTarget.MALE.value:
            options = [
                "Áo Nam (Áo thun, Polo, Sơ mi, Áo khoác)",
                "Quần Nam (Quần âu, Kaki, Jeans, Short)",
            ]
        elif gender == GenderTarget.FEMALE.value:
            options = [
                "Áo Nữ (Áo thun, Sơ mi, Áo khoác)",
                "Quần Nữ (Quần âu, Jeans, Kaki)",
                "Đầm / Váy liền / Chân váy",
            ]
        else:
            options = [
                "Áo (Sơ mi / Thun / Polo / Khoác)",
                "Quần (Quần âu / Kaki / Jeans / Short)",
                "Đầm / Váy liền / Chân váy (Nữ)",
            ]
        return ClarificationQuestion(
            id="category",
            question="Bạn cần tư vấn size cho nhóm trang phục nào?",
            type="single",
            options=options,
            allow_custom=False,
        )

    if occasion == "party":
        if gender == GenderTarget.MALE.value:
            options = [
                "Áo sơ mi / Blazer dự tiệc lịch lãm",
                "Quần âu / Quần tây phối vest",
                "Set đồ dự tiệc Nam hoàn chỉnh",
            ]
        elif gender == GenderTarget.FEMALE.value:
            options = [
                "Đầm / Váy dự tiệc sang trọng, tôn dáng",
                "Set đồ thanh lịch (Áo kiểu / Blazer + Chân váy / Quần)",
            ]
        else:
            options = [
                "Đầm / Váy dự tiệc sang trọng (Nữ)",
                "Trang phục dự tiệc Nam (Sơ mi / Blazer + Quần âu)",
                "Set đồ phối thanh lịch",
            ]
        return ClarificationQuestion(
            id="category",
            question="Bạn đang tìm kiểu trang phục dự tiệc nào?",
            type="single",
            options=options,
            allow_custom=False,
        )

    if occasion == "office":
        if gender == GenderTarget.MALE.value:
            options = [
                "Áo sơ mi / Áo polo thanh lịch đi làm",
                "Quần âu / Quần kaki công sở",
                "Áo khoác Blazer lịch sự",
            ]
        elif gender == GenderTarget.FEMALE.value:
            options = [
                "Áo sơ mi / Áo kiểu công sở",
                "Quần âu / Chân váy công sở",
                "Đầm công sở liền thân hoặc Áo khoác Blazer",
            ]
        else:
            options = [
                "Áo (Sơ mi / Polo / Blazer công sở)",
                "Quần (Quần âu / Quần kaki công sở)",
                "Đầm công sở liền thân / Chân váy (Nữ)",
            ]
        return ClarificationQuestion(
            id="category",
            question="Bạn đang tìm nhóm trang phục công sở nào?",
            type="single",
            options=options,
            allow_custom=False,
        )

    if gender == GenderTarget.MALE.value:
        options = [
            "Áo sơ mi / Áo polo lịch sự",
            "Áo thun / Áo khoác năng động",
            "Quần âu / Quần kaki / Quần jeans / Short",
        ]
    elif gender == GenderTarget.FEMALE.value:
        options = [
            "Áo thun / Áo sơ mi / Áo khoác",
            "Quần jeans / Quần âu / Kaki",
            "Đầm / Váy liền / Chân váy",
        ]
    else:
        options = [
            "Áo (Sơ mi / Thun / Polo / Khoác)",
            "Quần (Quần âu / Kaki / Jeans / Short)",
            "Đầm / Váy liền / Chân váy (Nữ)",
        ]

    return ClarificationQuestion(
        id="category",
        question="Bạn đang quan tâm đến nhóm trang phục nào?",
        type="single",
        options=options,
        allow_custom=False,
    )


def _build_category_subtype_question(
    slots: ConversationSlots,
) -> ClarificationQuestion:
    category = slots.value_of("category")
    if category == "tops":
        return ClarificationQuestion(
            id="category_subtype",
            question="Bạn đang muốn tìm kiểu áo nào?",
            type="single",
            options=[
                "Áo sơ mi thanh lịch",
                "Áo thun / Áo polo năng động",
                "Áo khoác / Blazer",
            ],
            allow_custom=False,
        )
    return ClarificationQuestion(
        id="category_subtype",
        question="Bạn đang muốn tìm kiểu quần nào?",
        type="single",
        options=[
            "Quần âu / Quần kaki lịch sự",
            "Quần jeans trẻ trung",
            "Quần short thoải mái",
        ],
        allow_custom=False,
    )


def _build_budget_question() -> ClarificationQuestion:
    return ClarificationQuestion(
        id="budget",
        question="Ngân sách dự kiến của bạn cho trang phục này?",
        type="single",
        options=[
            "Tiết kiệm (Dưới 300.000đ)",
            "Phổ biến (300.000đ – 600.000đ)",
            "Cao cấp (Trên 600.000đ)",
        ],
        allow_custom=True,
        custom_placeholder=("Hoặc nhập ngân sách cụ thể (VD: dưới 450k, tầm 500k)..."),
    )


def _build_fit_question(slots: ConversationSlots) -> ClarificationQuestion:
    category = slots.value_of("category")
    if category in ("dress", "skirt"):
        options = [
            "Dáng xòe nhẹ nhàng, che khuyết điểm tốt",
            "Chiết eo / ôm nhẹ tôn dáng thanh lịch",
            "Dáng suông thoải mái, trẻ trung",
        ]
    else:
        options = [
            "Form vừa vặn, chỉn chu (Regular fit)",
            "Form ôm nhẹ gọn gàng, tôn dáng (Slim fit)",
            "Form rộng thoải mái, năng động (Relaxed / Oversize)",
        ]
    return ClarificationQuestion(
        id="fit",
        question="Bạn yêu thích kiểu dáng (form) trang phục như thế nào?",
        type="single",
        options=options,
        allow_custom=False,
    )


def _build_color_question(slots: ConversationSlots) -> ClarificationQuestion:
    occasion = slots.value_of("occasion")
    if occasion == "party":
        options = [
            "Trung tính cơ bản (Đen, Trắng kem, Navy)",
            "Điểm nhấn nổi bật (Đỏ đô, Vàng ánh kim, Xanh cổ vịt)",
            "Tông nhẹ nhàng thanh lịch (Hồng pastel, Be sữa)",
        ]
    else:
        options = [
            "Trung tính cơ bản (Đen, Trắng, Navy, Xám)",
            "Tông đất ấm (Be, Nâu camel, Xanh rêu)",
            "Điểm nhấn nổi bật (Đỏ đô, Xanh dương, Pastel)",
        ]
    return ClarificationQuestion(
        id="color",
        question="Bạn ưu tiên tông màu nào?",
        type="single",
        options=options,
        allow_custom=True,
        custom_placeholder="Hoặc nhập màu cụ thể bạn thích (VD: xanh rêu, be)...",
    )


def _build_style_question(slots: ConversationSlots) -> ClarificationQuestion:
    occasion = slots.value_of("occasion")
    if occasion == "party":
        options = [
            "Sang trọng, nổi bật (Ren, Lụa)",
            "Thanh lịch, tối giản (Minimalist)",
            "Trẻ trung, hiện đại",
        ]
    elif occasion == "office":
        options = [
            "Công sở lịch sự, chuẩn mực",
            "Tối giản (Minimalist), hiện đại",
            "Smart-casual trẻ trung, mặc được cả đi chơi",
        ]
    else:
        options = [
            "Tối giản (Minimalist), tinh gọn",
            "Trẻ trung, năng động (Casual / Streetwear)",
            "Lịch sự, thanh lịch (Smart-casual)",
        ]
    return ClarificationQuestion(
        id="style",
        question="Bạn thích phong cách thẩm mỹ nào? (Có thể chọn nhiều)",
        type="multiple",
        options=options,
        allow_custom=False,
    )


def _build_measurements_question() -> ClarificationQuestion:
    return ClarificationQuestion(
        id="measurements",
        question="Chiều cao, cân nặng hoặc khung vóc dáng hiện tại của bạn?",
        type="single",
        options=[
            "Nhỏ nhắn – Khung Size S (45–53kg | 1m50–1m60)",
            "Trung bình – Khung Size M (54–62kg | 1m60–1m68)",
            "Cân đối / Đầy đặn – Khung Size L (63–72kg | 1m68–1m75)",
            "To cao – Khung Size XL (73–82kg | 1m75–1m82)",
        ],
        allow_custom=True,
        custom_placeholder=("Hoặc nhập chiều cao, cân nặng cụ thể (VD: 1m70, 65kg)..."),
    )


def _build_body_concern_question() -> ClarificationQuestion:
    return ClarificationQuestion(
        id="body_concern",
        question="Bạn muốn ưu tiên cải thiện hoặc tôn điểm nào trên vóc dáng?",
        type="single",
        options=[
            "Dáng gầy, muốn mặc trông đầy đặn hơn",
            "Muốn che vòng bụng gọn gàng, thoải mái",
            "Chiều cao khiêm tốn, muốn hack dáng cao hơn",
        ],
        allow_custom=True,
        custom_placeholder="Hoặc mô tả ngắn đặc điểm vóc dáng của bạn...",
    )


def _build_conflict_question(
    slot_name: str,
    slots: ConversationSlots,
    intent: TurnIntent,
) -> ClarificationQuestion | None:
    if slot_name == "target_gender":
        options = (
            ["Nam", "Nữ"]
            if intent in _MEASUREMENT_INTENTS
            else ["Nam", "Nữ", "Unisex (Nam nữ đều mặc được)"]
        )
        return ClarificationQuestion(
            id="target_gender",
            question=(
                "Yêu cầu hiện tại có thông tin mâu thuẫn, "
                "bạn xác nhận lại đối tượng mặc giúp mình nhé?"
            ),
            type="single",
            options=options,
            allow_custom=False,
        )
    if slot_name == "budget":
        budget_q = (
            "Khoảng giá bạn nêu đang mâu thuẫn, "
            "bạn chọn lại ngân sách ưu tiên giúp mình nhé?"
            if slots.is_conflicting("budget")
            else "Khoảng giá bạn nêu hiện chưa có bộ phối phù hợp, "
            "bạn chọn lại ngân sách ưu tiên giúp mình nhé?"
        )
        return ClarificationQuestion(
            id="budget",
            question=budget_q,
            type="single",
            options=[
                "Tiết kiệm (Dưới 300.000đ)",
                "Phổ biến (300.000đ – 600.000đ)",
                "Cao cấp (Trên 600.000đ)",
            ],
            allow_custom=True,
            custom_placeholder="Hoặc nhập khoảng giá chính xác (VD: dưới 500k)...",
        )
    if slot_name == "color":
        return ClarificationQuestion(
            id="color",
            question=(
                "Màu sắc bạn nêu đang mâu thuẫn hoặc chưa sẵn có, "
                "bạn chọn lại tông màu ưu tiên giúp mình nhé?"
            ),
            type="single",
            options=[
                "Trung tính cơ bản (Đen, Trắng, Navy, Xám)",
                "Tông đất ấm (Be, Nâu camel, Xanh rêu)",
                "Điểm nhấn nổi bật (Đỏ đô, Xanh dương, Pastel)",
            ],
            allow_custom=True,
            custom_placeholder="Hoặc nhập màu thay thế bạn thích (VD: trắng, navy)...",
        )
    return _build_question_for_slot(slot_name, slots, intent)


def _build_question_for_slot(
    slot_name: str, slots: ConversationSlots, intent: TurnIntent
) -> ClarificationQuestion | None:
    if slot_name == "target_gender":
        return _build_gender_question(intent)
    if slot_name == "occasion":
        return _build_occasion_question(slots)
    if slot_name == "category":
        return _build_category_question(slots, intent)
    if slot_name == "category_subtype":
        return _build_category_subtype_question(slots)
    if slot_name == "budget":
        return _build_budget_question()
    if slot_name == "fit":
        return _build_fit_question(slots)
    if slot_name == "color":
        return _build_color_question(slots)
    if slot_name == "style":
        return _build_style_question(slots)
    if slot_name == "measurements":
        return _build_measurements_question()
    if slot_name == "body_concern":
        return _build_body_concern_question()
    return None


def build_clarification_form(
    slots: ConversationSlots,
    intent: TurnIntent,
    max_questions: int = 3,
    *,
    target_slots: list[str] | None = None,
    allow_provided: bool = False,
) -> ClarificationForm:
    """Build a non-redundant ClarificationForm from required or target slots."""
    effective_intent = (
        TurnIntent.PRODUCT_SEARCH if intent == TurnIntent.NON_SHOPPING else intent
    )
    not_applicable = get_not_applicable_slots(effective_intent)

    if target_slots is not None:
        candidate_slots = [
            s
            for s in target_slots
            if s not in not_applicable
            and not slots.is_declined(s)
            and (allow_provided or slots.is_conflicting(s) or not slots.has(s))
        ]
    else:
        candidate_slots = get_missing_required_slots(effective_intent, slots)

    questions: list[ClarificationQuestion] = []
    for slot_name in candidate_slots:
        if len(questions) >= max_questions:
            break
        if slots.is_conflicting(slot_name) or (allow_provided and slots.has(slot_name)):
            q = _build_conflict_question(slot_name, slots, effective_intent)
        else:
            q = _build_question_for_slot(slot_name, slots, effective_intent)
        if q is not None:
            idx = len(questions) + 1
            q.question = f"{idx}. {q.question}"
            questions.append(q)

    occasion = slots.value_of("occasion")
    title = _FORM_TITLE_MAP.get(
        effective_intent,
        _OCCASION_TITLE_MAP.get(occasion or "", "Xác định nhu cầu trang phục của bạn"),
    )
    submit_label = (
        "Xác nhận số đo và xem tư vấn size"
        if effective_intent == TurnIntent.SIZE_ADVICE
        else "Xác nhận và xem gợi ý"
    )
    return ClarificationForm(
        title=title, submit_label=submit_label, questions=questions
    )


def _normalize_question_slot_id(raw_id: str) -> str:
    cleaned = raw_id.strip().lower()
    return _SLOT_ID_ALIASES.get(cleaned, cleaned)


def _is_question_invalid_for_state(
    q: ClarificationQuestion, slots: ConversationSlots, intent: TurnIntent
) -> bool:
    q_id = _normalize_question_slot_id(q.id)
    q_text = q.question.strip().lower()

    not_applicable = get_not_applicable_slots(intent)
    optional_slots = set(get_optional_slots(intent))

    if q_id in not_applicable or q_id in optional_slots:
        return True

    if slots.is_declined(q_id):
        return True

    if q_id != "measurements" and not slots.is_conflicting(q_id) and slots.has(q_id):
        return True

    if (
        q_id == "measurements"
        and intent == TurnIntent.SIZE_ADVICE
        and slots.has_complete_measurements(slots.value_of("category"))
    ):
        return True

    if (slots.has("target_gender") and not slots.is_conflicting("target_gender")) or (
        slots.is_declined("target_gender")
    ):
        if any(
            cue in q_text
            for cue in ("giới tính", "nam hay nữ", "đối tượng nào", "cho nam hay")
        ):
            return True

    if (slots.has("occasion") or slots.is_declined("occasion")) and any(
        cue in q_text for cue in ("dịp nào", "hoàn cảnh nào", "mục đích mặc")
    ):
        return True

    if (
        (slots.has("budget") and not slots.is_conflicting("budget"))
        or slots.is_declined("budget")
    ) and any(
        cue in q_text
        for cue in ("ngân sách", "tầm giá", "khoảng giá", "bao nhiêu tiền")
    ):
        return True

    if "measurements" in not_applicable and any(
        cue in q_text
        for cue in (
            "chiều cao",
            "cân nặng",
            "số đo",
            "3 vòng",
            "vòng eo",
            "vòng ngực",
        )
    ):
        return True

    return False


def validate_or_repair_clarification_form(
    form: ClarificationForm | None,
    slots: ConversationSlots,
    intent: TurnIntent,
    *,
    force_planner: bool = False,
    max_fallback_questions: int = 2,
    max_questions: int | None = None,
    target_slots: list[str] | None = None,
) -> ClarificationForm:
    """Validate LLM-proposed form or build a prioritized planner form."""
    cap = max_questions if max_questions is not None else max_fallback_questions
    if force_planner:
        return build_clarification_form(
            slots,
            intent,
            max_questions=max_questions if max_questions is not None else 3,
            target_slots=target_slots,
        )

    fallback = build_clarification_form(
        slots,
        intent,
        max_questions=cap,
        target_slots=target_slots,
    )
    if not form or not form.questions:
        return fallback

    if len(form.questions) > cap:
        return fallback

    for q in form.questions:
        if len(q.options) < 2 or len(q.options) > 4:
            return fallback
        if _is_question_invalid_for_state(q, slots, intent):
            return fallback
        for opt in q.options:
            opt_lower = opt.strip().lower()
            if opt_lower.endswith("?") or any(
                phrase in opt_lower
                for phrase in ("cho shop biết", "vui lòng", "bạn muốn", "bạn có thể")
            ):
                return fallback

    return form


def align_reply_with_clarification_form(
    reply_text: str,
    form: ClarificationForm,
    slots: ConversationSlots,
    intent: TurnIntent,
) -> str:
    """Keep intro reply concise (<=45 words) and conflict-free alongside the form."""
    del form
    cleaned = (reply_text or "").strip()
    lower_reply = cleaned.lower()

    welcome_phrases = (
        "mình có thể giúp bạn tìm kiếm trang phục",
        "bạn đang quan tâm đến phong cách nào hôm nay",
        "mình là stylist ai của fashion shop",
    )
    has_welcome_echo = any(wp in lower_reply for wp in welcome_phrases)
    has_inline_bullets = bool(
        re.search(
            r"(?:[:：]\s*(?:[-•*]|\d+\.)|\n\s*(?:[-•*]|\d+\.)|\b1\.\s+.+\b2\.)",
            cleaned,
        )
    )
    has_inline_table = bool(re.search(r"\|.*\|.*\n\s*\|[-:\s|]+\|", cleaned)) or (
        "|" in cleaned and "-|-" in cleaned
    )
    has_multiple_questions = cleaned.count("?") >= 2
    is_too_verbose = len(cleaned.split()) > 45 or len(cleaned) > 280
    has_garbled_phrases = any(p in lower_reply for p in ("tích điểm",))
    has_unsolicited_measurements = intent not in _MEASUREMENT_INTENTS and any(
        m in lower_reply
        for m in (
            "số đo 3 vòng",
            "chiều cao (cm)",
            "cân nặng (kg)",
            "dài, rộng, eo",
            "vòng eo",
            "vòng ngực",
        )
    )

    if (
        not cleaned
        or has_welcome_echo
        or has_inline_table
        or has_inline_bullets
        or has_multiple_questions
        or is_too_verbose
        or has_garbled_phrases
        or has_unsolicited_measurements
    ):
        if intent == TurnIntent.SIZE_ADVICE:
            return (
                "Chào bạn! Để tư vấn size vừa vặn và chuẩn xác nhất, "
                "bạn giúp mình chọn hoặc nhập thông tin còn thiếu ở bảng dưới đây nhé:"
            )
        known_occasion = slots.value_of("occasion")
        known_gender = slots.value_of("target_gender")
        occ_phrase = (
            f" {_OCCASION_LABEL_MAP[known_occasion]}"
            if known_occasion in _OCCASION_LABEL_MAP
            else (
                " phong cách và tôn dáng nam tính"
                if known_gender == GenderTarget.MALE.value
                else ""
            )
        )
        return (
            f"Chào bạn! Để mình chọn ra các mẫu trang phục{occ_phrase} "
            "phù hợp và ưng ý nhất, bạn giúp mình tích chọn nhanh "
            "vài tiêu chí ở bảng dưới đây nhé:"
        )

    return cleaned
