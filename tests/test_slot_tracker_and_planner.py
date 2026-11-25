"""Unit tests for multi-turn P0-P3 slot tracker and dynamic clarification planner."""

from langchain_core.messages import AIMessage, HumanMessage

from customer_shop_assistant.schemas.chat import (
    ClarificationForm,
    ClarificationQuestion,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.clarification_planner import (
    align_reply_with_clarification_form,
    build_clarification_form,
    validate_or_repair_clarification_form,
)
from customer_shop_assistant.services.slot_tracker import (
    TurnIntent,
    extract_conversation_slots,
    extract_session_profile_updates,
    format_slots_for_prompt,
)


def test_gender_is_first_question_for_shared_categories_and_size() -> None:
    queries = [
        "Tìm quần đi làm",
        "Tìm áo sơ mi",
        "Tư vấn đồ đi chơi",
        "Mình 1m70 65kg mặc áo size gì?",
    ]
    for query in queries:
        slots, intent = extract_conversation_slots([query])
        assert not slots.has("target_gender"), f"Should not assume gender for: {query}"
        form = build_clarification_form(slots, intent, max_questions=3)
        assert form.questions, f"Expected form questions for: {query}"
        assert form.questions[0].id == "target_gender", (
            f"Expected target_gender first for '{query}', got {form.questions[0].id}"
        )
        assert "Nam" in form.questions[0].options
        assert "Nữ" in form.questions[0].options


def test_dress_infers_female_gender_and_skips_gender_question() -> None:
    slots, intent = extract_conversation_slots(["Tìm đầm dự tiệc"])
    assert intent == TurnIntent.PRODUCT_SEARCH
    assert slots.target_gender is not None
    assert slots.target_gender.value == "FEMALE"
    assert slots.target_gender.source == "inferred"
    assert slots.value_of("occasion") == "party"
    assert slots.value_of("category") == "dress"

    form = build_clarification_form(slots, intent, max_questions=3)
    q_ids = [q.id for q in form.questions]
    assert "target_gender" not in q_ids
    assert "occasion" not in q_ids
    assert "measurements" not in q_ids


def test_detailed_query_asks_refinement_and_never_measurements() -> None:
    slots, intent = extract_conversation_slots(["Tìm áo sơ mi nam đi làm dưới 500k"])
    assert intent == TurnIntent.PRODUCT_SEARCH
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("category") == "shirt"
    assert slots.value_of("occasion") == "office"
    assert slots.has("budget")
    assert slots.value_of("budget").max_price == 500000.0

    # Spec v1.1: When all Required Slots are resolved, default planner returns 0
    # questions and never auto-asks Optional Slots or Not Applicable slots.
    form = build_clarification_form(slots, intent, max_questions=3)
    assert form.questions == []

    # Conditional slots (e.g. fit) can still be built when explicitly targeted
    cond_form = build_clarification_form(
        slots, intent, max_questions=2, target_slots=["fit"]
    )
    assert [q.id for q in cond_form.questions] == ["fit"]


def test_multi_turn_slot_accumulation_and_session_profile_sync() -> None:
    history = [
        HumanMessage(content="Tìm quần đi làm"),
        AIMessage(content="Bạn giúp mình chọn thông tin bên dưới nhé:"),
        HumanMessage(content="Nam\nQuần âu / Quần kaki lịch sự\n300.000đ – 600.000đ"),
    ]
    slots, intent = extract_conversation_slots(history, UserProfile())
    assert intent == TurnIntent.PRODUCT_SEARCH
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("category") == "pants"
    assert slots.value_of("occasion") == "office"
    assert slots.has("budget")
    assert slots.value_of("budget").min_price == 300000.0
    assert slots.value_of("budget").max_price == 600000.0

    updates = extract_session_profile_updates(slots)
    assert updates["gender"] == GenderTarget.MALE


def test_unisex_does_not_misclassify_as_male() -> None:
    slots, _ = extract_conversation_slots(["Unisex (Nam nữ đều mặc được)"])
    assert slots.value_of("target_gender") == "UNISEX"


def test_catalog_aligned_options_and_context_adaptation() -> None:
    # Casual jeans should not offer 'Dự tiệc' when occasion is targeted
    slots_jeans, intent_jeans = extract_conversation_slots(["Tìm quần jeans nam"])
    form_jeans = build_clarification_form(
        slots_jeans, intent_jeans, max_questions=3, target_slots=["occasion"]
    )
    occ_q = next(q for q in form_jeans.questions if q.id == "occasion")
    assert not any("Dự tiệc" in opt for opt in occ_q.options)

    # Male shopper should not receive skirt/dress options in category question
    slots_male, intent_male = extract_conversation_slots(["Tìm đồ đi chơi cho nam"])
    form_male = build_clarification_form(
        slots_male,
        intent_male,
        max_questions=3,
        target_slots=["category", "budget"],
    )
    cat_q = next(q for q in form_male.questions if q.id == "category")
    assert not any("Đầm" in opt or "Chân váy" in opt for opt in cat_q.options)

    # Budget options match catalog price distribution without mixing measurements
    budget_q = next(q for q in form_male.questions if q.id == "budget")
    assert any("300.000đ – 600.000đ" in opt for opt in budget_q.options)
    assert budget_q.custom_placeholder is not None
    assert "chiều cao" not in budget_q.custom_placeholder.lower()


def test_validate_or_repair_form_respects_valid_llm_form_in_conditional() -> None:
    slots, intent = extract_conversation_slots(["Tìm quần đi làm"])
    valid_llm_form = ClarificationForm(
        title="Chọn nhanh tiêu chí quần đi làm",
        submit_label="Xem gợi ý",
        questions=[
            ClarificationQuestion(
                id="target_gender",
                question="1. Bạn tìm quần đi làm cho Nam hay Nữ?",
                type="single",
                options=["Quần công sở Nam", "Quần công sở Nữ"],
            )
        ],
    )
    res = validate_or_repair_clarification_form(
        valid_llm_form, slots, intent, force_planner=False
    )
    assert res == valid_llm_form

    # Reject if LLM asks known occasion or unsolicited measurements
    bad_llm_form = ClarificationForm(
        title="Thông tin thêm",
        questions=[
            ClarificationQuestion(
                id="occasion",
                question="1. Bạn mặc vào dịp nào?",
                type="single",
                options=["Đi làm", "Đi chơi"],
            ),
            ClarificationQuestion(
                id="body_measurements",
                question="2. Chiều cao và cân nặng của bạn?",
                type="single",
                options=["Size M", "Size L"],
            ),
        ],
    )
    repaired = validate_or_repair_clarification_form(
        bad_llm_form, slots, intent, force_planner=False, max_fallback_questions=2
    )
    assert repaired != bad_llm_form
    assert len(repaired.questions) <= 2
    assert repaired.questions[0].id == "target_gender"
    assert all(q.id != "measurements" for q in repaired.questions)


def test_non_shopping_and_stock_intent_classification() -> None:
    for q in (
        "Phí ship của shop bao nhiêu?",
        "Chính sách đổi trả thế nào?",
        "Kiểm tra mã SUMMER50 cho đơn 600k",
        "Chi tiết sản phẩm ID 57",
        "Cảm ơn shop nhiều nhé",
    ):
        _, intent = extract_conversation_slots([q])
        assert intent == TurnIntent.NON_SHOPPING, f"Expected NON_SHOPPING for: {q}"

    _, stock_intent = extract_conversation_slots(
        ["Mẫu áo khoác màu đen size L còn hàng không?"]
    )
    assert stock_intent == TurnIntent.PRODUCT_SEARCH


def test_prompt_summary_excludes_p3_measurements_for_product_search() -> None:
    slots, intent = extract_conversation_slots(["Tìm quần đi làm"])
    summary = format_slots_for_prompt(slots, intent)
    assert "Giới tính/Đối tượng [P0]" in summary
    assert "Số đo cơ thể" not in summary


def test_align_reply_catches_single_line_numbered_list_and_garbled_phrase() -> None:
    slots, intent = extract_conversation_slots(["tìm giúp tôi đồ đi đám cưới cho nam"])
    form = build_clarification_form(slots, intent, max_questions=3)
    single_line_leaked = (
        "Chào anh! Rất vui được tư vấn đồ đi đám cưới cho nam giới của anh nhé! 🎉 "
        "Để giúp shop tư vấn chính xác nhất, anh có thể tích điểm vào các mục dưới "
        "đây: 1. Bạn đang tìm kiểu trang phục dự tiệc nào? (Đám cưới, Gala, Party, "
        "Wedding) 2. Ngân sách dự kiến của bạn cho trang phục này? (300k - 500k, "
        "500k - 800k, 800k+) 3. Bạn thích phong cách thẩm mỹ nào? (Minimalism, "
        "Công sở, Streetwear, Casual) Anh có thể chọn và tích vào các mục trên để "
        "shop tư vấn chính xác hơn nhé! 😊"
    )
    aligned = align_reply_with_clarification_form(
        single_line_leaked, form, slots, intent
    )
    assert "1." not in aligned
    assert "tích điểm" not in aligned
    assert "buổi tiệc" in aligned
    assert len(aligned.split()) <= 45


def test_outfit_dual_category_extraction_and_multi_turn_intent_preservation() -> None:
    slots, intent = extract_conversation_slots(
        ["Shop gợi ý cho mình một set đồ gồm áo thun trắng và quần jeans đen cho nam"]
    )
    assert intent == TurnIntent.OUTFIT_ADVICE
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("top_category") == "tshirt"
    assert slots.value_of("bottom_category") == "jeans"
    assert slots.value_of("top_color") == "Trắng"
    assert slots.value_of("bottom_color") == "Đen"

    summary = format_slots_for_prompt(slots, intent)
    assert "Loại áo phối" in summary
    assert "Loại quần/váy phối" in summary

    # Clarification form for OUTFIT_ADVICE must not ask single-category selection
    slots_vague, intent_vague = extract_conversation_slots(["Phối đồ cho mình với"])
    assert intent_vague == TurnIntent.OUTFIT_ADVICE
    form = build_clarification_form(slots_vague, intent_vague, max_questions=3)
    assert all(q.id not in ("category", "category_subtype") for q in form.questions)

    # Answering clarification form preserves OUTFIT_ADVICE in turn 2
    history = [
        HumanMessage(content="Phối đồ cho mình với"),
        AIMessage(content="Bạn giúp mình chọn thông tin bên dưới nhé:"),
        HumanMessage(content="Nam\nĐi làm công sở – Lịch sự, chỉn chu"),
    ]
    slots_t2, intent_t2 = extract_conversation_slots(history)
    assert intent_t2 == TurnIntent.OUTFIT_ADVICE
    assert slots_t2.value_of("target_gender") == "MALE"
    assert slots_t2.value_of("occasion") == "office"


def test_switching_from_shirt_to_pants_updates_category_and_hides_outfit_slots() -> (
    None
):
    history = [
        HumanMessage(content="Tìm áo sơ mi công sở"),
        AIMessage(content="Bạn giúp mình chọn thêm tiêu chí nhé:"),
        HumanMessage(
            content=(
                "Mình chọn:\n1. Nam\n2. Phổ biến (300.000đ – 600.000đ)\n"
                "3. Form ôm nhẹ gọn gàng, tôn dáng (Slim fit)"
            )
        ),
        AIMessage(content="Dưới đây là các mẫu áo sơ mi..."),
        HumanMessage(content="gợi ý vài loại quần phù hợp"),
    ]
    slots, intent = extract_conversation_slots(history)
    assert intent == TurnIntent.PRODUCT_SEARCH
    assert slots.value_of("category") == "pants"
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("occasion") == "office"

    summary = format_slots_for_prompt(slots, intent)
    assert "Danh mục [P0]=pants" in summary
    assert "Loại áo phối" not in summary


def test_followup_pairing_question_transitions_to_outfit_advice() -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    from customer_shop_assistant.config import Settings
    from customer_shop_assistant.services.agent_service import ShopAssistantAgent

    base_history = [
        HumanMessage(content="Tìm áo sơ mi công sở"),
        AIMessage(content="Bạn giúp mình chọn thêm tiêu chí nhé:"),
        HumanMessage(
            content=(
                "Mình chọn:\n1. Nữ\n2. Phổ biến (300.000đ – 600.000đ)\n"
                "3. Form ôm nhẹ gọn gàng, tôn dáng (Slim fit)"
            )
        ),
        AIMessage(content="Dưới đây là 4 mẫu áo sơ mi nữ công sở..."),
    ]

    for pairing_q in (
        "mặc gì bên dưới thì phù hợp?",
        "tìm giúp tôi đồ mặc bên dưới phù hợp với áo",
        "tìm quần phù hợp với áo sơ mi",
    ):
        slots, intent = extract_conversation_slots(
            [*base_history, HumanMessage(content=pairing_q)]
        )
        assert intent == TurnIntent.OUTFIT_ADVICE, f"Failed for: {pairing_q}"
        assert slots.value_of("shopping_intent") == "OUTFIT"
        assert slots.value_of("top_category") == "shirt"
        assert slots.value_of("bottom_category") is None
        assert slots.value_of("target_gender") == "FEMALE"
        assert slots.value_of("occasion") == "office"

        summary = format_slots_for_prompt(slots, intent)
        assert "TÓM TẮT NGỮ CẢNH (OUTFIT_ADVICE)" in summary
        assert "Loại áo phối [P0]=shirt" in summary
        assert "Danh mục [P0]" not in summary

        agent = ShopAssistantAgent(
            chat_model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
            settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True),
        )
        active_tool_names = [
            t.name for t in agent._get_active_tools_for_turn(intent, False)
        ]
        assert "suggest_outfit" in active_tool_names
        assert "search_products" not in active_tool_names

    # Specific bottom paired with shirt sets both top_category and bottom_category
    slots_skirt, intent_skirt = extract_conversation_slots(
        [*base_history, HumanMessage(content="tìm chân váy hợp với áo sơ mi")]
    )
    assert intent_skirt == TurnIntent.OUTFIT_ADVICE
    assert slots_skirt.value_of("top_category") == "shirt"
    assert slots_skirt.value_of("bottom_category") == "skirt"

    # Bottom -> asking what to wear on top keeps bottom_category as anchor
    slots_top, intent_top = extract_conversation_slots(
        [
            HumanMessage(content="Tìm quần jeans nữ"),
            AIMessage(content="Đây là các mẫu quần jeans nữ..."),
            HumanMessage(content="mặc gì bên trên thì hợp?"),
        ]
    )
    assert intent_top == TurnIntent.OUTFIT_ADVICE
    assert slots_top.value_of("shopping_intent") == "OUTFIT"
    assert slots_top.value_of("bottom_category") == "jeans"
    assert slots_top.value_of("top_category") is None

    # Switching back from OUTFIT_ADVICE to standalone single-item search resets intent
    slots_back, intent_back = extract_conversation_slots(
        [
            *base_history,
            HumanMessage(content="mặc gì bên dưới thì phù hợp?"),
            AIMessage(content="Bộ 1: Áo sơ mi + Quần âu..."),
            HumanMessage(content="tìm quần đi làm"),
        ]
    )
    assert intent_back == TurnIntent.PRODUCT_SEARCH
    assert slots_back.value_of("shopping_intent") == "PRODUCT"
    assert slots_back.value_of("category") == "pants"

    # Explicit top+bottom pairing questions without 'phối' also trigger OUTFIT_ADVICE
    slots_shirt_q, intent_shirt_q = extract_conversation_slots(
        [HumanMessage(content="áo sơ mi trắng thì nên chọn loại quần nào?")]
    )
    assert intent_shirt_q == TurnIntent.OUTFIT_ADVICE
    assert slots_shirt_q.value_of("top_category") == "shirt"
    assert slots_shirt_q.value_of("top_color") == "Trắng"
    assert slots_shirt_q.value_of("bottom_category") is None

    slots_pants_q, intent_pants_q = extract_conversation_slots(
        [HumanMessage(content="quần âu đen thì nên mặc kiểu áo nào?")]
    )
    assert intent_pants_q == TurnIntent.OUTFIT_ADVICE
    assert slots_pants_q.value_of("bottom_category") == "pants"
    assert slots_pants_q.value_of("bottom_color") == "Đen"
    assert slots_pants_q.value_of("top_category") is None


def test_size_advice_form_submit_preserves_intent_and_parses_ranges_cleanly() -> None:
    history = [
        HumanMessage(content="Tư vấn chọn size giúp mình với"),
        AIMessage(
            content=(
                "Chào bạn! Để tư vấn size vừa vặn và chuẩn xác nhất, "
                "bạn giúp mình chọn hoặc nhập thông tin còn thiếu ở bảng dưới đây nhé:"
            )
        ),
        HumanMessage(
            content=(
                "Mình chọn:\n"
                "1. Nam\n"
                "2. Áo (Sơ mi / Thun / Polo / Khoác)\n"
                "3. Cân đối / Đầy đặn – Khung Size L (63–72kg | 1m68–1m75)"
            )
        ),
    ]
    slots, intent = extract_conversation_slots(history)
    assert intent == TurnIntent.SIZE_ADVICE
    assert slots.value_of("target_gender") == "MALE"
    assert slots.value_of("category") == "tops"
    assert not slots.has("budget")
    assert not slots.has("body_concern")
    meas = slots.value_of("measurements")
    assert meas is not None
    assert meas.height_cm == 171.5
    assert meas.weight_kg == 67.5

    # Custom height/weight submission without word 'size' still preserves SIZE_ADVICE
    custom_history = [
        history[0],
        history[1],
        HumanMessage(
            content=(
                "Mình chọn:\n1. Nam\n2. Áo (Sơ mi / Thun / Polo / Khoác)\n3. 1m70, 65kg"
            )
        ),
    ]
    slots_custom, intent_custom = extract_conversation_slots(custom_history)
    assert intent_custom == TurnIntent.SIZE_ADVICE
    assert slots_custom.value_of("measurements").height_cm == 170.0
    assert slots_custom.value_of("measurements").weight_kg == 65.0

    # Partial text answer preserves SIZE_ADVICE, while explicit search switches intent
    _, intent_partial = extract_conversation_slots(
        [history[0], history[1], HumanMessage(content="Mình là nam")]
    )
    assert intent_partial == TurnIntent.SIZE_ADVICE

    _, intent_switch = extract_conversation_slots(
        [history[0], history[1], HumanMessage(content="Tìm áo sơ mi nam")]
    )
    assert intent_switch == TurnIntent.PRODUCT_SEARCH


def test_build_form_allows_provided_reconfirm_on_unfulfillable_constraints() -> None:
    slots_col, intent_col = extract_conversation_slots(
        [HumanMessage(content="Phối áo sơ mi nam màu xanh mint với quần âu đi làm.")]
    )
    assert slots_col.has("color")
    assert (
        build_clarification_form(
            slots_col, intent_col, max_questions=1, target_slots=["color"]
        ).questions
        == []
    )
    reconfirm_col = build_clarification_form(
        slots_col,
        intent_col,
        max_questions=1,
        target_slots=["color"],
        allow_provided=True,
    )
    assert [q.id for q in reconfirm_col.questions] == ["color"]

    slots_bud, intent_bud = extract_conversation_slots(
        [HumanMessage(content="Phối đồ đi tiệc cho nam dưới 100k")]
    )
    assert slots_bud.has("budget")
    reconfirm_bud = build_clarification_form(
        slots_bud,
        intent_bud,
        max_questions=1,
        target_slots=["budget"],
        allow_provided=True,
    )
    assert [q.id for q in reconfirm_bud.questions] == ["budget"]
    assert "chưa có bộ phối phù hợp" in reconfirm_bud.questions[0].question
