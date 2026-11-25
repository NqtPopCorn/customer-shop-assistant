"""Unit tests for v2.0 composable prompt_builder, policy, and tool scoping."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from customer_shop_assistant.config import Settings
from customer_shop_assistant.schemas.chat import (
    ChatRequest,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.agent_service import ShopAssistantAgent
from customer_shop_assistant.services.clarification_policy import (
    PromptKey,
    resolve_turn_policy,
)
from customer_shop_assistant.services.prompt_builder import (
    EXTRACTOR_SYSTEM_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    MODE_A_DIRECTIVE,
    MODE_B_OPEN_DIRECTIVE,
    MODE_C_DIRECTIVE,
    NON_SHOPPING_DIRECTIVE,
    PERSONA_CORE,
    PRODUCT_RULES,
    PROMPT_B_INTERVIEW,
    PROMPT_BLOCKS_BY_KEY,
    PROMPT_NAMES_BY_KEY,
    SUPPORT_RULES,
    SYSTEM_PROMPT_VERSION,
    build_extractor_system_prompt_with_debug,
    build_interview_system_prompt,
    build_interview_system_prompt_with_debug,
    build_system_prompt,
    build_system_prompt_with_debug,
    format_profile_notes,
    get_policy_prompt_block,
)
from customer_shop_assistant.services.session_manager import ClarifyPhase
from customer_shop_assistant.services.slot_tracker import TurnIntent
from customer_shop_assistant.tools.user_tool import update_customer_preference


def test_resolve_turn_policy_prompt_keys() -> None:
    p_a = resolve_turn_policy(
        ClarificationMode.IMMEDIATE_ANSWER, ClarifyPhase.INTERVIEW
    )
    assert p_a.gate_search is False
    assert p_a.prompt_key == PromptKey.MODE_A

    p_b_int = resolve_turn_policy(
        ClarificationMode.ALWAYS_CLARIFY, ClarifyPhase.INTERVIEW, is_shopping_query=True
    )
    assert p_b_int.gate_search is True
    assert p_b_int.prompt_key == PromptKey.MODE_B_INTERVIEW

    for mode in (
        ClarificationMode.IMMEDIATE_ANSWER,
        ClarificationMode.ALWAYS_CLARIFY,
        ClarificationMode.CONDITIONAL_CLARIFICATION,
    ):
        p_non = resolve_turn_policy(
            mode, ClarifyPhase.INTERVIEW, is_shopping_query=False
        )
        assert p_non.gate_search is False
        assert p_non.prompt_key == PromptKey.NON_SHOPPING

    p_b_open = resolve_turn_policy(
        ClarificationMode.ALWAYS_CLARIFY, ClarifyPhase.OPEN, is_shopping_query=True
    )
    assert p_b_open.gate_search is False
    assert p_b_open.prompt_key == PromptKey.MODE_B_OPEN

    p_c = resolve_turn_policy(
        ClarificationMode.CONDITIONAL_CLARIFICATION, ClarifyPhase.INTERVIEW
    )
    assert p_c.gate_search is False
    assert p_c.prompt_key == PromptKey.MODE_C

    p_legacy = resolve_turn_policy("MINIMAL_CLARIFICATION", ClarifyPhase.INTERVIEW)
    assert p_legacy.gate_search is False
    assert p_legacy.prompt_key == PromptKey.MODE_C


def test_prompt_blocks_cover_all_keys() -> None:
    for key in PromptKey:
        block = get_policy_prompt_block(key)
        assert block
        assert PROMPT_BLOCKS_BY_KEY[key] == block
        assert PROMPT_NAMES_BY_KEY[key]


def test_format_profile_notes() -> None:
    assert format_profile_notes(None) == ""
    assert format_profile_notes(UserProfile()) == ""

    profile = UserProfile(
        height_cm=168.0,
        weight_kg=56.0,
        waist_cm=66.0,
        chest_cm=86.0,
        hip_cm=92.0,
        gender=GenderTarget.FEMALE,
        fit_preference="oversized",
    )
    notes = format_profile_notes(profile)
    assert notes.startswith("\nThông tin khách hàng đã biết")
    for expected in (
        "Chiều cao: 168.0cm",
        "Cân nặng: 56.0kg",
        "Vòng eo: 66.0cm",
        "Vòng ngực: 86.0cm",
        "Vòng mông: 92.0cm",
        "Giới tính: FEMALE",
        "Gu mặc: oversized",
    ):
        assert expected in notes


def test_composable_block_order_and_formulas() -> None:
    profile = UserProfile(gender=GenderTarget.MALE)
    slot_summary = "TÓM TẮT NGỮ CẢNH (PRODUCT_SEARCH):\n- Đã biết: Giới tính=MALE"

    text_a, dbg_a = build_system_prompt_with_debug(
        None, prompt_key=PromptKey.MODE_A, slot_summary=slot_summary
    )
    assert text_a == build_system_prompt(
        None, prompt_key=PromptKey.MODE_A, slot_summary=slot_summary
    )
    assert dbg_a.version == SYSTEM_PROMPT_VERSION
    assert dbg_a.formula == (
        "PERSONA_CORE + MODE_A_DIRECTIVE + PRODUCT_RULES + SUPPORT_RULES + SLOT_SUMMARY"
    )
    assert dbg_a.total_chars == len(text_a)
    assert [s.id for s in dbg_a.sections] == [
        "base",
        "policy",
        "product_rules",
        "support_rules",
        "profile",
        "slot_summary",
    ]
    assert dbg_a.sections[4].included is False
    # Mode directive appears BEFORE product and support rules
    assert text_a.index(MODE_A_DIRECTIVE) < text_a.index(PRODUCT_RULES)
    assert text_a.index(PRODUCT_RULES) < text_a.index(SUPPORT_RULES)

    text_c, dbg_c = build_system_prompt_with_debug(
        profile, prompt_key=PromptKey.MODE_C, slot_summary=slot_summary
    )
    assert dbg_c.formula == (
        "PERSONA_CORE + MODE_C_DIRECTIVE + PRODUCT_RULES + "
        "SUPPORT_RULES + PROFILE_NOTES + SLOT_SUMMARY"
    )
    assert dbg_c.sections[4].included is True
    assert "Giới tính: MALE" in dbg_c.sections[4].content
    assert MODE_C_DIRECTIVE in text_c

    text_open = build_system_prompt(None, prompt_key=PromptKey.MODE_B_OPEN)
    assert MODE_B_OPEN_DIRECTIVE in text_open

    text_non, dbg_non = build_system_prompt_with_debug(
        None, prompt_key=PromptKey.NON_SHOPPING, slot_summary=slot_summary
    )
    assert dbg_non.formula == "PERSONA_CORE + NON_SHOPPING_DIRECTIVE + SUPPORT_RULES"
    assert NON_SHOPPING_DIRECTIVE in text_non
    assert PRODUCT_RULES not in text_non
    assert slot_summary not in text_non


def test_build_system_prompt_fallback_by_clarification_mode() -> None:
    assert MODE_A_DIRECTIVE in build_system_prompt(
        None, clarification_mode=ClarificationMode.IMMEDIATE_ANSWER
    )
    assert PROMPT_B_INTERVIEW in build_system_prompt(
        None, clarification_mode=ClarificationMode.ALWAYS_CLARIFY
    )
    assert MODE_C_DIRECTIVE in build_system_prompt(
        None, clarification_mode=ClarificationMode.CONDITIONAL_CLARIFICATION
    )
    assert MODE_C_DIRECTIVE in build_system_prompt(None)


def test_build_interview_system_prompt() -> None:
    profile = UserProfile(height_cm=175.0)
    slot_summary = "TÓM TẮT NGỮ CẢNH (PRODUCT_SEARCH):\n- Đã biết: Dịp=Đám cưới"

    compact_plain = build_interview_system_prompt(None)
    assert compact_plain == INTERVIEW_SYSTEM_PROMPT
    assert "Quy định bắt buộc:" not in compact_plain

    compact_full, int_dbg = build_interview_system_prompt_with_debug(
        profile, slot_summary=slot_summary
    )
    assert compact_full == (
        f"{INTERVIEW_SYSTEM_PROMPT}{format_profile_notes(profile)}\n{slot_summary}\n"
    )
    assert int_dbg.formula == "INTERVIEW_SYSTEM_PROMPT + PROFILE_NOTES + SLOT_SUMMARY"


def test_prompt_metadata_and_v2_isolation() -> None:
    assert SYSTEM_PROMPT_VERSION == "v2.1"
    assert "Fashion Shop" in EXTRACTOR_SYSTEM_PROMPT
    assert "AgentReply" in EXTRACTOR_SYSTEM_PROMPT

    ext_text, ext_dbg = build_extractor_system_prompt_with_debug()
    assert ext_text == EXTRACTOR_SYSTEM_PROMPT
    assert ext_dbg.formula == "EXTRACTOR_SYSTEM_PROMPT"

    mode_a_prompt = build_system_prompt(None, prompt_key=PromptKey.MODE_A)
    assert "bảng trắc nghiệm" not in mode_a_prompt
    assert "needs_clarification" not in mode_a_prompt
    assert "needs_clarification" not in PERSONA_CORE
    assert "needs_clarification" not in SUPPORT_RULES
    assert "bảng trắc nghiệm" in MODE_C_DIRECTIVE
    assert "cảm ơn để chốt lưu" not in (update_customer_preference.description or "")


def test_active_tools_scoped_by_intent_and_user_id() -> None:
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    agent = ShopAssistantAgent(
        chat_model=fake_model,
        settings=Settings(_env_file=None, OUTFIT_KG_ENABLED=True),
    )

    assert agent._get_active_tools_for_turn(TurnIntent.PRODUCT_SEARCH, True) == []
    assert [
        t.name
        for t in agent._get_active_tools_for_turn(TurnIntent.PRODUCT_SEARCH, False)
    ] == ["search_products", "get_product_detail"]
    assert [
        t.name
        for t in agent._get_active_tools_for_turn(TurnIntent.OUTFIT_ADVICE, False)
    ] == ["get_product_detail", "suggest_outfit"]
    assert [
        t.name for t in agent._get_active_tools_for_turn(TurnIntent.SIZE_ADVICE, False)
    ] == ["search_products", "get_product_detail", "recommend_size"]
    assert [
        t.name
        for t in agent._get_active_tools_for_turn(
            TurnIntent.NON_SHOPPING, False, user_id=None
        )
    ] == ["get_product_detail", "check_voucher", "search_policy"]
    assert [
        t.name
        for t in agent._get_active_tools_for_turn(
            TurnIntent.NON_SHOPPING, False, user_id=42
        )
    ] == [
        "get_product_detail",
        "check_voucher",
        "search_policy",
        "update_customer_preference",
    ]


@pytest.mark.anyio
async def test_agent_run_trace_includes_v2_composable_formula() -> None:
    fake_model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Shop có nhiều mẫu áo sơ mi nữ đẹp ạ!")])
    )
    agent = ShopAssistantAgent(chat_model=fake_model)
    res = await agent.chat(
        ChatRequest(
            message="Tìm áo sơ mi nữ công sở",
            session_id="test-sys-prompt-debug-v2",
            clarification_mode=ClarificationMode.IMMEDIATE_ANSWER,
        )
    )
    assert res.debug_meta is not None
    assert len(res.debug_meta.runs) >= 1
    first_run = res.debug_meta.runs[0]
    assert first_run.system_prompt is not None
    assert first_run.system_prompt.version == "v2.1"
    assert first_run.system_prompt.formula == (
        "PERSONA_CORE + MODE_A_DIRECTIVE + PRODUCT_RULES + "
        "SUPPORT_RULES + PROFILE_NOTES + SLOT_SUMMARY"
    )
    assert "system_prompt_debug" not in first_run.metadata
