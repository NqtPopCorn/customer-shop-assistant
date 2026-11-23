import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.llm.callbacks import LangChainDebugCallbackHandler
from customer_shop_assistant.llm.factory import get_chat_model
from customer_shop_assistant.schemas.chat import (
    AgentReply,
    ChatRecommendedProduct,
    ChatRequest,
    ChatResponse,
    ClarificationForm,
    ClarificationMode,
    StateSnapshotDebug,
    TurnDebugMeta,
    UserProfile,
)
from customer_shop_assistant.services.clarification_planner import (
    align_reply_with_clarification_form as planner_align_reply,
)
from customer_shop_assistant.services.clarification_planner import (
    build_clarification_form,
)
from customer_shop_assistant.services.clarification_planner import (
    validate_or_repair_clarification_form as planner_validate_or_repair,
)
from customer_shop_assistant.services.clarification_policy import (
    PROMPT_A,
    PROMPT_B_INTERVIEW,
    PROMPT_C,
    TurnPolicy,
    next_phase,
    resolve_turn_policy,
)
from customer_shop_assistant.services.session_manager import (
    ClarifyPhase,
    SessionManager,
    SessionState,
    get_session_manager,
)
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.services.slot_tracker import (
    ConversationSlots,
    TurnIntent,
    extract_conversation_slots,
    extract_session_profile_updates,
    format_slots_for_prompt,
)
from customer_shop_assistant.services.telemetry_logger import log_chat_turn
from customer_shop_assistant.tools import ALL_TOOLS
from customer_shop_assistant.tools.catalog_tool import (
    get_cached_product,
    get_latest_searched_product_ids,
    product_schema_to_recommended,
    product_turn_context,
    record_product,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý thời trang thông minh của Fashion Shop, \
chuyên tư vấn và gợi ý sản phẩm phù hợp nhất cho khách hàng.

Phong cách giao tiếp:
- Tinh gọn, trọng tâm, gu thẩm mỹ tốt, đúng sự thật khách quan (fact-first).
- Trả lời bằng tiếng Việt lịch sự, thân thiện, nhiệt tình gợi ý phối đồ.
- Sử dụng cú pháp Markdown chuẩn (xuống dòng tự nhiên, gạch đầu dòng, in đậm).
  Tuyệt đối KHÔNG viết các thẻ HTML như <br>, <p> vào câu trả lời.

Quy định bắt buộc:
1. GỢI Ý SẢN PHẨM (TRỌNG TÂM HÀNG ĐẦU):
   - Khi đã đủ thông tin để tìm sản phẩm theo chế độ làm rõ hiện tại, nếu khách
     hỏi tìm đồ hoặc gợi ý trang phục:
     * BẮT BUỘC gọi tool `search_products` để tìm sản phẩm thực tế trong cửa hàng.
     * BẮT BUỘC GIỚI THIỆU NGAY 2-4 sản phẩm tìm được vào câu trả lời kèm mã `[ID]`
       (ví dụ: `[57] Đầm Dạ Hội Ren`) và nêu bật lý do thiết kế, chất vải, form dáng.
     * KHÔNG chặn khách để đòi số đo P3 khi khách chỉ đang tìm sản phẩm.
     * Đơn vị tiền là VNĐ. Nếu nói '300k', '400k' thì truyền vào tool là 300000, 400000.
   - Điền vào `product_ids` danh sách ID các sản phẩm thực sự được gợi ý trong \
câu trả lời (ví dụ: ['57', '27']). Trả về danh sách rỗng nếu không gợi ý sản phẩm.
   - Tuyệt đối không tự bịa đặt tên sản phẩm, mã hàng hay trạng thái tồn kho.
2. TƯ VẤN SIZE (HỖ TRỢ KÈM THEO):
   - CHỈ HỎI XIN SỐ ĐO khi khách CHỦ ĐỘNG HỎI VỀ SIZE mà chưa có số đo \
(ví dụ: 'mình mặc size gì?', 'tư vấn size', 'bảng size'). \
Khi đó đặt `needs_clarification = True`.
   - Khi đã có số đo, hãy dùng `recommend_size` để tính toán size chuẩn \
theo bảng quy đổi.
3. CHÍNH SÁCH BÁN HÀNG:
   - Dùng `search_policy` khi khách hỏi về đổi trả, phí vận chuyển, freeship, \
thanh toán VietQR/COD. Bắt buộc trích dẫn nguồn `[tên_tệp: tiêu_đề_mục]`.
   - Nếu tài liệu không chứa thông tin, hãy nói rõ là cửa hàng chưa có thông tin \
về vấn đề này. Tuyệt đối không tự suy diễn hoặc bịa đặt chính sách ngoài tài liệu.
4. HỒ SƠ & LƯU THÔNG TIN CÁ NHÂN:
   - Toàn bộ số đo và sở thích của khách CHỈ ĐƯỢC LƯU TẠM THỜI trong phiên hiện tại \
để gợi ý đồ. TUYỆT ĐỐI KHÔNG tự động lưu vĩnh viễn vào tài khoản.
   - CHỈ gọi tool `update_customer_preference` khi khách hàng CHỦ ĐỘNG YÊU CẦU LƯU \
(ví dụ: 'lưu lại số đo này giúp mình', 'nhớ số đo này cho mình nhé') \
hoặc ĐỒNG Ý RÕ RÀNG.
   - KHI KHÁCH NÓI CẢM ƠN (ví dụ: 'cảm ơn shop', 'thanks', 'cảm ơn bạn nhé'): \
Chỉ đáp lại lời cảm ơn một cách thân thiện, lịch sự; \
TUYỆT ĐỐI KHÔNG gọi tool `update_customer_preference`.
   - Đối với khách vãng lai (chưa có user_id): Không gọi tool lưu, phản hồi \
cảm ơn lịch sự và gợi ý khách đăng nhập nếu muốn lưu hồ sơ lâu dài.
5. CÂU HỎI NGOÀI PHẠM VI:
   - Nếu câu hỏi không liên quan đến thời trang hay mua sắm tại shop, \
hãy từ chối ngắn gọn, lịch sự và hướng khách về việc chọn đồ. Đặt `product_ids = []`.
6. FORM LÀM RÕ:
   - Khi cần hỏi làm rõ (needs_clarification=True): phần `reply` chỉ viết 1-2 câu \
dẫn ngắn gọn, lịch sự. Tuyệt đối KHÔNG viết danh sách câu hỏi gạch đầu dòng hay \
bảng trắc nghiệm vào văn bản `reply`.
"""


def detect_inquiry_slots(
    user_message: str = "",
    user_profile: UserProfile | None = None,
) -> dict[str, Any]:
    """Backwards-compatible wrapper delegating to multi-turn slot tracker."""
    msgs = [user_message] if user_message else []
    slots, intent = extract_conversation_slots(msgs, user_profile)
    return slots.to_debug_dict(intent)


def build_default_clarification_form(
    user_message: str = "",
    user_profile: UserProfile | None = None,
) -> ClarificationForm:
    """Backwards-compatible wrapper delegating to dynamic clarification planner."""
    msgs = [user_message] if user_message else []
    slots, intent = extract_conversation_slots(msgs, user_profile)
    return build_clarification_form(slots, intent, max_questions=3)


class AgentState(TypedDict):
    """LangGraph conversation state with automatic message appending."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: int | None
    user_profile: UserProfile | None
    clarification_mode: ClarificationMode
    clarify_phase: ClarifyPhase
    gated_turns: int
    turn_intent: str
    is_shopping_query: bool
    tools_used: list[str]
    reply: str
    needs_clarification: bool
    clarification_form: ClarificationForm | None
    product_ids: list[str]
    products: list[ChatRecommendedProduct]


class ShopAssistantAgent:
    """Orchestrates chatbot conversation flow using LangGraph and LangChain."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chat_model = chat_model or get_chat_model(self.settings)
        self.session_manager = session_manager
        self.tools_by_name = {t.name: t for t in ALL_TOOLS}

        try:
            self.model_with_tools = self.chat_model.bind_tools(ALL_TOOLS)
        except (NotImplementedError, AttributeError):
            self.model_with_tools = self.chat_model

        try:
            self.structured_model = self.chat_model.with_structured_output(AgentReply)
        except (NotImplementedError, AttributeError):
            self.structured_model = None

        self.graph = self._create_graph()

    def _build_system_prompt(
        self,
        user_profile: UserProfile | None,
        policy_prompt_block: str = "",
        clarification_mode: ClarificationMode | None = None,
    ) -> str:
        """Construct system prompt with persona, profile, and clarification rules."""
        sys_content = SYSTEM_PROMPT

        if user_profile:
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
            if prof_notes:
                sys_content += (
                    "\nThông tin khách hàng đã biết (áp dụng để gợi ý sản phẩm "
                    f"& chọn size): {', '.join(prof_notes)}"
                )

        if policy_prompt_block:
            sys_content += f"\n{policy_prompt_block}\n"
        elif clarification_mode == ClarificationMode.IMMEDIATE_ANSWER:
            sys_content += f"\n{PROMPT_A}\n"
        elif clarification_mode == ClarificationMode.ALWAYS_CLARIFY:
            sys_content += f"\n{PROMPT_B_INTERVIEW}\n"
        else:
            sys_content += f"\n{PROMPT_C}\n"

        return sys_content

    def _create_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph workflow with tool execution and memory."""
        builder = StateGraph(AgentState)

        async def agent_node(
            state: AgentState, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            is_shopping_query = state.get("is_shopping_query", True)
            policy = resolve_turn_policy(
                mode, phase, is_shopping_query=is_shopping_query
            )

            if policy.gate_search:
                active_tools = [t for t in ALL_TOOLS if t.name != "search_products"]
            else:
                active_tools = list(ALL_TOOLS)

            try:
                bound_model = self.chat_model.bind_tools(active_tools)
            except (NotImplementedError, AttributeError):
                bound_model = self.chat_model

            slots, intent = extract_conversation_slots(
                state["messages"], state.get("user_profile")
            )
            include_missing = mode != ClarificationMode.IMMEDIATE_ANSWER
            slot_summary = format_slots_for_prompt(
                slots, intent, include_missing=include_missing
            )

            policy_block = policy.prompt_block
            if policy.gate_search:
                planned_form = build_clarification_form(slots, intent, max_questions=3)
                q_summary = " | ".join(q.question for q in planned_form.questions)
                policy_block = (
                    f"{policy_block}\n"
                    f"{slot_summary}\n"
                    "BẢNG TRẮC NGHIỆM ĐÃ HIỂN THỊ SẴN BÊN DƯỚI GỒM CÁC CÂU HỎI: "
                    f"[{q_summary}].\n"
                    "YÊU CẦU BẮT BUỘC: Trong câu trả lời của bạn, CHỈ viết đúng 1 câu "
                    "chào và dẫn nhập ngắn gọn mời khách chọn các mục trong bảng "
                    "bên dưới. Không lặp lại danh sách gạch đầu dòng."
                )
            elif is_shopping_query:
                policy_block = f"{policy_block}\n{slot_summary}"

            sys_prompt = self._build_system_prompt(
                user_profile=state.get("user_profile"),
                policy_prompt_block=policy_block,
                clarification_mode=mode,
            )

            trimmed = trim_messages(
                state["messages"],
                max_tokens=self.settings.MAX_SESSION_HISTORY_TURNS * 2,
                token_counter=len,
                strategy="last",
                start_on="human",
                include_system=False,
            )
            full_prompt_msgs = [SystemMessage(content=sys_prompt), *trimmed]

            response = await bound_model.ainvoke(full_prompt_msgs, config=config)
            return {"messages": [response]}

        async def tools_node(
            state: AgentState, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", [])
            tool_messages: list[ToolMessage] = []
            tools_used = list(state.get("tools_used") or [])

            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            is_shopping_query = state.get("is_shopping_query", True)
            policy = resolve_turn_policy(
                mode, phase, is_shopping_query=is_shopping_query
            )

            active_tool_names = (
                {t.name for t in ALL_TOOLS if t.name != "search_products"}
                if policy.gate_search
                else {t.name for t in ALL_TOOLS}
            )

            for tc in tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc.get("id") or str(uuid.uuid4())

                if t_name not in active_tool_names:
                    logger.warning(
                        "Tool '%s' is gated in phase %s, returning polite error",
                        t_name,
                        phase,
                    )
                    t_result = (
                        f"Công cụ '{t_name}' hiện không khả dụng trong lượt này. "
                        "Vui lòng tương tác hoặc hỏi thêm thông tin từ khách hàng."
                    )
                else:
                    if t_name not in tools_used:
                        tools_used.append(t_name)
                    tool = self.tools_by_name.get(t_name)
                    if tool:
                        try:
                            t_result = await tool.ainvoke(t_args, config=config)
                        except Exception as t_err:
                            logger.error("Error executing tool %s: %s", t_name, t_err)
                            t_result = f"Error executing tool {t_name}: {t_err}"
                    else:
                        t_result = f"Tool {t_name} not found."

                tool_messages.append(
                    ToolMessage(content=str(t_result), tool_call_id=t_id)
                )

            return {"messages": tool_messages, "tools_used": tools_used}

        def route_tools(state: AgentState) -> str:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "tools"
            return "finalize"

        async def finalize_node(
            state: AgentState, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            last_text = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_text = str(msg.content or "")
                    break

            slots, intent = extract_conversation_slots(
                state["messages"], state.get("user_profile")
            )
            is_shopping_query = intent != TurnIntent.NON_SHOPPING
            tools_used = list(state.get("tools_used") or [])
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            gated_turns = state.get("gated_turns", 0)
            allow_clarification = not (
                mode == ClarificationMode.IMMEDIATE_ANSWER
                and intent != TurnIntent.SIZE_ADVICE
            )

            agent_reply = await self._generate_agent_reply(
                state["messages"],
                last_reply_text=last_text,
                is_shopping_query=is_shopping_query,
                tools_used=tools_used,
                allow_clarification=allow_clarification,
                config=config,
            )

            searched_ids = (
                get_latest_searched_product_ids()
                if "search_products" in tools_used
                else None
            )
            products = await self._resolve_products(
                agent_reply.product_ids,
                reply_text=last_text or agent_reply.reply,
                fallback_product_ids=searched_ids,
            )
            agent_reply.product_ids = [str(p.id) for p in products]

            is_broad_empty_inquiry = (
                not slots.has("occasion")
                and not slots.has("category")
                and not slots.has("budget")
                and not slots.has("measurements")
                and not slots.has("body_concern")
            )

            if (
                mode == ClarificationMode.IMMEDIATE_ANSWER
                and intent != TurnIntent.SIZE_ADVICE
            ):
                lower_reply = (last_text or agent_reply.reply).lower()
                llm_tried_to_clarify = (
                    agent_reply.needs_clarification
                    or "?" in (last_text or agent_reply.reply)
                    or any(
                        cue in lower_reply
                        for cue in (
                            "đối tượng nào",
                            "nam hay nữ",
                            "cho mình biết",
                            "cho shop biết",
                            "cần biết thêm",
                            "tư vấn chính xác",
                        )
                    )
                )
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None

                if (
                    is_shopping_query
                    and llm_tried_to_clarify
                    and not products
                    and not tools_used
                ):
                    search_tool = self.tools_by_name.get("search_products")
                    if search_tool is not None:
                        fallback_args = self._build_immediate_search_args(
                            slots, state["messages"]
                        )
                        try:
                            tool_res = await search_tool.ainvoke(
                                fallback_args, config=config
                            )
                            if "search_products" not in tools_used:
                                tools_used.append("search_products")
                            products = await self._resolve_products(
                                [],
                                reply_text=str(tool_res),
                                fallback_product_ids=(
                                    get_latest_searched_product_ids()
                                ),
                            )
                            agent_reply.product_ids = [str(p.id) for p in products]
                            if products:
                                agent_reply.reply = (
                                    self._format_immediate_fallback_reply(
                                        products, slots
                                    )
                                )
                        except Exception as fallback_err:
                            logger.warning(
                                "Immediate fallback search failed: %s",
                                fallback_err,
                            )
            elif (
                mode == ClarificationMode.ALWAYS_CLARIFY
                and phase == ClarifyPhase.INTERVIEW
                and is_shopping_query
            ):
                agent_reply.needs_clarification = True
            elif (
                mode == ClarificationMode.CONDITIONAL_CLARIFICATION
                and is_shopping_query
                and is_broad_empty_inquiry
                and not products
                and not tools_used
            ):
                agent_reply.needs_clarification = True

            agent_reply.reply = agent_reply.reply.strip()

            if agent_reply.needs_clarification:
                force_planner = (
                    mode == ClarificationMode.ALWAYS_CLARIFY
                    and phase == ClarifyPhase.INTERVIEW
                    and is_shopping_query
                )
                agent_reply.clarification_form = planner_validate_or_repair(
                    agent_reply.clarification_form,
                    slots=slots,
                    intent=intent,
                    force_planner=force_planner,
                    max_fallback_questions=2,
                )
                agent_reply.reply = planner_align_reply(
                    agent_reply.reply,
                    agent_reply.clarification_form,
                    slots=slots,
                    intent=intent,
                )

            new_phase = next_phase(
                mode=mode,
                current_phase=phase,
                gated_turns=gated_turns,
                needs_clarification=agent_reply.needs_clarification,
                is_shopping_query=is_shopping_query,
            )

            session_id = state["session_id"]
            sess_mgr = self.session_manager or get_session_manager(
                ttl_seconds=self.settings.SESSION_TTL_SECONDS,
                default_max_turns=self.settings.MAX_SESSION_HISTORY_TURNS,
            )
            sess = sess_mgr.get_session(session_id)
            if sess:
                sess.clarification_mode = str(
                    mode.value if hasattr(mode, "value") else mode
                )
                sess.clarify_phase = new_phase
                sess.gated_turns = gated_turns
                sess_mgr.add_assistant_message(session_id, agent_reply.reply)

            return {
                "reply": agent_reply.reply,
                "needs_clarification": agent_reply.needs_clarification,
                "clarification_form": agent_reply.clarification_form,
                "product_ids": agent_reply.product_ids,
                "products": products,
                "tools_used": tools_used,
                "clarify_phase": new_phase,
            }

        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)
        builder.add_node("finalize", finalize_node)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", route_tools, {"tools": "tools", "finalize": "finalize"}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=MemorySaver())

    async def _resolve_turn_session(
        self,
        request: ChatRequest,
        user_id: int | None = None,
        auth_token: str | None = None,
    ) -> tuple[
        str,
        SessionManager,
        SessionState,
        ClarificationMode,
        TurnPolicy,
        ConversationSlots,
        TurnIntent,
    ]:
        """Resolve and initialize session, profile, slots, and turn policy."""
        session_id = request.session_id or str(uuid.uuid4())
        session_mgr = self.session_manager or get_session_manager(
            ttl_seconds=self.settings.SESSION_TTL_SECONDS,
            default_max_turns=self.settings.MAX_SESSION_HISTORY_TURNS,
        )
        session = session_mgr.get_or_create_session(session_id, user_id=user_id)

        raw_mode = (
            request.clarification_mode or self.settings.DEFAULT_CLARIFICATION_MODE
        )
        if isinstance(raw_mode, str):
            mode_clean = raw_mode.strip().upper()
            if mode_clean in (
                ClarificationMode.CONDITIONAL_CLARIFICATION,
                "MINIMAL_CLARIFICATION",
            ):
                mode = ClarificationMode.CONDITIONAL_CLARIFICATION
            elif mode_clean in ClarificationMode.__members__:
                mode = ClarificationMode[mode_clean]
            else:
                mode = ClarificationMode.CONDITIONAL_CLARIFICATION
        else:
            mode = raw_mode

        if session.clarification_mode != str(
            mode.value if hasattr(mode, "value") else mode
        ):
            session.reset_clarification(
                str(mode.value if hasattr(mode, "value") else mode)
            )

        if user_id is not None and session.active_profile is None:
            try:
                async with ShopClient(settings=self.settings) as client:
                    pref_data = await client.get_user_preferences(
                        user_id=user_id, token=auth_token
                    )
                    if pref_data:
                        h = pref_data.get("heightCm")
                        w = pref_data.get("weightKg")
                        waist = pref_data.get("waistCm")
                        chest = pref_data.get("chestCm")
                        hip = pref_data.get("hipCm")
                        gender = pref_data.get("gender")
                        fit = pref_data.get("fitPreference")
                        if any([h, w, waist, chest, hip, gender, fit]):
                            session.active_profile = UserProfile(
                                height_cm=h,
                                weight_kg=w,
                                waist_cm=waist,
                                chest_cm=chest,
                                hip_cm=hip,
                                gender=gender,
                                fit_preference=fit,
                            )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch preferences for user #%s: %s", user_id, exc
                )

        if request.user_profile is not None:
            session.update_active_profile(request.user_profile)

        if not session.messages and request.history:
            for item in request.history:
                if (
                    item.role == "user"
                    and item.content.strip() == request.message.strip()
                ):
                    continue
                if item.role == "assistant":
                    session.messages.append(AIMessage(content=item.content))
                else:
                    session.messages.append(HumanMessage(content=item.content))

        if not (
            session.messages
            and isinstance(session.messages[-1], HumanMessage)
            and session.messages[-1].content.strip() == request.message.strip()
        ):
            session_mgr.add_user_message(session_id, request.message)

        slots, intent = extract_conversation_slots(
            session.messages, session.active_profile
        )
        profile_updates = extract_session_profile_updates(slots)
        if profile_updates:
            session.update_active_profile(profile_updates)

        is_shopping_query = intent != TurnIntent.NON_SHOPPING
        policy = resolve_turn_policy(
            mode, session.clarify_phase, is_shopping_query=is_shopping_query
        )
        if policy.gate_search:
            session.gated_turns += 1

        return session_id, session_mgr, session, mode, policy, slots, intent

    @staticmethod
    def _build_immediate_search_args(
        slots: ConversationSlots, messages: list[BaseMessage]
    ) -> dict[str, Any]:
        """Build search_products tool arguments for IMMEDIATE_ANSWER fallback."""
        cat_kw_map = {
            "shirt": "áo sơ mi",
            "polo": "áo polo",
            "tshirt": "áo thun",
            "jacket": "áo khoác",
            "tops": "áo",
            "jeans": "quần jeans",
            "kaki": "quần kaki",
            "short": "quần short",
            "pants": "quần",
            "dress": "đầm",
            "skirt": "chân váy",
            "accessory": "phụ kiện",
        }
        occ_kw_map = {
            "party": "dự tiệc",
            "office": "công sở",
            "casual": "dạo phố",
            "sport": "thể thao",
        }
        kw_parts: list[str] = []
        cat_val = slots.value_of("category")
        if cat_val in cat_kw_map:
            kw_parts.append(cat_kw_map[cat_val])
        occ_val = slots.value_of("occasion")
        if occ_val in occ_kw_map:
            kw_parts.append(occ_kw_map[occ_val])
        if slots.has("color"):
            kw_parts.append(str(slots.value_of("color")).lower())

        if kw_parts:
            keyword = " ".join(kw_parts)
        else:
            latest_text = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    latest_text = str(m.content or "").strip()
                    break
            keyword = latest_text or "thời trang"

        args: dict[str, Any] = {"keyword": keyword, "limit": 4}
        if slots.has("target_gender"):
            args["target_gender"] = slots.value_of("target_gender")
        budget_val = slots.value_of("budget")
        if budget_val is not None:
            min_p = getattr(budget_val, "min_price", None)
            max_p = getattr(budget_val, "max_price", None)
            if min_p is not None:
                args["min_price"] = min_p
            if max_p is not None:
                args["max_price"] = max_p
        return args

    @staticmethod
    def _format_immediate_fallback_reply(
        products: list[ChatRecommendedProduct], slots: ConversationSlots
    ) -> str:
        """Format a direct fact-first recommendation list with clear assumption."""
        if not slots.has("target_gender"):
            intro = (
                "Vì bạn chưa nêu cụ thể đối tượng mặc nên mình gợi ý ngay "
                "các mẫu tiêu biểu phù hợp nhất hiện có tại cửa hàng:"
            )
        else:
            intro = "Dưới đây là các mẫu trang phục phù hợp hiện có tại cửa hàng:"
        lines = [intro]
        for p in products[:4]:
            price_txt = f"{p.price:,.0f}đ" if p.price > 0 else "Liên hệ"
            lines.append(f"- [{p.id}] **{p.name}** — Giá: {price_txt}")
        return "\n".join(lines)

    async def _generate_agent_reply(
        self,
        turn_messages: list[BaseMessage],
        last_reply_text: str = "",
        is_shopping_query: bool = True,
        tools_used: list[str] | None = None,
        allow_clarification: bool = True,
        config: RunnableConfig | None = None,
    ) -> AgentReply:
        """Obtain AgentReply via structured output or fallback parsing."""
        welcome_phrases = (
            "mình có thể giúp bạn tìm kiếm trang phục",
            "bạn đang quan tâm đến phong cách nào hôm nay",
            "mình là stylist ai của fashion shop",
        )
        used_set = set(tools_used or [])

        if self.structured_model is not None:
            try:
                recent_msgs = [
                    m
                    for m in turn_messages
                    if isinstance(m, (HumanMessage, ToolMessage))
                    or (isinstance(m, AIMessage) and m.content == last_reply_text)
                ][-3:]
                sys_extractor = SystemMessage(
                    content=(
                        "Bạn là bộ trích xuất thông tin của Fashion Shop. "
                        "Hãy chuyển phản hồi của lượt hiện tại thành cấu trúc "
                        "AgentReply. Tuyệt đối không lặp lại câu chào hỏi ban đầu."
                    )
                )
                eval_msgs: list[BaseMessage] = (
                    [sys_extractor, *recent_msgs] if recent_msgs else turn_messages
                )

                result = await self.structured_model.ainvoke(eval_msgs, config=config)
                reply_obj: AgentReply | None = None
                if isinstance(result, AgentReply):
                    reply_obj = result
                elif isinstance(result, dict):
                    reply_obj = AgentReply.model_validate(result)

                if reply_obj is not None:
                    is_echo = any(
                        wp in reply_obj.reply.lower() for wp in welcome_phrases
                    )
                    if is_echo:
                        logger.warning(
                            "Structured model echoed welcome greeting; "
                            "falling back to last_reply_text"
                        )
                        reply_obj.reply = last_reply_text or ""
                    elif not reply_obj.reply and last_reply_text:
                        reply_obj.reply = last_reply_text

                    if not reply_obj.product_ids and last_reply_text:
                        for match in re.findall(
                            r"(?:ID:\s*\[|\[)(\d+)\]", last_reply_text
                        ):
                            if match not in reply_obj.product_ids:
                                reply_obj.product_ids.append(match)

                    if (
                        allow_clarification
                        and not reply_obj.needs_clarification
                        and is_shopping_query
                        and not reply_obj.product_ids
                        and "search_products" not in used_set
                        and last_reply_text
                    ):
                        lower_txt = last_reply_text.lower()
                        if "?" in last_reply_text or any(
                            cue in lower_txt
                            for cue in (
                                "bảng câu hỏi",
                                "trắc nghiệm",
                                "lựa chọn",
                                "tiêu chí",
                                "để tư vấn chính xác",
                            )
                        ):
                            reply_obj.needs_clarification = True

                    return reply_obj
            except Exception as exc:
                logger.warning(
                    "Structured output invocation failed (%s), falling back to parsing",
                    exc,
                )

        if last_reply_text:
            cleaned = last_reply_text.strip()
            if "```" in cleaned:
                json_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL
                )
                if json_match:
                    cleaned = json_match.group(1)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "reply" in parsed:
                    return AgentReply.model_validate(parsed)
            except Exception:
                pass

        if not last_reply_text:
            if get_latest_searched_product_ids():
                text = (
                    "Dưới đây là một số mẫu trang phục phù hợp với nhu cầu và "
                    "phong cách của bạn:"
                )
            else:
                text = "Mình có thể hỗ trợ gì thêm cho bạn không?"
        else:
            text = last_reply_text

        found_pids: list[str] = []
        for match in re.findall(r"(?:ID:\s*\[|\[)(\d+)\]", text):
            if match not in found_pids:
                found_pids.append(match)

        is_clarification = (
            allow_clarification
            and is_shopping_query
            and not found_pids
            and "search_products" not in used_set
            and (
                "?" in text
                or any(
                    cue in text.lower()
                    for cue in (
                        "vui lòng cho",
                        "vui lòng cung cấp",
                        "cho mình biết",
                        "cho shop biết",
                        "cho shop xin",
                        "cần biết",
                        "chia sẻ thêm",
                    )
                )
            )
            and any(
                w in text.lower()
                for w in (
                    "chiều cao",
                    "cân nặng",
                    "số đo",
                    "nam hay nữ",
                    "bạn tìm",
                    "dịp",
                    "phong cách",
                    "ngân sách",
                    "gu",
                    "sở thích",
                    "màu sắc",
                )
            )
        )

        return AgentReply(
            reply=text,
            needs_clarification=is_clarification,
            product_ids=found_pids,
        )

    async def _resolve_products(
        self,
        product_ids: list[str],
        reply_text: str = "",
        fallback_product_ids: list[int] | None = None,
    ) -> list[ChatRecommendedProduct]:
        """Resolve products by IDs, extracting from text and fetching if missing."""
        candidate_ids = list(product_ids)

        for match in re.findall(r"(?:ID:\s*\[|\[)(\d+)\]", reply_text):
            if match not in candidate_ids:
                candidate_ids.append(match)

        if not candidate_ids and fallback_product_ids:
            candidate_ids = [str(pid) for pid in fallback_product_ids[:4]]

        resolved: list[ChatRecommendedProduct] = []
        seen_ids: set[int] = set()

        for pid_str in candidate_ids:
            try:
                pid = int(pid_str)
            except (ValueError, TypeError):
                continue
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            prod = get_cached_product(pid)
            if prod:
                if isinstance(prod, ChatRecommendedProduct):
                    resolved.append(prod)
                else:
                    resolved.append(product_schema_to_recommended(prod))
            else:
                try:
                    async with ShopClient(settings=self.settings) as client:
                        p_schema = await client.get_product_by_id(pid)
                        if p_schema:
                            rec_prod = product_schema_to_recommended(p_schema)
                            record_product(rec_prod)
                            resolved.append(rec_prod)
                except Exception:
                    pass

        return resolved

    async def chat(
        self,
        request: ChatRequest,
        user_id: int | None = None,
        auth_token: str | None = None,
    ) -> ChatResponse:
        """Process customer request through the LangGraph conversational workflow."""
        start_time = time.time()
        try:
            (
                session_id,
                session_mgr,
                session,
                mode,
                policy,
                slots,
                intent,
            ) = await self._resolve_turn_session(
                request=request, user_id=user_id, auth_token=auth_token
            )

            history_msgs = session_mgr.get_messages(
                session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
            )
            is_shopping_query = intent != TurnIntent.NON_SHOPPING

            initial_state: AgentState = {
                "messages": history_msgs,
                "session_id": session_id,
                "user_id": user_id,
                "user_profile": session.active_profile,
                "clarification_mode": mode,
                "clarify_phase": session.clarify_phase,
                "gated_turns": session.gated_turns,
                "turn_intent": intent.value,
                "is_shopping_query": is_shopping_query,
                "tools_used": [],
                "reply": "",
                "needs_clarification": False,
                "clarification_form": None,
                "product_ids": [],
                "products": [],
            }

            debug_handler = LangChainDebugCallbackHandler()
            config: RunnableConfig = {
                "configurable": {"thread_id": session_id},
                "callbacks": [debug_handler],
            }
            with product_turn_context():
                result_state = await self.graph.ainvoke(initial_state, config=config)

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "CHAT_LANGGRAPH_METRICS session_id=%s mode=%s phase=%s "
                "tools=%s needs_clarification=%s products_count=%d latency_ms=%d",
                session_id[:8],
                mode.value if hasattr(mode, "value") else mode,
                session.clarify_phase.value,
                result_state.get("tools_used", []),
                result_state.get("needs_clarification", False),
                len(result_state.get("products", [])),
                latency_ms,
            )

            runs = debug_handler.get_runs()
            usage_meta = debug_handler.get_aggregated_usage()
            active_tools_names = [
                t.name
                for t in (
                    [t for t in ALL_TOOLS if t.name != "search_products"]
                    if policy.gate_search
                    else list(ALL_TOOLS)
                )
            ]
            detected_slots = slots.to_debug_dict(intent)
            state_snapshot = StateSnapshotDebug(
                clarify_phase=session.clarify_phase.value,
                gated_turns=session.gated_turns,
                tools_available=active_tools_names,
                gate_search=policy.gate_search,
                active_profile=(
                    session.active_profile.model_dump()
                    if session.active_profile
                    else None
                ),
                detected_slots=detected_slots,
                context_messages_count=len(session_mgr.get_messages(session_id)),
            )

            debug_meta = None
            if self.settings.DEBUG_CHAT_META:
                debug_meta = TurnDebugMeta(
                    mode=mode.value if hasattr(mode, "value") else mode,
                    phase=session.clarify_phase.value,
                    gated_turns=session.gated_turns,
                    tools_available=active_tools_names,
                    latency_ms=latency_ms,
                    usage_metadata=usage_meta,
                    runs=runs,
                    state_snapshot=state_snapshot,
                )

            turn_id = (len(session.messages) // 2) + 1
            await log_chat_turn(
                session_id=session_id,
                turn_id=turn_id,
                user_message=request.message,
                assistant_message=result_state.get("reply", ""),
                latency_ms=latency_ms,
                token_usage=usage_meta,
                error=None,
            )

            return ChatResponse(
                reply=result_state.get("reply", ""),
                session_id=session_id,
                intent=intent.value,
                clarification_asked=result_state.get("needs_clarification", False),
                clarification_form=result_state.get("clarification_form"),
                tools_used=result_state.get("tools_used", []),
                products=result_state.get("products", []),
                debug_meta=debug_meta,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("Error running agent graph: %s", e, exc_info=True)
            turn_id = (len(session.messages) // 2) + 1 if "session" in locals() else 1
            err_sess_id = request.session_id or str(uuid.uuid4())
            await log_chat_turn(
                session_id=err_sess_id,
                turn_id=turn_id,
                user_message=request.message,
                assistant_message=None,
                latency_ms=latency_ms,
                token_usage=None,
                error=str(e),
            )
            return ChatResponse(
                reply=(
                    "Hệ thống tư vấn đang gặp gián đoạn tạm thời. "
                    "Bạn vui lòng gửi lại câu hỏi sau giây lát nhé!"
                ),
                session_id=request.session_id or str(uuid.uuid4()),
                intent="ERROR",
                clarification_asked=False,
                clarification_form=None,
                tools_used=[],
                products=[],
            )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_id: int | None = None,
        auth_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream assistant response tokens and events in real time."""
        start_time = time.time()
        (
            session_id,
            session_mgr,
            session,
            mode,
            policy,
            slots,
            intent,
        ) = await self._resolve_turn_session(
            request=request, user_id=user_id, auth_token=auth_token
        )

        history_msgs = session_mgr.get_messages(
            session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
        )
        is_shopping_query = intent != TurnIntent.NON_SHOPPING

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": session.active_profile,
            "clarification_mode": mode,
            "clarify_phase": session.clarify_phase,
            "gated_turns": session.gated_turns,
            "turn_intent": intent.value,
            "is_shopping_query": is_shopping_query,
            "tools_used": [],
            "reply": "",
            "needs_clarification": False,
            "clarification_form": None,
            "product_ids": [],
            "products": [],
        }

        debug_handler = LangChainDebugCallbackHandler()
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": [debug_handler],
        }
        tools_used: list[str] = []
        full_reply_parts: list[str] = []

        try:
            with product_turn_context():
                async for chunk_mode, payload in self.graph.astream(
                    initial_state,
                    config=config,
                    stream_mode=["messages", "updates"],
                ):
                    if chunk_mode == "messages":
                        msg, meta = payload
                        if meta.get("langgraph_node") == "agent" and getattr(
                            msg, "content", None
                        ):
                            text = str(msg.content)
                            full_reply_parts.append(text)
                            yield {
                                "type": "token",
                                "content": text,
                                "session_id": session_id,
                            }
                    elif chunk_mode == "updates":
                        if "tools" in payload:
                            new_tools = payload["tools"].get("tools_used", [])
                            for t in new_tools:
                                if t not in tools_used:
                                    tools_used.append(t)
                                    yield {
                                        "type": "tool_start",
                                        "tool": t,
                                        "session_id": session_id,
                                    }
                        if "finalize" in payload:
                            fin_data = payload["finalize"]
                            for t in fin_data.get("tools_used", []):
                                if t not in tools_used:
                                    tools_used.append(t)
                            final_reply = (
                                fin_data.get("reply")
                                or "".join(full_reply_parts).strip()
                            ).strip()

                            latency_ms = int((time.time() - start_time) * 1000)
                            runs = debug_handler.get_runs()
                            usage_meta = debug_handler.get_aggregated_usage()
                            active_tools_names = [
                                t.name
                                for t in (
                                    [
                                        t
                                        for t in ALL_TOOLS
                                        if t.name != "search_products"
                                    ]
                                    if policy.gate_search
                                    else list(ALL_TOOLS)
                                )
                            ]
                            detected_slots = slots.to_debug_dict(intent)
                            state_snapshot = StateSnapshotDebug(
                                clarify_phase=session.clarify_phase.value,
                                gated_turns=session.gated_turns,
                                tools_available=active_tools_names,
                                gate_search=policy.gate_search,
                                active_profile=(
                                    session.active_profile.model_dump()
                                    if session.active_profile
                                    else None
                                ),
                                detected_slots=detected_slots,
                                context_messages_count=len(
                                    session_mgr.get_messages(session_id)
                                ),
                            )

                            debug_meta = None
                            if self.settings.DEBUG_CHAT_META:
                                debug_meta = TurnDebugMeta(
                                    mode=(
                                        mode.value if hasattr(mode, "value") else mode
                                    ),
                                    phase=session.clarify_phase.value,
                                    gated_turns=session.gated_turns,
                                    tools_available=active_tools_names,
                                    latency_ms=latency_ms,
                                    usage_metadata=usage_meta,
                                    runs=runs,
                                    state_snapshot=state_snapshot,
                                ).model_dump()

                            turn_id = (len(session.messages) // 2) + 1
                            await log_chat_turn(
                                session_id=session_id,
                                turn_id=turn_id,
                                user_message=request.message,
                                assistant_message=final_reply,
                                latency_ms=latency_ms,
                                token_usage=usage_meta,
                                error=None,
                            )

                            yield {
                                "type": "done",
                                "reply": final_reply,
                                "session_id": session_id,
                                "intent": intent.value,
                                "clarification_asked": fin_data.get(
                                    "needs_clarification", False
                                ),
                                "clarification_form": (
                                    fin_data.get("clarification_form").model_dump()
                                    if fin_data.get("clarification_form")
                                    else None
                                ),
                                "tools_used": tools_used,
                                "products": [
                                    p.model_dump() for p in fin_data.get("products", [])
                                ],
                                "debug_meta": debug_meta,
                            }
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("Error in chat_stream: %s", e, exc_info=True)
            turn_id = (len(session.messages) // 2) + 1 if "session" in locals() else 1
            await log_chat_turn(
                session_id=session_id,
                turn_id=turn_id,
                user_message=request.message,
                assistant_message=None,
                latency_ms=latency_ms,
                token_usage=None,
                error=str(e),
            )
            yield {
                "type": "error",
                "content": (
                    "Hệ thống tư vấn đang gặp gián đoạn tạm thời. "
                    "Bạn vui lòng gửi lại câu hỏi sau giây lát nhé!"
                ),
                "session_id": session_id,
            }


AgentService = ShopAssistantAgent
