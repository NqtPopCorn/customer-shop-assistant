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
    OutfitSuggestion,
    StateSnapshotDebug,
    TurnDebugMeta,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
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
    INTERVIEW_SYSTEM_PROMPT,
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
    FEMALE_ONLY_CATEGORIES,
    balanced_gender_search_context,
    get_cached_product,
    get_cached_product_gender,
    get_latest_outfits,
    get_latest_searched_product_ids,
    product_schema_to_recommended,
    product_turn_context,
    record_product,
)

logger = logging.getLogger(__name__)

_GENDERED_KEYWORD_PATTERN = re.compile(
    r"(?:đầm|váy|chân váy|\bdress(?:es)?\b|\bskirt(?:s)?\b|\bgown\b"
    r"|\bnữ\b|\bnu\b|\bnam\b|\bfemale\b|\bmale\b|\bwomen\b|\bwoman\b"
    r"|\bmen\b|\bman\b)",
    re.IGNORECASE,
)

_PRODUCT_ID_PATTERN = re.compile(
    r"(?:\[ID\]\s*:?\s*(\d+)\b|(?:ID:\s*\[|\[)(\d+)\])", re.IGNORECASE
)


def _extract_product_ids_from_text(text: str) -> list[str]:
    """Extract unique numeric product IDs from [123], ID: [123], or [ID] 123."""
    found: list[str] = []
    for g1, g2 in _PRODUCT_ID_PATTERN.findall(text):
        pid = g1 or g2
        if pid and pid not in found:
            found.append(pid)
    return found


_PRODUCT_PRICE_PATTERN = re.compile(
    r"\b\d{1,3}(?:[.,]\d{3})+\s*(?:đ|vnđ|vnd)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """Bạn là trợ lý thời trang thông minh của Fashion Shop, \
chuyên tư vấn và gợi ý sản phẩm phù hợp nhất cho khách hàng.

Phong cách giao tiếp:
- Tinh gọn, trọng tâm, gu thẩm mỹ tốt, đúng sự thật khách quan (fact-first).
- Trả lời bằng tiếng Việt lịch sự, thân thiện, nhiệt tình gợi ý phối đồ.
- Sử dụng cú pháp Markdown chuẩn (xuống dòng tự nhiên, gạch đầu dòng, in đậm).
  Tuyệt đối KHÔNG viết các thẻ HTML như <br>, <p> vào câu trả lời.

Quy định bắt buộc:
1. GỢI Ý SẢN PHẨM & PHỐI ĐỒ (TRỌNG TÂM HÀNG ĐẦU):
   - Khi đã đủ thông tin để tìm 1 loại sản phẩm đơn lẻ theo chế độ làm rõ hiện tại:
     * BẮT BUỘC gọi tool `search_products` (truyền `limit=5` hoặc nhỏ hơn) trong
       chính lượt hiện tại để lấy sản phẩm thực tế trong cửa hàng.
     * CHỈ GIỚI THIỆU tối đa 3-5 sản phẩm tìm được (TUYỆT ĐỐI không quá 5 sản phẩm),
       viết đúng định dạng `[ID] Tên sản phẩm` (ví dụ: `[485] Áo sơ mi...`) kèm
       giá tiền thực tế từ tool, nêu bật thiết kế, chất vải, form dáng.
     * Khi dùng `search_products` để tìm 1 loại sản phẩm, TUYỆT ĐỐI KHÔNG tự trình
       bày thành `Bộ 1 / Bộ 2` hay viết dòng `- Quần: ... (cần thêm gợi ý)`.
     * KHÔNG chặn khách để đòi số đo P3 khi khách chỉ đang tìm sản phẩm.
     * Đơn vị tiền là VNĐ. Nếu nói '300k', '400k' thì truyền vào tool là 300000, 400000.
   - CHỈ KHI khách hỏi PHỐI ĐỒ, mix & match hoặc tìm SET/BỘ/COMBO (Áo + Quần/Váy):
     * BẮT BUỘC gọi trực tiếp tool `suggest_outfit` (tối đa `limit=3` bộ, truyền
       `top_category`, `bottom_category`, `top_color`, `bottom_color`, `occasion`,
       `gender` hoặc `product_id` nếu đã có ID). TUYỆT ĐỐI KHÔNG gọi `search_products`
       để chỉ gợi ý 1 loại sản phẩm đơn lẻ.
     * BẮT BUỘC trình bày kết quả theo từng **Bộ / Combo hoàn chỉnh**, mỗi bộ nêu rõ
       cả **Áo kèm mã `[ID]` (màu, giá)** VÀ **Quần/Váy kèm mã `[ID]` (màu, giá)**
       cùng lý do phối đồ, nhắc khách nếu màu đang hết hàng.
   - Điền vào `product_ids` danh sách ID các sản phẩm thực sự được gợi ý trong \
câu trả lời (ví dụ: ['ID1', 'ID2']). Trả về danh sách rỗng nếu không gợi ý sản phẩm.
   - Tuyệt đối KHÔNG tự bịa đặt tên sản phẩm, giá tiền, mã [ID] hay trạng thái tồn kho \
khi chưa gọi `search_products` hoặc `suggest_outfit` trong lượt hiện tại.
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
    outfits: list[OutfitSuggestion]


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
        self._enabled_tools = [
            t
            for t in ALL_TOOLS
            if self.settings.OUTFIT_KG_ENABLED or t.name != "suggest_outfit"
        ]
        self.tools_by_name = {t.name: t for t in self._enabled_tools}

        try:
            self.model_with_tools = self.chat_model.bind_tools(self._enabled_tools)
        except (NotImplementedError, AttributeError):
            self.model_with_tools = self.chat_model

        try:
            self.structured_model = self.chat_model.with_structured_output(AgentReply)
        except (NotImplementedError, AttributeError):
            self.structured_model = None

        self.graph = self._create_graph()

    def _get_active_tools_for_turn(
        self, intent: TurnIntent, gate_search: bool
    ) -> list[Any]:
        """Return tools exposed to the LLM for the current turn (strict outfit mode)."""
        if gate_search:
            return []
        if (
            self.settings.OUTFIT_KG_ENABLED
            and intent == TurnIntent.OUTFIT_ADVICE
            and "suggest_outfit" in self.tools_by_name
        ):
            return [t for t in self._enabled_tools if t.name != "search_products"]
        return list(self._enabled_tools)

    @staticmethod
    def _format_profile_notes(user_profile: UserProfile | None) -> str:
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

    def _build_system_prompt(
        self,
        user_profile: UserProfile | None,
        policy_prompt_block: str = "",
        clarification_mode: ClarificationMode | None = None,
    ) -> str:
        """Construct system prompt with persona, profile, and clarification rules."""
        sys_content = SYSTEM_PROMPT + self._format_profile_notes(user_profile)

        if policy_prompt_block:
            sys_content += f"\n{policy_prompt_block}\n"
        elif clarification_mode == ClarificationMode.IMMEDIATE_ANSWER:
            sys_content += f"\n{PROMPT_A}\n"
        elif clarification_mode == ClarificationMode.ALWAYS_CLARIFY:
            sys_content += f"\n{PROMPT_B_INTERVIEW}\n"
        else:
            sys_content += f"\n{PROMPT_C}\n"

        return sys_content

    def _build_interview_system_prompt(
        self,
        user_profile: UserProfile | None,
        slot_summary: str = "",
    ) -> str:
        """Construct compact system prompt for ALWAYS_CLARIFY INTERVIEW turns."""
        sys_content = INTERVIEW_SYSTEM_PROMPT + self._format_profile_notes(user_profile)
        if slot_summary:
            sys_content += f"\n{slot_summary}\n"
        return sys_content

    @staticmethod
    def _should_use_structured_output(
        mode: ClarificationMode,
        phase: ClarifyPhase,
        intent: TurnIntent,
        is_shopping_query: bool,
        tools_used: list[str],
    ) -> bool:
        """Invoke structured_model only when LLM must autonomously build a form."""
        del phase
        if not is_shopping_query:
            return False
        if (
            "search_products" in tools_used
            or "recommend_size" in tools_used
            or "suggest_outfit" in tools_used
        ):
            return False
        if mode == ClarificationMode.ALWAYS_CLARIFY:
            return False
        if mode == ClarificationMode.IMMEDIATE_ANSWER:
            return intent == TurnIntent.SIZE_ADVICE
        return True

    def _build_debug_meta(
        self,
        *,
        mode: ClarificationMode,
        executed_phase: str,
        next_phase_value: str,
        gated_turns: int,
        gate_search: bool,
        active_profile: UserProfile | None,
        slots: ConversationSlots,
        intent: TurnIntent,
        context_messages_count: int,
        latency_ms: int,
        usage_meta: Any,
        runs: list[Any],
    ) -> TurnDebugMeta | None:
        """Build TurnDebugMeta capturing executed phase, next phase, and traces."""
        if not self.settings.DEBUG_CHAT_META:
            return None
        active_tools_names = [
            t.name for t in self._get_active_tools_for_turn(intent, gate_search)
        ]
        state_snapshot = StateSnapshotDebug(
            clarify_phase=executed_phase,
            next_phase=next_phase_value,
            gated_turns=gated_turns,
            tools_available=active_tools_names,
            gate_search=gate_search,
            active_profile=(active_profile.model_dump() if active_profile else None),
            detected_slots=slots.to_debug_dict(intent),
            context_messages_count=context_messages_count,
        )
        return TurnDebugMeta(
            mode=mode.value if hasattr(mode, "value") else mode,
            phase=executed_phase,
            gated_turns=gated_turns,
            tools_available=active_tools_names,
            latency_ms=latency_ms,
            usage_metadata=usage_meta,
            runs=runs,
            state_snapshot=state_snapshot,
        )

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

            slots, intent = extract_conversation_slots(
                state["messages"], state.get("user_profile")
            )

            if policy.gate_search:
                bound_model = self.chat_model
                slot_summary = format_slots_for_prompt(
                    slots, intent, include_missing=False
                )
                sys_prompt = self._build_interview_system_prompt(
                    user_profile=state.get("user_profile"),
                    slot_summary=slot_summary,
                )
            else:
                active_tools = self._get_active_tools_for_turn(
                    intent, policy.gate_search
                )
                try:
                    bound_model = self.chat_model.bind_tools(active_tools)
                except (NotImplementedError, AttributeError):
                    bound_model = self.chat_model

                include_missing = mode != ClarificationMode.IMMEDIATE_ANSWER
                slot_summary = format_slots_for_prompt(
                    slots, intent, include_missing=include_missing
                )
                policy_block = (
                    f"{policy.prompt_block}\n{slot_summary}"
                    if is_shopping_query
                    else policy.prompt_block
                )
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
                set() if policy.gate_search else {t.name for t in self._enabled_tools}
            )
            turn_slots, turn_intent = extract_conversation_slots(
                state["messages"], state.get("user_profile")
            )
            mode_a_slots: ConversationSlots | None = (
                turn_slots
                if mode == ClarificationMode.IMMEDIATE_ANSWER
                and any(tc.get("name") == "search_products" for tc in tool_calls)
                else None
            )

            for tc in tool_calls:
                t_name = tc["name"]
                t_args = tc.get("args", {})
                t_id = tc.get("id") or str(uuid.uuid4())

                if (
                    not policy.gate_search
                    and self.settings.OUTFIT_KG_ENABLED
                    and turn_intent == TurnIntent.OUTFIT_ADVICE
                    and t_name == "search_products"
                    and "suggest_outfit" in self.tools_by_name
                ):
                    logger.info(
                        "Redirecting search_products -> suggest_outfit in OUTFIT_ADVICE"
                    )
                    t_name = "suggest_outfit"
                    t_args = self._build_outfit_tool_args(
                        t_args, turn_slots, state.get("user_id")
                    )
                elif t_name == "suggest_outfit":
                    t_args = self._build_outfit_tool_args(
                        t_args, turn_slots, state.get("user_id")
                    )

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
                            if (
                                t_name == "search_products"
                                and mode_a_slots is not None
                                and not mode_a_slots.has("target_gender")
                            ):
                                effective_args = self._sanitize_mode_a_balanced_args(
                                    t_args, mode_a_slots, state["messages"]
                                )
                                with balanced_gender_search_context(
                                    occasion=mode_a_slots.value_of("occasion")
                                ):
                                    t_result = await tool.ainvoke(
                                        effective_args, config=config
                                    )
                            else:
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
            is_open_no_clarify = intent != TurnIntent.SIZE_ADVICE and (
                mode == ClarificationMode.IMMEDIATE_ANSWER
                or (
                    mode == ClarificationMode.ALWAYS_CLARIFY
                    and phase == ClarifyPhase.OPEN
                )
            )
            allow_clarification = not is_open_no_clarify
            use_structured_output = self._should_use_structured_output(
                mode=mode,
                phase=phase,
                intent=intent,
                is_shopping_query=is_shopping_query,
                tools_used=tools_used,
            )

            agent_reply = await self._generate_agent_reply(
                state["messages"],
                last_reply_text=last_text,
                is_shopping_query=is_shopping_query,
                tools_used=tools_used,
                allow_clarification=allow_clarification,
                use_structured_output=use_structured_output,
                config=config,
            )

            outfits: list[OutfitSuggestion] = []
            outfit_products: list[ChatRecommendedProduct] = []
            if "suggest_outfit" in tools_used:
                outfits = get_latest_outfits()[:3]
                seen_outfit_pids: set[int] = set()
                for outfit in outfits:
                    for item in (outfit.top, outfit.bottom):
                        if item.product_id in seen_outfit_pids:
                            continue
                        seen_outfit_pids.add(item.product_id)
                        cached = get_cached_product(item.product_id)
                        if cached is not None:
                            outfit_products.append(cached)

            has_catalog_tool = (
                "search_products" in tools_used
                or "get_product_detail" in tools_used
                or "suggest_outfit" in tools_used
            )
            if "search_products" in tools_used or "get_product_detail" in tools_used:
                searched_ids = (
                    get_latest_searched_product_ids()
                    if "search_products" in tools_used
                    else None
                )
                resolved_catalog = await self._resolve_products(
                    agent_reply.product_ids,
                    reply_text=last_text or agent_reply.reply,
                    fallback_product_ids=searched_ids,
                )
            else:
                resolved_catalog = []

            if outfit_products:
                seen_pids = {p.id for p in outfit_products}
                products = (
                    outfit_products
                    + [p for p in resolved_catalog if p.id not in seen_pids]
                )[:5]
            else:
                products = resolved_catalog[:5]
            agent_reply.product_ids = [str(p.id) for p in products]

            is_broad_empty_inquiry = (
                not slots.has("occasion")
                and not slots.has("category")
                and not slots.has("top_category")
                and not slots.has("bottom_category")
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
                    if (
                        intent == TurnIntent.OUTFIT_ADVICE
                        and self.settings.OUTFIT_KG_ENABLED
                        and "suggest_outfit" in self.tools_by_name
                    ):
                        (
                            outfits,
                            products,
                            fallback_reply,
                        ) = await self._run_verified_fallback_outfit(
                            slots=slots,
                            user_id=state.get("user_id"),
                            tools_used=tools_used,
                            config=config,
                        )
                        agent_reply.product_ids = [str(p.id) for p in products]
                        if fallback_reply:
                            agent_reply.reply = fallback_reply
                    else:
                        products = await self._run_verified_fallback_search(
                            slots=slots,
                            messages=state["messages"],
                            tools_used=tools_used,
                            config=config,
                        )
                        agent_reply.product_ids = [str(p.id) for p in products]
                        if products:
                            agent_reply.reply = self._format_immediate_fallback_reply(
                                products, slots
                            )
            elif (
                mode == ClarificationMode.ALWAYS_CLARIFY
                and phase == ClarifyPhase.INTERVIEW
                and is_shopping_query
            ):
                agent_reply.needs_clarification = True
            elif (
                mode == ClarificationMode.ALWAYS_CLARIFY
                and phase == ClarifyPhase.OPEN
                and intent != TurnIntent.SIZE_ADVICE
            ):
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None
            elif (
                mode == ClarificationMode.CONDITIONAL_CLARIFICATION
                and is_shopping_query
                and is_broad_empty_inquiry
                and not products
                and not tools_used
            ):
                agent_reply.needs_clarification = True

            has_catalog_tool = (
                "search_products" in tools_used
                or "get_product_detail" in tools_used
                or "suggest_outfit" in tools_used
            )
            has_other_lookup_tool = (
                "search_policy" in tools_used or "check_voucher" in tools_used
            )
            needs_open_mode_b_search = (
                mode == ClarificationMode.ALWAYS_CLARIFY and phase == ClarifyPhase.OPEN
            )
            if (
                is_shopping_query
                and intent
                in (
                    TurnIntent.PRODUCT_SEARCH,
                    TurnIntent.OUTFIT_ADVICE,
                    TurnIntent.BODY_FIT_ADVICE,
                )
                and not agent_reply.needs_clarification
                and not has_catalog_tool
                and not has_other_lookup_tool
                and (
                    needs_open_mode_b_search
                    or self._has_ungrounded_product_listing(
                        last_text or agent_reply.reply
                    )
                )
            ):
                if (
                    intent == TurnIntent.OUTFIT_ADVICE
                    and self.settings.OUTFIT_KG_ENABLED
                    and "suggest_outfit" in self.tools_by_name
                ):
                    logger.warning(
                        "Ungrounded or missing suggest_outfit in %s/%s; "
                        "running verified fallback outfit",
                        mode,
                        phase,
                    )
                    (
                        outfits,
                        products,
                        fallback_reply,
                    ) = await self._run_verified_fallback_outfit(
                        slots=slots,
                        user_id=state.get("user_id"),
                        tools_used=tools_used,
                        config=config,
                    )
                    agent_reply.product_ids = [str(p.id) for p in products]
                    if fallback_reply:
                        agent_reply.reply = fallback_reply
                    else:
                        agent_reply.reply = (
                            "Hiện tại mình chưa tìm thấy bộ phối đồ nào khớp hoàn toàn "
                            "với tiêu chí này trong cửa hàng. Bạn có thể thử dịp mặc "
                            "hoặc màu sắc khác để mình gợi ý lại nhé!"
                        )
                else:
                    logger.warning(
                        "Ungrounded or missing search_products in %s/%s; "
                        "running verified fallback search",
                        mode,
                        phase,
                    )
                    products = await self._run_verified_fallback_search(
                        slots=slots,
                        messages=state["messages"],
                        tools_used=tools_used,
                        config=config,
                    )
                    agent_reply.product_ids = [str(p.id) for p in products]
                    if products:
                        agent_reply.reply = self._format_immediate_fallback_reply(
                            products, slots
                        )
                    else:
                        agent_reply.reply = (
                            "Hiện tại mình chưa tìm thấy sản phẩm nào khớp hoàn toàn "
                            "với tiêu chí này trong cửa hàng. Bạn có thể nới rộng "
                            "khoảng giá hoặc mô tả kiểu đồ khác để mình tìm lại nhé!"
                        )

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
                "outfits": outfits,
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
    def _build_neutral_keyword(
        slots: ConversationSlots, messages: list[BaseMessage]
    ) -> str:
        """Build a clean gender-neutral Vietnamese semantic search keyword."""
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
        occ_val = slots.value_of("occasion")
        if cat_val == "tops" and occ_val in ("party", "office"):
            kw_parts.append("áo sơ mi áo polo áo khoác")
        elif cat_val == "pants" and occ_val in ("party", "office"):
            kw_parts.append("quần âu quần kaki")
        elif cat_val in cat_kw_map:
            kw_parts.append(cat_kw_map[cat_val])
        if occ_val in occ_kw_map:
            if cat_val is None and occ_val == "party":
                kw_parts.append("trang phục dự tiệc lịch sự")
            elif cat_val is None and occ_val == "office":
                kw_parts.append("trang phục công sở lịch sự")
            else:
                kw_parts.append(occ_kw_map[occ_val])
        if slots.has("color"):
            kw_parts.append(str(slots.value_of("color")).lower())

        if kw_parts:
            return " ".join(kw_parts)

        latest_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                latest_text = str(m.content or "").strip()
                break
        cleaned_text = re.sub(r"[^\w\s,.-]", " ", latest_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text or "trang phục thời trang"

    @staticmethod
    def _sanitize_mode_a_balanced_args(
        t_args: dict[str, Any],
        slots: ConversationSlots,
        messages: list[BaseMessage],
    ) -> dict[str, Any]:
        """Strip LLM-hallucinated gender/category biases for Mode A balanced search."""
        sanitized = dict(t_args)
        sanitized.pop("target_gender", None)

        raw_cat = sanitized.get("category")
        if (
            isinstance(raw_cat, str)
            and raw_cat.strip().lower() in FEMALE_ONLY_CATEGORIES
        ):
            sanitized.pop("category", None)

        raw_kw = str(sanitized.get("keyword") or "").strip()
        if not raw_kw or _GENDERED_KEYWORD_PATTERN.search(raw_kw):
            sanitized["keyword"] = ShopAssistantAgent._build_neutral_keyword(
                slots, messages
            )

        return sanitized

    @staticmethod
    def _build_immediate_search_args(
        slots: ConversationSlots, messages: list[BaseMessage]
    ) -> dict[str, Any]:
        """Build search_products tool arguments for verified fallback search."""
        keyword = ShopAssistantAgent._build_neutral_keyword(slots, messages)
        limit = 5 if not slots.has("target_gender") else 4
        args: dict[str, Any] = {"keyword": keyword, "limit": limit}
        exact_cat_map = {
            "shirt": "Áo sơ mi",
            "polo": "Áo polo",
            "tshirt": "Áo thun",
            "jacket": "Áo khoác",
            "jeans": "Quần jeans",
            "kaki": "Quần kaki",
            "short": "Quần short",
            "skirt": "Chân váy",
        }
        cat_val = slots.value_of("category")
        if cat_val in exact_cat_map:
            args["category"] = exact_cat_map[cat_val]
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
    def _has_ungrounded_product_listing(text: str) -> bool:
        """Detect fabricated product IDs or VND prices when no catalog tool ran."""
        if not text:
            return False
        return bool(
            _PRODUCT_ID_PATTERN.search(text) or _PRODUCT_PRICE_PATTERN.search(text)
        )

    @staticmethod
    def _format_product_bullet(p: ChatRecommendedProduct) -> str:
        """Format a single grounded product recommendation bullet."""
        price_txt = f"{p.price:,.0f}đ" if p.price > 0 else "Liên hệ"
        meta_parts = [part for part in (p.category, p.brand) if part]
        meta_txt = f" ({' | '.join(meta_parts)})" if meta_parts else ""
        return f"- [{p.id}] **{p.name}**{meta_txt} — Giá: {price_txt}"

    @staticmethod
    def _build_outfit_tool_args(
        raw_args: dict[str, Any],
        slots: ConversationSlots,
        user_id: int | None,
    ) -> dict[str, Any]:
        """Normalize and enrich suggest_outfit arguments from detected slots."""
        allowed_keys = {
            "product_id",
            "color",
            "top_category",
            "bottom_category",
            "top_color",
            "bottom_color",
            "occasion",
            "gender",
            "user_id",
            "favorite_colors",
            "disliked_colors",
            "limit",
        }
        args: dict[str, Any] = {
            k: v for k, v in raw_args.items() if k in allowed_keys and v is not None
        }

        if not args.get("gender") and raw_args.get("target_gender"):
            args["gender"] = raw_args["target_gender"]
        if (
            not args.get("top_category")
            and not args.get("bottom_category")
            and raw_args.get("category")
        ):
            args["top_category"] = raw_args["category"]

        cat_code_map = {
            "shirt": "Áo sơ mi",
            "polo": "Áo polo",
            "tshirt": "Áo thun",
            "jacket": "Áo khoác",
            "jeans": "Quần jeans",
            "kaki": "Quần kaki",
            "short": "Quần short",
            "pants": "Quần âu",
            "skirt": "Chân váy",
            "dress": "Váy liền",
            "accessory": "Phụ kiện",
        }
        if not args.get("top_category") and slots.has("top_category"):
            raw_top = str(slots.value_of("top_category"))
            args["top_category"] = cat_code_map.get(raw_top, raw_top)
        if not args.get("bottom_category") and slots.has("bottom_category"):
            raw_bot = str(slots.value_of("bottom_category"))
            args["bottom_category"] = cat_code_map.get(raw_bot, raw_bot)
        if (
            args.get("product_id") is None
            and not args.get("top_category")
            and not args.get("bottom_category")
            and slots.value_of("category") in ("dress", "accessory")
        ):
            raw_cat = str(slots.value_of("category"))
            args["top_category"] = cat_code_map.get(raw_cat, raw_cat)

        if not args.get("top_color") and slots.has("top_color"):
            args["top_color"] = slots.value_of("top_color")
        if not args.get("bottom_color") and slots.has("bottom_color"):
            args["bottom_color"] = slots.value_of("bottom_color")
        if (
            not args.get("color")
            and not args.get("top_color")
            and not args.get("bottom_color")
            and slots.has("color")
        ):
            args["color"] = slots.value_of("color")

        if not args.get("occasion") and slots.has("occasion"):
            occ_map = {
                "office": "đi làm",
                "party": "hẹn hò",
                "casual": "đi chơi",
                "sport": "đi chơi",
            }
            occ_val = str(slots.value_of("occasion"))
            args["occasion"] = occ_map.get(occ_val, occ_val)

        if not args.get("gender") and slots.has("target_gender"):
            args["gender"] = slots.value_of("target_gender")
        if args.get("user_id") is None and user_id is not None:
            args["user_id"] = user_id

        return args

    @staticmethod
    def _format_immediate_outfit_fallback_reply(
        outfits: list[OutfitSuggestion], slots: ConversationSlots
    ) -> str:
        """Format complete top + bottom combos from OutfitGraph for fallback paths."""
        if not slots.has("target_gender") and not slots.has("bottom_category"):
            intro = (
                "Vì bạn chưa nêu rõ giới tính nên mình gợi ý các bộ phối đồ "
                "cân bằng cho cả Nữ và Nam:"
            )
        else:
            intro = "Dưới đây là các bộ phối đồ (combo Áo + Quần/Váy) phù hợp nhất:"

        lines = [intro]
        for idx, outfit in enumerate(outfits, start=1):
            top_stock = "" if outfit.top.in_stock else " *(màu này đang hết hàng)*"
            bot_stock = "" if outfit.bottom.in_stock else " *(màu này đang hết hàng)*"
            lines.append(f"\n**Bộ {idx}** (Độ phù hợp: {outfit.score:.2f}):")
            lines.append(
                f"- **Áo**: [{outfit.top.product_id}] **{outfit.top.name}** "
                f"— Màu: {outfit.top.color} — Giá: {outfit.top.price:,.0f}đ{top_stock}"
            )
            lines.append(
                f"- **Quần/Váy**: [{outfit.bottom.product_id}] "
                f"**{outfit.bottom.name}** — Màu: {outfit.bottom.color} "
                f"— Giá: {outfit.bottom.price:,.0f}đ{bot_stock}"
            )
            if outfit.reasons:
                lines.append(f"- *Lý do phối*: {'; '.join(outfit.reasons)}")
        return "\n".join(lines)

    async def _run_verified_fallback_outfit(
        self,
        slots: ConversationSlots,
        user_id: int | None,
        tools_used: list[str],
        config: RunnableConfig | None = None,
    ) -> tuple[list[OutfitSuggestion], list[ChatRecommendedProduct], str]:
        """Execute suggest_outfit fallback and resolve combo products & reply."""
        outfit_tool = self.tools_by_name.get("suggest_outfit")
        if outfit_tool is None:
            return [], [], ""
        fallback_args = self._build_outfit_tool_args({}, slots, user_id)
        try:
            tool_res = await outfit_tool.ainvoke(fallback_args, config=config)
            if "suggest_outfit" not in tools_used:
                tools_used.append("suggest_outfit")
            outfits = get_latest_outfits()
            products: list[ChatRecommendedProduct] = []
            seen_pids: set[int] = set()
            for outfit in outfits:
                for item in (outfit.top, outfit.bottom):
                    if item.product_id in seen_pids:
                        continue
                    seen_pids.add(item.product_id)
                    cached = get_cached_product(item.product_id)
                    if cached is not None:
                        products.append(cached)
            if outfits:
                reply = self._format_immediate_outfit_fallback_reply(outfits, slots)
                return outfits, products, reply
            res_text = str(tool_res or "").strip()
            return [], [], res_text
        except Exception as fallback_err:
            logger.warning("Verified fallback outfit failed: %s", fallback_err)
            return [], [], ""

    async def _run_verified_fallback_search(
        self,
        slots: ConversationSlots,
        messages: list[BaseMessage],
        tools_used: list[str],
        config: RunnableConfig | None = None,
    ) -> list[ChatRecommendedProduct]:
        """Execute a grounded search_products fallback and resolve real products."""
        search_tool = self.tools_by_name.get("search_products")
        if search_tool is None:
            return []
        fallback_args = self._build_immediate_search_args(slots, messages)
        try:
            if not slots.has("target_gender"):
                with balanced_gender_search_context(
                    occasion=slots.value_of("occasion")
                ):
                    tool_res = await search_tool.ainvoke(fallback_args, config=config)
            else:
                tool_res = await search_tool.ainvoke(fallback_args, config=config)
            if "search_products" not in tools_used:
                tools_used.append("search_products")
            return await self._resolve_products(
                [],
                reply_text=str(tool_res),
                fallback_product_ids=get_latest_searched_product_ids(),
            )
        except Exception as fallback_err:
            logger.warning("Verified fallback search failed: %s", fallback_err)
            return []

    @staticmethod
    def _format_immediate_fallback_reply(
        products: list[ChatRecommendedProduct], slots: ConversationSlots
    ) -> str:
        """Format a direct fact-first recommendation list with clear assumption."""
        top_products = products[:5]
        if not slots.has("target_gender"):
            buckets: dict[GenderTarget, list[ChatRecommendedProduct]] = {
                GenderTarget.FEMALE: [],
                GenderTarget.MALE: [],
                GenderTarget.UNISEX: [],
            }
            for p in top_products:
                g = get_cached_product_gender(p.id)
                if g in buckets:
                    buckets[g].append(p)

            if any(buckets.values()):
                lines = [
                    "Vì bạn chưa nêu rõ giới tính nên mình gợi ý ngay các mẫu "
                    "tiêu biểu cho cả Nữ, Nam và Unisex (nếu có):"
                ]
                group_labels = (
                    (GenderTarget.FEMALE, "**Gợi ý cho Nữ:**"),
                    (GenderTarget.MALE, "**Gợi ý cho Nam:**"),
                    (GenderTarget.UNISEX, "**Gợi ý Unisex:**"),
                )
                for g_key, label in group_labels:
                    items = buckets[g_key]
                    if not items:
                        continue
                    lines.append(f"\n{label}")
                    for p in items:
                        lines.append(ShopAssistantAgent._format_product_bullet(p))
                return "\n".join(lines)

            intro = (
                "Vì bạn chưa nêu cụ thể đối tượng mặc nên mình gợi ý ngay "
                "các mẫu tiêu biểu phù hợp nhất hiện có tại cửa hàng:"
            )
        else:
            intro = "Dưới đây là các mẫu trang phục phù hợp hiện có tại cửa hàng:"
        lines = [intro]
        for p in top_products:
            lines.append(ShopAssistantAgent._format_product_bullet(p))
        return "\n".join(lines)

    async def _generate_agent_reply(
        self,
        turn_messages: list[BaseMessage],
        last_reply_text: str = "",
        is_shopping_query: bool = True,
        tools_used: list[str] | None = None,
        allow_clarification: bool = True,
        use_structured_output: bool = True,
        config: RunnableConfig | None = None,
    ) -> AgentReply:
        """Obtain AgentReply via structured output or fallback parsing."""
        welcome_phrases = (
            "mình có thể giúp bạn tìm kiếm trang phục",
            "bạn đang quan tâm đến phong cách nào hôm nay",
            "mình là stylist ai của fashion shop",
        )
        used_set = set(tools_used or [])

        if use_structured_output and self.structured_model is not None:
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
                        reply_obj.product_ids = _extract_product_ids_from_text(
                            last_reply_text
                        )

                    if (
                        allow_clarification
                        and not reply_obj.needs_clarification
                        and is_shopping_query
                        and not reply_obj.product_ids
                        and "search_products" not in used_set
                        and "suggest_outfit" not in used_set
                        and last_reply_text
                        and not self._has_ungrounded_product_listing(last_reply_text)
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
            if get_latest_searched_product_ids() or get_latest_outfits():
                text = (
                    "Dưới đây là một số mẫu trang phục phù hợp với nhu cầu và "
                    "phong cách của bạn:"
                )
            else:
                text = "Mình có thể hỗ trợ gì thêm cho bạn không?"
        else:
            text = last_reply_text

        found_pids = _extract_product_ids_from_text(text)

        is_clarification = (
            allow_clarification
            and is_shopping_query
            and not found_pids
            and "search_products" not in used_set
            and "suggest_outfit" not in used_set
            and not self._has_ungrounded_product_listing(text)
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
        *,
        allow_unverified_fetch: bool = False,
    ) -> list[ChatRecommendedProduct]:
        """Resolve products from verified turn cache (and optional ID fetch)."""
        candidate_ids = list(product_ids)

        for match in _extract_product_ids_from_text(reply_text):
            if match not in candidate_ids:
                candidate_ids.append(match)

        if not candidate_ids and fallback_product_ids:
            candidate_ids = [str(pid) for pid in fallback_product_ids[:5]]

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
            elif allow_unverified_fetch:
                try:
                    async with ShopClient(settings=self.settings) as client:
                        p_schema = await client.get_product_by_id(pid)
                        if p_schema:
                            rec_prod = product_schema_to_recommended(p_schema)
                            record_product(rec_prod)
                            resolved.append(rec_prod)
                except Exception:
                    pass

        return resolved[:5]

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
            executed_phase = session.clarify_phase.value

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
                "outfits": [],
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
                "next_phase=%s tools=%s needs_clarification=%s products_count=%d "
                "latency_ms=%d",
                session_id[:8],
                mode.value if hasattr(mode, "value") else mode,
                executed_phase,
                session.clarify_phase.value,
                result_state.get("tools_used", []),
                result_state.get("needs_clarification", False),
                len(result_state.get("products", [])),
                latency_ms,
            )

            runs = debug_handler.get_runs()
            usage_meta = debug_handler.get_aggregated_usage()
            debug_meta = self._build_debug_meta(
                mode=mode,
                executed_phase=executed_phase,
                next_phase_value=session.clarify_phase.value,
                gated_turns=session.gated_turns,
                gate_search=policy.gate_search,
                active_profile=session.active_profile,
                slots=slots,
                intent=intent,
                context_messages_count=len(session_mgr.get_messages(session_id)),
                latency_ms=latency_ms,
                usage_meta=usage_meta,
                runs=runs,
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
                outfits=result_state.get("outfits", []),
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
                outfits=[],
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
        executed_phase = session.clarify_phase.value

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
            "outfits": [],
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
                            if not policy.gate_search:
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
                            debug_meta_obj = self._build_debug_meta(
                                mode=mode,
                                executed_phase=executed_phase,
                                next_phase_value=session.clarify_phase.value,
                                gated_turns=session.gated_turns,
                                gate_search=policy.gate_search,
                                active_profile=session.active_profile,
                                slots=slots,
                                intent=intent,
                                context_messages_count=len(
                                    session_mgr.get_messages(session_id)
                                ),
                                latency_ms=latency_ms,
                                usage_meta=usage_meta,
                                runs=runs,
                            )
                            debug_meta = (
                                debug_meta_obj.model_dump() if debug_meta_obj else None
                            )

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
                                "outfits": [
                                    o.model_dump() for o in fin_data.get("outfits", [])
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
