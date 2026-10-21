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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.llm.factory import get_chat_model
from customer_shop_assistant.schemas.chat import (
    AgentReply,
    ChatRecommendedProduct,
    ChatRequest,
    ChatResponse,
    ClarificationMode,
    UserProfile,
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
from customer_shop_assistant.tools import ALL_TOOLS
from customer_shop_assistant.tools.catalog_tool import (
    LATEST_SEARCHED_PRODUCT_IDS,
    PRODUCT_REGISTRY,
    product_schema_to_recommended,
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
   - Khi khách hỏi tìm đồ, gợi ý trang phục (ví dụ: 'gợi ý đầm', 'tìm áo sơ mi'):
     * BẮT BUỘC gọi tool `search_products` để tìm sản phẩm thực tế trong cửa hàng.
     * BẮT BUỘC GIỚI THIỆU NGAY 2-4 sản phẩm tìm được vào câu trả lời kèm mã `[ID]`
       (ví dụ: `[57] Đầm Dạ Hội Ren`) và nêu bật lý do thiết kế, chất vải, form dáng.
     * TUYỆT ĐỐI KHÔNG chặn khách để đòi số đo trước. LUÔN GỢI Ý SẢN PHẨM TRƯỚC.
     * Đơn vị tiền là VNĐ. Nếu nói '300k', '400k' thì truyền vào tool là 300000, 400000.
     * Ở cuối câu trả lời, có thể hỏi nhẹ nhàng: 'Để chọn size vừa vặn nhất cho mẫu \
này, bạn có thể cho shop biết thêm chiều cao, cân nặng nhé!'.
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
4. MÃ GIẢM GIÁ:
   - Dùng `check_voucher` kiểm tra điều kiện trước khi báo. Báo đúng giá trị giảm \
và điều kiện tối thiểu từ tool, không tự tính giảm giá sai khác.
5. HỒ SƠ & LƯU THÔNG TIN CÁ NHÂN:
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
6. CÂU HỎI NGOÀI PHẠM VI:
   - Nếu câu hỏi không liên quan đến thời trang hay mua sắm tại shop, \
hãy từ chối ngắn gọn, lịch sự và hướng khách về việc chọn đồ. Đặt `product_ids = []`.
7. GỢI Ý TRẢ LỜI NHANH (QUICK REPLIES):
   - Gợi ý 2-4 câu trả lời nhanh / câu hỏi tiếp theo phù hợp với ngữ cảnh hiện tại. \
Không chứa số đo cá nhân của khách hàng, không tự bịa % giảm giá hay cam kết \
ngoài kết quả tool.
"""


class AgentState(TypedDict):
    """LangGraph conversation state with automatic message appending."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: int | None
    user_profile: UserProfile | None
    clarification_mode: ClarificationMode
    clarify_phase: ClarifyPhase
    gated_turns: int
    tools_used: list[str]
    reply: str
    needs_clarification: bool
    product_ids: list[str]
    products: list[ChatRecommendedProduct]
    quick_replies: list[str]


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
        session_id: str,
        user_id: int | None,
        user_profile: UserProfile | None,
        policy_prompt_block: str = "",
        clarification_mode: ClarificationMode | None = None,
    ) -> str:
        """Construct system prompt with persona, profile, and clarification rules."""
        sys_content = SYSTEM_PROMPT

        if user_id:
            sys_content += (
                f"\nKhách hàng hiện tại đã đăng nhập với User ID: #{user_id}, "
                f"Session ID: '{session_id}'.\n"
                "QUY ĐỊNH BỘ NHỚ LÂU DÀI:\n"
                "- Ưu tiên trọng tâm tư vấn và gợi ý các sản phẩm phù hợp.\n"
                "- Mọi thông tin số đo CHỈ lưu tạm trong phiên để gợi ý đồ, "
                "TUYỆT ĐỐI KHÔNG tự ý lưu vĩnh viễn vào tài khoản trong lúc tư vấn.\n"
                "- CHỈ KHI KHÁCH HÀNG YÊU CẦU LƯU (ví dụ: 'lưu lại số đo giúp mình', "
                "'nhớ số đo này cho mình nhé', 'cập nhật vào tài khoản') "
                "HOẶC ĐỒNG Ý RÕ RÀNG:\n"
                f"  Hãy gọi tool `update_customer_preference(user_id={user_id}, "
                f"session_id='{session_id}')` để lưu số đo phiên vào tài khoản, "
                "và phản hồi lịch sự kèm thông báo đã lưu.\n"
                "- KHI KHÁCH NÓI CẢM ƠN (ví dụ: 'cảm ơn shop', 'thanks'): "
                "Chỉ phản hồi cảm ơn lịch sự, "
                "TUYỆT ĐỐI KHÔNG gọi tool update_customer_preference.\n"
            )
        else:
            sys_content += (
                f"\nKhách hàng là khách vãng lai (Session ID: '{session_id}').\n"
                "Tập trung tư vấn và gợi ý sản phẩm phù hợp. "
                "Mọi thông tin chỉ dùng tạm trong phiên này, "
                "không thể lưu lâu dài vào tài khoản.\n"
            )

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

        async def agent_node(state: AgentState) -> dict[str, Any]:
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            policy = resolve_turn_policy(mode, phase)

            if policy.gate_search:
                active_tools = [t for t in ALL_TOOLS if t.name != "search_products"]
            else:
                active_tools = list(ALL_TOOLS)

            try:
                bound_model = self.chat_model.bind_tools(active_tools)
            except (NotImplementedError, AttributeError):
                bound_model = self.chat_model

            sys_prompt = self._build_system_prompt(
                session_id=state["session_id"],
                user_id=state.get("user_id"),
                user_profile=state.get("user_profile"),
                policy_prompt_block=policy.prompt_block,
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

            response = await bound_model.ainvoke(full_prompt_msgs)
            return {"messages": [response]}

        async def tools_node(state: AgentState) -> dict[str, Any]:
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", [])
            tool_messages: list[ToolMessage] = []
            tools_used = list(state.get("tools_used") or [])

            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            policy = resolve_turn_policy(mode, phase)

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
                            t_result = await tool.ainvoke(t_args)
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

        async def finalize_node(state: AgentState) -> dict[str, Any]:
            last_text = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_text = str(msg.content or "")
                    break

            agent_reply = await self._generate_agent_reply(
                state["messages"], last_reply_text=last_text
            )

            searched_ids = (
                list(LATEST_SEARCHED_PRODUCT_IDS)
                if "search_products" in state.get("tools_used", [])
                else None
            )
            products = await self._resolve_products(
                agent_reply.product_ids,
                reply_text=last_text or agent_reply.reply,
                fallback_product_ids=searched_ids,
            )
            agent_reply.product_ids = [str(p.id) for p in products]

            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            gated_turns = state.get("gated_turns", 0)

            if (
                mode == ClarificationMode.ALWAYS_CLARIFY
                and phase == ClarifyPhase.INTERVIEW
            ):
                agent_reply.needs_clarification = True

            new_phase = next_phase(
                mode=mode,
                current_phase=phase,
                gated_turns=gated_turns,
                needs_clarification=agent_reply.needs_clarification,
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
                "product_ids": agent_reply.product_ids,
                "products": products,
                "quick_replies": agent_reply.quick_replies,
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
    ) -> tuple[str, SessionManager, SessionState, ClarificationMode, TurnPolicy]:
        """Resolve and initialize session, profile, and clarification policy."""
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

        policy = resolve_turn_policy(mode, session.clarify_phase)
        if policy.gate_search:
            session.gated_turns += 1

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
                if item.role == "assistant":
                    session.messages.append(AIMessage(content=item.content))
                else:
                    session.messages.append(HumanMessage(content=item.content))

        session_mgr.add_user_message(session_id, request.message)
        return session_id, session_mgr, session, mode, policy

    async def _generate_agent_reply(
        self,
        turn_messages: list[BaseMessage],
        last_reply_text: str = "",
    ) -> AgentReply:
        """Obtain AgentReply via structured output or fallback parsing."""
        if self.structured_model is not None:
            try:
                result = await self.structured_model.ainvoke(turn_messages)
                reply_obj: AgentReply | None = None
                if isinstance(result, AgentReply):
                    reply_obj = result
                elif isinstance(result, dict):
                    reply_obj = AgentReply.model_validate(result)

                if reply_obj is not None:
                    if not reply_obj.product_ids and last_reply_text:
                        for match in re.findall(
                            r"(?:ID:\s*\[|\[)(\d+)\]", last_reply_text
                        ):
                            if match not in reply_obj.product_ids:
                                reply_obj.product_ids.append(match)
                    if not reply_obj.reply and last_reply_text:
                        reply_obj.reply = last_reply_text
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
            if LATEST_SEARCHED_PRODUCT_IDS:
                text = (
                    "Dưới đây là một số mẫu trang phục phù hợp với nhu cầu và "
                    "phong cách của bạn:"
                )
            else:
                text = "Mình có thể hỗ trợ gì thêm cho bạn không?"
        else:
            text = last_reply_text

        is_clarification = (
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
        ) and any(
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

        found_pids: list[str] = []
        for match in re.findall(r"(?:ID:\s*\[|\[)(\d+)\]", text):
            if match not in found_pids:
                found_pids.append(match)

        return AgentReply(
            reply=text,
            needs_clarification=is_clarification,
            product_ids=found_pids,
            quick_replies=[],
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
            prod = PRODUCT_REGISTRY.get(pid)
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
                            PRODUCT_REGISTRY[pid] = rec_prod
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
            ) = await self._resolve_turn_session(
                request=request, user_id=user_id, auth_token=auth_token
            )

            history_msgs = session_mgr.get_messages(
                session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
            )

            initial_state: AgentState = {
                "messages": history_msgs,
                "session_id": session_id,
                "user_id": user_id,
                "user_profile": session.active_profile,
                "clarification_mode": mode,
                "clarify_phase": session.clarify_phase,
                "gated_turns": session.gated_turns,
                "tools_used": [],
                "reply": "",
                "needs_clarification": False,
                "product_ids": [],
                "products": [],
                "quick_replies": [],
            }

            config = {"configurable": {"thread_id": session_id}}
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

            debug_meta = None
            if self.settings.DEBUG_CHAT_META:
                debug_meta = {
                    "mode": mode.value if hasattr(mode, "value") else mode,
                    "phase": session.clarify_phase.value,
                    "gated_turns": session.gated_turns,
                    "tools_available": [
                        t.name
                        for t in (
                            [t for t in ALL_TOOLS if t.name != "search_products"]
                            if policy.gate_search
                            else list(ALL_TOOLS)
                        )
                    ],
                    "latency_ms": latency_ms,
                }

            return ChatResponse(
                reply=result_state.get("reply", ""),
                session_id=session_id,
                intent=None,
                clarification_asked=result_state.get("needs_clarification", False),
                tools_used=result_state.get("tools_used", []),
                products=result_state.get("products", []),
                quick_replies=result_state.get("quick_replies", []),
                debug_meta=debug_meta,
            )
        except Exception as e:
            logger.error("Error running agent graph: %s", e, exc_info=True)
            return ChatResponse(
                reply=(
                    "Hệ thống tư vấn đang gặp gián đoạn tạm thời. "
                    "Bạn vui lòng gửi lại câu hỏi sau giây lát nhé!"
                ),
                session_id=request.session_id or str(uuid.uuid4()),
                intent="ERROR",
                clarification_asked=False,
                tools_used=[],
                products=[],
                quick_replies=[],
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
        ) = await self._resolve_turn_session(
            request=request, user_id=user_id, auth_token=auth_token
        )

        history_msgs = session_mgr.get_messages(
            session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
        )

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": session.active_profile,
            "clarification_mode": mode,
            "clarify_phase": session.clarify_phase,
            "gated_turns": session.gated_turns,
            "tools_used": [],
            "reply": "",
            "needs_clarification": False,
            "product_ids": [],
            "products": [],
            "quick_replies": [],
        }

        config = {"configurable": {"thread_id": session_id}}
        tools_used: list[str] = []
        full_reply_parts: list[str] = []

        try:
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
                        final_reply = "".join(full_reply_parts).strip()
                        if not final_reply:
                            final_reply = fin_data.get("reply", "")

                        latency_ms = int((time.time() - start_time) * 1000)
                        debug_meta = None
                        if self.settings.DEBUG_CHAT_META:
                            debug_meta = {
                                "mode": (
                                    mode.value if hasattr(mode, "value") else mode
                                ),
                                "phase": session.clarify_phase.value,
                                "gated_turns": session.gated_turns,
                                "tools_available": [
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
                                ],
                                "latency_ms": latency_ms,
                            }

                        yield {
                            "type": "done",
                            "reply": final_reply,
                            "session_id": session_id,
                            "intent": None,
                            "tools_used": tools_used,
                            "products": [
                                p.model_dump() for p in fin_data.get("products", [])
                            ],
                            "quick_replies": fin_data.get("quick_replies", []),
                            "debug_meta": debug_meta,
                        }
        except Exception as e:
            logger.error("Error in chat_stream: %s", e, exc_info=True)
            yield {
                "type": "error",
                "content": (
                    "Hệ thống tư vấn đang gặp gián đoạn tạm thời. "
                    "Bạn vui lòng gửi lại câu hỏi sau giây lát nhé!"
                ),
                "session_id": session_id,
            }


AgentService = ShopAssistantAgent
