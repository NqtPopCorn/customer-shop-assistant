import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.llm.factory import get_chat_model
from customer_shop_assistant.schemas.chat import (
    ChatRecommendedProduct,
    ChatRequest,
    ChatResponse,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.schemas.shop import GenderTarget
from customer_shop_assistant.services.clarification_engine import (
    ClarificationEngine,
)
from customer_shop_assistant.services.session_manager import get_session_manager
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.tools import ALL_TOOLS
from customer_shop_assistant.tools.catalog_tool import PRODUCT_REGISTRY

logger = logging.getLogger(__name__)


def extract_measurements_from_text(text: str) -> dict[str, Any]:
    """Extract spontaneous customer body measurements from conversation text."""
    lower = text.lower()
    res: dict[str, Any] = {}

    h_match = re.search(
        r"(?:cao\s*)?(?:(\d)m(\d{1,2})|(\d{3})\s*cm|cao\s*(\d{3}))", lower
    )
    if h_match:
        if h_match.group(1) and h_match.group(2):
            m = int(h_match.group(1))
            cm = int(h_match.group(2))
            cm_val = m * 100 + (cm * 10 if cm < 10 else cm)
            if 100 <= cm_val <= 230:
                res["height_cm"] = float(cm_val)
        elif h_match.group(3):
            val = float(h_match.group(3))
            if 100 <= val <= 230:
                res["height_cm"] = val
        elif h_match.group(4):
            val = float(h_match.group(4))
            if 100 <= val <= 230:
                res["height_cm"] = val

    w_match = re.search(r"(?:nặng\s*(\d{2,3})|\b(\d{2,3})\s*(?:kg|ký))", lower)
    if w_match:
        val_str = w_match.group(1) or w_match.group(2)
        if val_str:
            val = float(val_str)
            if 30 <= val <= 200:
                res["weight_kg"] = val

    if any(
        k in lower
        for k in ["bạn nam", "cho nam", "nam giới", "mình là nam", "con trai"]
    ):
        res["gender"] = GenderTarget.MALE
    elif any(
        k in lower for k in ["bạn nữ", "cho nữ", "nữ giới", "mình là nữ", "con gái"]
    ):
        res["gender"] = GenderTarget.FEMALE

    if any(k in lower for k in ["mặc rộng", "form rộng", "oversize", "thoải mái"]):
        res["fit_preference"] = "LOOSE"
    elif any(k in lower for k in ["mặc ôm", "ôm sát", "body", "slim"]):
        res["fit_preference"] = "TIGHT"
    elif any(k in lower for k in ["vừa vặn", "vừa người", "regular"]):
        res["fit_preference"] = "REGULAR"

    return res


def extract_recommended_products(
    messages: list[BaseMessage], reply_text: str
) -> list[ChatRecommendedProduct]:
    """Extract product IDs referenced in tool or assistant messages."""
    found_ids: list[int] = []

    # Check messages (tool outputs and assistant texts)
    for msg in messages:
        content = str(msg.content)
        for match in re.findall(r"\[(\d+)\]", content):
            try:
                pid = int(match)
                if pid not in found_ids and pid in PRODUCT_REGISTRY:
                    found_ids.append(pid)
            except ValueError:
                pass

    # Also check final reply text
    for match in re.findall(r"\[(\d+)\]", reply_text):
        try:
            pid = int(match)
            if pid not in found_ids and pid in PRODUCT_REGISTRY:
                found_ids.append(pid)
        except ValueError:
            pass

    results = [PRODUCT_REGISTRY[pid] for pid in found_ids[:5]]

    # If products found directly in registry but IDs not formatted in reply,
    # pull recent items from PRODUCT_REGISTRY
    if not results and PRODUCT_REGISTRY:
        recent_ids = list(PRODUCT_REGISTRY.keys())[-4:]
        results = [PRODUCT_REGISTRY[pid] for pid in recent_ids]

    return results


def generate_quick_replies(intent: str | None, clarification_asked: bool) -> list[str]:
    """Generate contextual quick reply chips for frontend widget."""
    if clarification_asked:
        return [
            "Mình cao 1m70, nặng 65kg",
            "Tư vấn đồ đi làm công sở",
            "Tư vấn đồ đi tiệc cuối tuần",
        ]
    if intent == "SIZE_ADVICE":
        return [
            "Chính sách đổi trả nếu không vừa 🔄",
            "Gợi ý áo sơ mi hoặc polo phối cùng 👔",
            "Xem bảng size chi tiết 📏",
        ]
    if intent in ("CATALOG_SEARCH", "PRODUCT_ADVICE"):
        return [
            "Tư vấn chọn size chuẩn 📏",
            "Có mẫu màu đen hoặc trắng không?",
            "Chính sách vận chuyển & đổi hàng 🚚",
        ]
    if intent == "POLICY_INQUIRY":
        return [
            "Phí vận chuyển và thời gian giao hàng",
            "Quy định đổi trả hàng trong bao lâu?",
            "Phương thức thanh toán VietQR / COD",
        ]
    return [
        "Gợi ý đồ đi tiệc cuối tuần 🥂",
        "Tìm áo sơ mi công sở 👔",
        "Săn sale hot giảm 50% 🔥",
        "Bảng tư vấn size 📏",
    ]


SYSTEM_PROMPT = """Bạn là trợ lý thời trang thông minh của Fashion Shop, \
chuyên tư vấn và gợi ý sản phẩm phù hợp nhất cho khách hàng.

Phong cách giao tiếp:
- Tinh gọn, trọng tâm, gu thẩm mỹ tốt, đúng sự thật khách quan (fact-first).
- Trả lời bằng tiếng Việt lịch sự, thân thiện, nhiệt tình gợi ý phối đồ.

Quy định bắt buộc:
1. GỢI Ý SẢN PHẨM (TRỌNG TÂM HÀNG ĐẦU):
   - Luôn sử dụng `search_products` hoặc `get_product_detail` để tìm kiếm
     sản phẩm thực tế có sẵn, còn hàng trong cửa hàng.
   - Ưu tiên gợi ý sản phẩm theo nhu cầu khách hàng: dịp mặc (đi làm, đi tiệc,
     dạo phố, du lịch), phong cách (công sở, streetwear, tối giản, thanh lịch),
     gu mặc (form rộng, ôm dáng), chất liệu, màu sắc hoặc vóc dáng.
   - Khi gợi ý sản phẩm, hãy kèm mã định danh `[ID]` (ví dụ: `[1] Áo Oxford`)
     và nêu bật lý do thiết kế, chất vải, màu sắc hoặc form dáng phù hợp với khách.
   - Tuyệt đối không tự bịa đặt tên sản phẩm, mã hàng hay trạng thái tồn kho.
2. Tư vấn size (Hỗ trợ kèm theo):
   - Khi khách hàng hỏi size hoặc sau khi đã chọn được mẫu ưng ý,
     hãy dùng `recommend_size` để tính toán size chuẩn theo bảng quy đổi.
3. Chính sách bán hàng:
   - Dùng `search_policy` khi khách hỏi về đổi trả, phí vận chuyển, freeship,
     thanh toán VietQR/COD. Bắt buộc trích dẫn nguồn `[tên_tệp: tiêu_đề_mục]`.
4. Mã giảm giá: Dùng `check_voucher` kiểm tra điều kiện trước khi báo.
5. Hồ sơ & Lưu thông tin cá nhân:
   - Trong suốt cuộc trò chuyện, toàn bộ số đo (chiều cao, cân nặng, số đo 3 vòng)
     và sở thích của khách CHỈ ĐƯỢC LƯU TẠM THỜI trong phiên hiện tại để gợi ý đồ.
     TUYỆT ĐỐI KHÔNG tự động lưu vĩnh viễn vào tài khoản khi khách chưa kết thúc.
   - CHỈ gọi tool `update_customer_preference` khi:
     (a) Khách hàng nói lời cảm ơn (ví dụ: 'cảm ơn shop', 'thanks', 'cảm ơn bạn nhé')
         để chốt lưu lại các thông tin/số đo của phiên vào hồ sơ tài khoản cho khách.
     HOẶC (b) Khách hàng chủ động yêu cầu lưu (ví dụ: 'lưu lại số đo này giúp mình',
         'nhớ số đo này cho mình nhé', 'cập nhật vào hồ sơ').
   - Đối với khách vãng lai (chưa có user_id): Không gọi tool lưu, phản hồi cảm ơn
     lịch sự và gợi ý khách đăng nhập nếu muốn lưu hồ sơ lâu dài.
"""

class AgentState(TypedDict):
    """LangGraph conversation state."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: int | None
    user_profile: UserProfile | None
    clarification_mode: ClarificationMode
    intent: str
    clarification_asked: bool
    tools_used: list[str]


def classify_intent(query: str) -> str:
    """Classify user utterance into one of 6 core business intents."""
    lower = query.lower()

    # 1. Out of Scope: personal account, external tech, direct order manipulation
    if any(
        k in lower
        for k in [
            "đơn hàng #",
            "tình trạng đơn",
            "hủy đơn",
            "huỷ đơn",
            "hủy giùm",
            "huỷ giùm",
            "chuyển tiền hoàn",
            "số tài khoản",
            "stk",
            "quên mật khẩu",
            "đổi mật khẩu",
            "mã code python",
            "viết code",
            "thời tiết hôm nay",
            "thời tiết ở",
            "dự báo thời tiết",
        ]
    ):
        return "OUT_OF_SCOPE"

    # 2. Stock & Inventory Check (priority over size to catch 'size L còn hàng không')
    if any(
        k in lower
        for k in [
            "còn hàng không",
            "còn hàng ko",
            "còn ko",
            "còn không",
            "còn mẫu",
            "còn màu",
            "còn những size",
            "còn size nào",
            "còn size",
            "hết hàng",
            "hết rồi",
            "tạm hết",
            "bao giờ có hàng",
            "đợt hàng mới",
            "tồn kho",
        ]
    ):
        return "CATALOG_SEARCH"

    # 3. Policy & FAQ inquiry
    if any(
        k in lower
        for k in [
            "đổi trả",
            "hoàn tiền",
            "phí ship",
            "vận chuyển",
            "freeship",
            "thanh toán",
            "vietqr",
            "mbbank",
            "bảo quản",
            "giặt",
            "chính sách",
            "địa chỉ shop",
            "áp thêm voucher",
            "kết hợp voucher",
        ]
    ):
        return "POLICY_INQUIRY"

    # 4. Size & Fit recommendation
    if any(
        k in lower
        for k in [
            "size",
            "mặc vừa",
            "bảng size",
            "chọn size",
            "lấy size",
            "tăng 1 size",
            "lùi 1 size",
            "giảm 1 size",
            "rộng chật",
            "vòng eo",
            "vòng ngực",
            "vòng bụng",
        ]
    ):
        return "SIZE_ADVICE"

    # 5. Product Suitability & Occasion & Body Form Advisory
    if any(
        k in lower
        for k in [
            "tư vấn",
            "phù hợp",
            "phối đồ",
            "phối với",
            "mix",
            "dịp",
            "đi làm",
            "công sở",
            "đi tiệc",
            "đám cưới",
            "dự tiệc",
            "đi chơi",
            "dạo phố",
            "du lịch",
            "đi biển",
            "form gì",
            "dáng gì",
            "che bụng",
            "đỡ gầy",
            "hack chiều cao",
            "chất liệu",
            "chất vải",
            "vải gì",
            "mùa hè",
            "mùa đông",
            "thoáng mát",
            "giữ ấm",
            "chống nhăn",
            "gợi ý",
            "phong cách",
            "set đồ",
        ]
    ):
        return "PRODUCT_ADVICE"

    # 6. Catalog Search & Voucher & Stock
    if any(
        k in lower
        for k in [
            "voucher",
            "mã giảm",
            "coupon",
            "khuyến mãi",
            "sale",
            "áp dụng được mã",
            "áp được mã",
            "tìm áo",
            "tìm quần",
            "tìm váy",
            "mẫu áo",
            "mẫu váy",
            "mẫu đầm",
            "mẫu quần",
            "giá bao nhiêu",
            "giá dưới",
            "còn hàng",
            "hết hàng",
            "tồn kho",
            "shop có",
            "áo thun",
            "áo khoác",
            "áo sơ mi",
            "áo polo",
            "quần jean",
            "quần kaki",
            "quần short",
            "quần tây",
            "đầm",
            "váy",
        ]
    ):
        return "CATALOG_SEARCH"

    return "GENERAL_CHAT"


class ShopAssistantAgent:
    """Orchestrates chatbot conversation flow using LangGraph."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chat_model = chat_model or get_chat_model(self.settings)
        try:
            self.model_with_tools = self.chat_model.bind_tools(ALL_TOOLS)
        except (NotImplementedError, AttributeError):
            self.model_with_tools = self.chat_model
        self.clarification_engine = ClarificationEngine()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(AgentState)

        # Node 1: Intent classification
        def node_classify(state: AgentState) -> dict[str, Any]:
            last_msg = state["messages"][-1]
            content = str(last_msg.content)
            intent = classify_intent(content)
            return {"intent": intent}

        # Node 2: Clarification check
        def node_clarification_gate(state: AgentState) -> dict[str, Any]:
            last_msg = state["messages"][-1]
            content = str(last_msg.content)
            decision = self.clarification_engine.evaluate(
                query=content,
                intent=state["intent"],
                mode=state["clarification_mode"],
                profile=state["user_profile"],
            )
            if decision.needs_clarification and decision.clarification_question:
                return {
                    "messages": [AIMessage(content=decision.clarification_question)],
                    "clarification_asked": True,
                }
            return {"clarification_asked": False}

        # Node 3: LLM generation
        async def node_agent(state: AgentState) -> dict[str, Any]:
            sys_content = SYSTEM_PROMPT
            uid = state.get("user_id")
            sid = state.get("session_id")
            if uid:
                sys_content += (
                    f"\nKhách hàng hiện tại đã đăng nhập với User ID: #{uid}, "
                    f"Session ID: '{sid}'.\n"
                    "QUY ĐỊNH BỘ NHỚ LÂU DÀI:\n"
                    "- Ưu tiên trọng tâm tư vấn và gợi ý các sản phẩm phù hợp.\n"
                    "- Mọi thông tin số đo CHỈ lưu tạm trong phiên để gợi ý đồ, "
                    "TUYỆT ĐỐI KHÔNG gọi tool lưu vĩnh viễn trong lúc tư vấn.\n"
                    "- KHI KHÁCH NÓI CẢM ƠN (ví dụ: 'cảm ơn shop', 'thanks') "
                    "HOẶC YÊU CẦU LƯU (ví dụ: 'lưu lại số đo giúp mình'):\n"
                    f"  Hãy gọi tool `update_customer_preference(user_id={uid}, "
                    f"session_id='{sid}')` để lưu số đo phiên vào tài khoản, "
                    "và phản hồi lịch sự kèm thông báo đã lưu.\n"
                )
            else:
                sys_content += (
                    f"\nKhách hàng là khách vãng lai (Session ID: '{sid}').\n"
                    "Tập trung tư vấn và gợi ý sản phẩm phù hợp. "
                    "Mọi thông tin chỉ dùng tạm trong phiên này, "
                    "không thể lưu lâu dài vào tài khoản.\n"
                )
            prof = state.get("user_profile")
            if prof:
                prof_notes = []
                if prof.height_cm:
                    prof_notes.append(f"Chiều cao: {prof.height_cm}cm")
                if prof.weight_kg:
                    prof_notes.append(f"Cân nặng: {prof.weight_kg}kg")
                if prof.waist_cm:
                    prof_notes.append(f"Vòng eo: {prof.waist_cm}cm")
                if prof.chest_cm:
                    prof_notes.append(f"Vòng ngực: {prof.chest_cm}cm")
                if prof.hip_cm:
                    prof_notes.append(f"Vòng mông: {prof.hip_cm}cm")
                if prof.gender:
                    prof_notes.append(f"Giới tính: {prof.gender.value}")
                if prof.fit_preference:
                    prof_notes.append(f"Gu mặc: {prof.fit_preference}")
                if prof_notes:
                    sys_content += (
                        "\nThông tin khách hàng đã biết (áp dụng để gợi ý sản phẩm "
                        f"& chọn size): {', '.join(prof_notes)}"
                    )

            prompt_messages = [
                SystemMessage(content=sys_content),
                *state["messages"],
            ]
            response = await self.model_with_tools.ainvoke(prompt_messages)
            return {"messages": [response]}

        # Node 4: Tool execution
        async def node_tools(state: AgentState) -> dict[str, Any]:
            last_msg = state["messages"][-1]
            tools_used = list(state.get("tools_used", []))
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    name = tc["name"]
                    if name not in tools_used:
                        tools_used.append(name)

            tool_node = ToolNode(ALL_TOOLS)
            result = await tool_node.ainvoke(state)
            result["tools_used"] = tools_used
            return result

        # Conditional branching
        def route_clarification(
            state: AgentState,
        ) -> Literal["node_agent", "__end__"]:
            if state.get("clarification_asked"):
                return END
            return "node_agent"

        def route_tools(
            state: AgentState,
        ) -> Literal["node_tools", "__end__"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                return "node_tools"
            return END

        workflow.add_node("node_classify", node_classify)
        workflow.add_node("node_clarification_gate", node_clarification_gate)
        workflow.add_node("node_agent", node_agent)
        workflow.add_node("node_tools", node_tools)

        workflow.add_edge(START, "node_classify")
        workflow.add_edge("node_classify", "node_clarification_gate")
        workflow.add_conditional_edges(
            "node_clarification_gate",
            route_clarification,
            {"node_agent": "node_agent", END: END},
        )
        workflow.add_conditional_edges(
            "node_agent",
            route_tools,
            {"node_tools": "node_tools", END: END},
        )
        workflow.add_edge("node_tools", "node_agent")

        return workflow.compile()

    async def chat(
        self,
        request: ChatRequest,
        user_id: int | None = None,
        auth_token: str | None = None,
    ) -> ChatResponse:
        """Process customer request through conversation graph."""
        session_id = request.session_id or str(uuid.uuid4())
        session_mgr = get_session_manager(
            ttl_seconds=self.settings.SESSION_TTL_SECONDS,
            default_max_turns=self.settings.MAX_SESSION_HISTORY_TURNS,
        )
        session = session_mgr.get_or_create_session(session_id, user_id=user_id)

        # 1. Load baseline shared profile from Backend DB if user_id is known
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
                        fit = pref_data.get("fitPreference")
                        if any([h, w, waist, chest, hip, fit]):
                            session.active_profile = UserProfile(
                                height_cm=h,
                                weight_kg=w,
                                waist_cm=waist,
                                chest_cm=chest,
                                hip_cm=hip,
                                fit_preference=fit,
                            )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch preferences for user #%s: %s", user_id, exc
                )

        # 2. If client passed explicit user_profile, apply it to session
        if request.user_profile is not None:
            session.update_active_profile(request.user_profile)

        # 3. Detect any in-conversation measurements (session-level override)
        spontaneous = extract_measurements_from_text(request.message)
        if spontaneous:
            session.update_active_profile(spontaneous)

        # 4. Populate session history if session is empty and client provided history
        if not session.messages and request.history:
            for item in request.history:
                if item.role == "assistant":
                    session.messages.append(AIMessage(content=item.content))
                else:
                    session.messages.append(HumanMessage(content=item.content))

        # 5. Build prompt messages using sliding window from SessionManager
        history_msgs = session_mgr.get_messages(
            session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
        )
        history_msgs.append(HumanMessage(content=request.message))
        session_mgr.add_user_message(session_id, request.message)

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": session.active_profile,
            "clarification_mode": request.clarification_mode,
            "intent": "GENERAL_CHAT",
            "clarification_asked": False,
            "tools_used": [],
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)
            last_msg = final_state["messages"][-1]
            reply = (
                str(last_msg.content)
                if last_msg.content
                else "Mình có thể hỗ trợ gì thêm cho bạn không?"
            )
            intent = final_state.get("intent")
            clarification_asked = final_state.get("clarification_asked", False)
            products = extract_recommended_products(
                final_state.get("messages", []), reply
            )
            quick_replies = generate_quick_replies(intent, clarification_asked)

            session_mgr.add_assistant_message(
                session_id, reply, product_ids=[p.id for p in products]
            )

            return ChatResponse(
                reply=reply,
                session_id=session_id,
                intent=intent,
                clarification_asked=clarification_asked,
                tools_used=final_state.get("tools_used", []),
                products=products,
                quick_replies=quick_replies,
            )
        except Exception as e:
            logger.error("Error running agent graph: %s", e, exc_info=True)
            return ChatResponse(
                reply=(
                    "Hệ thống tư vấn đang gặp gián đoạn tạm thời. "
                    "Bạn vui lòng gửi lại câu hỏi sau giây lát nhé!"
                ),
                session_id=session_id,
                intent="ERROR",
                clarification_asked=False,
                tools_used=[],
            )

    async def chat_stream(
        self,
        request: ChatRequest,
        user_id: int | None = None,
        auth_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream assistant response tokens and events in real time."""
        session_id = request.session_id or str(uuid.uuid4())
        session_mgr = get_session_manager(
            ttl_seconds=self.settings.SESSION_TTL_SECONDS,
            default_max_turns=self.settings.MAX_SESSION_HISTORY_TURNS,
        )
        session = session_mgr.get_or_create_session(session_id, user_id=user_id)

        # 1. Load baseline shared profile from Backend DB if user_id is known
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
                        fit = pref_data.get("fitPreference")
                        if any([h, w, waist, chest, hip, fit]):
                            session.active_profile = UserProfile(
                                height_cm=h,
                                weight_kg=w,
                                waist_cm=waist,
                                chest_cm=chest,
                                hip_cm=hip,
                                fit_preference=fit,
                            )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch preferences for user #%s: %s", user_id, exc
                )

        # 2. If client passed explicit user_profile, apply it to session
        if request.user_profile is not None:
            session.update_active_profile(request.user_profile)

        # 3. Detect any in-conversation measurements (session-level override)
        spontaneous = extract_measurements_from_text(request.message)
        if spontaneous:
            session.update_active_profile(spontaneous)

        # 4. Populate session history if session is empty and client provided history
        if not session.messages and request.history:
            for item in request.history:
                if item.role == "assistant":
                    session.messages.append(AIMessage(content=item.content))
                else:
                    session.messages.append(HumanMessage(content=item.content))

        # 5. Build prompt messages using sliding window from SessionManager
        history_msgs = session_mgr.get_messages(
            session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
        )
        history_msgs.append(HumanMessage(content=request.message))
        session_mgr.add_user_message(session_id, request.message)

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": session.active_profile,
            "clarification_mode": request.clarification_mode,
            "intent": "GENERAL_CHAT",
            "clarification_asked": False,
            "tools_used": [],
        }

        detected_intent: str | None = None
        tools_used: list[str] = []
        full_reply_parts: list[str] = []

        try:
            async for event in self.graph.astream_events(initial_state, version="v2"):
                event_type = event.get("event")
                event_name = event.get("name", "")

                # 1. Intent detected
                if event_name == "node_classify" and event_type == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "intent" in output:
                        detected_intent = output["intent"]
                        yield {
                            "type": "intent",
                            "intent": detected_intent,
                            "session_id": session_id,
                        }

                # 2. Clarification asked (gate intercepted)
                elif (
                    event_name == "node_clarification_gate"
                    and event_type == "on_chain_end"
                ):
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and output.get("clarification_asked"):
                        msgs = output.get("messages", [])
                        if msgs:
                            clarification_text = str(msgs[-1].content)
                            full_reply_parts.append(clarification_text)
                            yield {
                                "type": "clarification",
                                "content": clarification_text,
                                "session_id": session_id,
                                "intent": detected_intent,
                            }

                # 3. Tool execution start
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    yield {
                        "type": "tool_start",
                        "tool": tool_name,
                        "session_id": session_id,
                    }

                # 4. LLM token streaming
                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        text_chunk = str(chunk.content)
                        full_reply_parts.append(text_chunk)
                        yield {
                            "type": "token",
                            "content": text_chunk,
                            "session_id": session_id,
                        }

            full_reply = "".join(full_reply_parts)
            suggested_products = extract_recommended_products([], full_reply)
            is_clarification = (
                any(p.strip().endswith("?") for p in full_reply_parts)
                and len(tools_used) == 0
            )
            quick_replies = generate_quick_replies(detected_intent, is_clarification)

            session_mgr.add_assistant_message(
                session_id,
                full_reply,
                product_ids=[p.id for p in suggested_products],
            )

            yield {
                "type": "done",
                "reply": full_reply,
                "session_id": session_id,
                "intent": detected_intent,
                "tools_used": tools_used,
                "products": [p.model_dump() for p in suggested_products],
                "quick_replies": quick_replies,
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
