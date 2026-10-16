import logging
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
    ChatRequest,
    ChatResponse,
    ClarificationMode,
    UserProfile,
)
from customer_shop_assistant.services.clarification_engine import (
    ClarificationEngine,
)
from customer_shop_assistant.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý AI bán hàng thông minh của Fashion Shop.
Phong cách giao tiếp:
- Tinh gọn, trọng tâm, sự thật khách quan (fact-first), không văn vẻ.
- Trả lời bằng tiếng Việt lịch sự, thân thiện và nhiệt tình.

Quy định bắt buộc:
1. Tư vấn size: Dùng `recommend_size` để tính theo bảng size quy chuẩn.
2. Tồn kho & Sản phẩm: Dùng `search_products` hoặc `get_product_detail`.
   Tuyệt đối không tự bịa tên sản phẩm hay size còn hàng.
3. Chính sách: Dùng `search_policy` khi hỏi về đổi trả, vận chuyển,
   thanh toán VietQR/COD. Bắt buộc dẫn nguồn chuẩn xác dạng
   `[tên_tệp: tiêu_đề_mục]` (ví dụ: `[shipping.md: Biểu phí vận chuyển]`).
4. Mã giảm giá: Dùng `check_voucher` kiểm tra điều kiện trước khi báo.
"""


class AgentState(TypedDict):
    """LangGraph conversation state."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
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
            prof = state.get("user_profile")
            if prof:
                prof_notes = []
                if prof.height_cm:
                    prof_notes.append(f"Chiều cao: {prof.height_cm}cm")
                if prof.weight_kg:
                    prof_notes.append(f"Cân nặng: {prof.weight_kg}kg")
                if prof.gender:
                    prof_notes.append(f"Giới tính: {prof.gender.value}")
                if prof.fit_preference:
                    prof_notes.append(f"Gu mặc: {prof.fit_preference}")
                if prof_notes:
                    sys_content += (
                        f"\nThông tin khách hàng đã biết: {', '.join(prof_notes)}"
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

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process customer request through conversation graph."""
        session_id = request.session_id or str(uuid.uuid4())

        history_msgs: list[BaseMessage] = []
        for msg in request.history:
            if msg.role == "assistant":
                history_msgs.append(AIMessage(content=msg.content))
            else:
                history_msgs.append(HumanMessage(content=msg.content))

        history_msgs.append(HumanMessage(content=request.message))

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_profile": request.user_profile,
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
            return ChatResponse(
                reply=reply,
                session_id=session_id,
                intent=final_state.get("intent"),
                clarification_asked=final_state.get("clarification_asked", False),
                tools_used=final_state.get("tools_used", []),
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

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream assistant response tokens and events in real time."""
        session_id = request.session_id or str(uuid.uuid4())

        history_msgs: list[BaseMessage] = []
        for msg in request.history:
            if msg.role == "assistant":
                history_msgs.append(AIMessage(content=msg.content))
            else:
                history_msgs.append(HumanMessage(content=msg.content))

        history_msgs.append(HumanMessage(content=request.message))

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_profile": request.user_profile,
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
            yield {
                "type": "done",
                "reply": full_reply,
                "session_id": session_id,
                "intent": detected_intent,
                "tools_used": tools_used,
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
