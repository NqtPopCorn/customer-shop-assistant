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
    OutfitItem,
    OutfitSuggestion,
    StateSnapshotDebug,
    SystemPromptDebug,
    TurnDebugMeta,
    UserProfile,
)
from customer_shop_assistant.schemas.clarification import ClarificationDecision
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
    MAX_MODE_B_TURNS,
    MAX_MODE_C_QUESTIONS,
    GrayZoneEvaluator,
    TurnPolicy,
    evaluate_clarification_policy,
    policy_to_turn_policy,
)
from customer_shop_assistant.services.counterfactual_policy import (
    CounterfactualClarificationPolicy,
)
from customer_shop_assistant.services.outfit_kg_repository import (
    get_outfit_kg_repository,
)
from customer_shop_assistant.services.prompt_builder import (
    SYSTEM_PROMPT_VERSION,
    build_extractor_system_prompt_with_debug,
    build_interview_system_prompt_with_debug,
    build_system_prompt_with_debug,
)
from customer_shop_assistant.services.session_manager import (
    ClarifyPhase,
    SessionManager,
    SessionState,
    get_session_manager,
)
from customer_shop_assistant.services.shop_client import ShopClient
from customer_shop_assistant.services.size_recommender import SizeRecommender
from customer_shop_assistant.services.slot_tracker import (
    STANDARD_SLOTS,
    ConversationSlots,
    TurnIntent,
    extract_conversation_slots,
    extract_conversation_slots_with_lifecycle,
    extract_session_profile_updates,
    format_slots_for_prompt,
)
from customer_shop_assistant.services.snapshot_catalog_index import (
    get_snapshot_catalog_index,
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
from customer_shop_assistant.tools.outfit_tool import outfit_diversity_context

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

_PRODUCT_PRICE_PATTERN = re.compile(
    r"\b\d{1,3}(?:[.,]\d{3})+\s*(?:đ|vnđ|vnd)\b",
    re.IGNORECASE,
)

_SLOT_ID_ALIASES = {
    "gender": "target_gender",
    "body_measurements": "measurements",
}

MAX_RECOMMENDED_PRODUCTS: int = 6


def _with_prompt_debug_metadata(
    config: RunnableConfig | None,
    prompt_debug: SystemPromptDebug,
) -> RunnableConfig:
    """Return a shallow-copied RunnableConfig carrying system_prompt_debug metadata."""
    next_config: RunnableConfig = dict(config) if config else {}
    existing_meta = dict(next_config.get("metadata") or {})
    existing_meta["system_prompt_debug"] = prompt_debug.model_dump()
    next_config["metadata"] = existing_meta
    return next_config


def _extract_product_ids_from_text(text: str) -> list[str]:
    """Extract unique numeric product IDs from [123], ID: [123], or [ID] 123."""
    found: list[str] = []
    for g1, g2 in _PRODUCT_ID_PATTERN.findall(text):
        pid = g1 or g2
        if pid and pid not in found:
            found.append(pid)
    return found


def _extract_normalized_prices(text: str) -> list[str]:
    """Extract VND price amounts normalized to digit strings."""
    return [
        re.sub(r"\D", "", m.group(0)) for m in _PRODUCT_PRICE_PATTERN.finditer(text)
    ]


def _get_prior_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return messages through the latest HumanMessage (excluding AI draft)."""
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return list(messages[: idx + 1])
    return []


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
    clarification_turns: int
    declined_slots: list[str]
    last_asked_slots: list[str]
    topic_switched: bool
    turn_intent: str
    is_shopping_query: bool
    policy_decision: dict[str, Any] | None
    decision_source: str
    decision_reason: str
    asked_slots: list[str]
    tools_used: list[str]
    reply: str
    needs_clarification: bool
    clarification_form: ClarificationForm | None
    product_ids: list[str]
    products: list[ChatRecommendedProduct]
    outfits: list[OutfitSuggestion]
    prefetched_tool: str | None
    prefetched_context: str


class ShopAssistantAgent:
    """Orchestrates chatbot conversation flow using LangGraph and LangChain."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        settings: Settings | None = None,
        session_manager: SessionManager | None = None,
        gray_zone_evaluator: GrayZoneEvaluator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chat_model = chat_model or get_chat_model(self.settings)
        self.session_manager = session_manager
        if gray_zone_evaluator is not None:
            self.gray_zone_evaluator: GrayZoneEvaluator | None = gray_zone_evaluator
        elif self.settings.COUNTERFACTUAL_POLICY_ENABLED:
            try:
                self.gray_zone_evaluator = CounterfactualClarificationPolicy(
                    kg_repo=get_outfit_kg_repository(self.settings),
                    snapshot_index=get_snapshot_catalog_index(self.settings),
                )
            except Exception as cf_err:
                logger.warning(
                    "CounterfactualClarificationPolicy init skipped (%s)", cf_err
                )
                self.gray_zone_evaluator = None
        else:
            self.gray_zone_evaluator = None
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
        self,
        intent: TurnIntent,
        gate_search: bool,
        user_id: int | None = None,
    ) -> list[Any]:
        """Return intent-scoped tools exposed to the LLM for the current turn."""
        if gate_search:
            return []

        if intent == TurnIntent.PRODUCT_SEARCH:
            allowed = {"search_products", "get_product_detail"}
        elif intent == TurnIntent.OUTFIT_ADVICE:
            if (
                self.settings.OUTFIT_KG_ENABLED
                and "suggest_outfit" in self.tools_by_name
            ):
                allowed = {"suggest_outfit", "get_product_detail"}
            else:
                allowed = {"search_products", "get_product_detail"}
        elif intent in (TurnIntent.SIZE_ADVICE, TurnIntent.BODY_FIT_ADVICE):
            allowed = {"recommend_size", "search_products", "get_product_detail"}
        elif intent == TurnIntent.NON_SHOPPING:
            allowed = {"search_policy", "check_voucher", "get_product_detail"}
            if user_id is not None:
                allowed.add("update_customer_preference")
        else:
            allowed = {t.name for t in self._enabled_tools}
            if user_id is None:
                allowed.discard("update_customer_preference")

        return [t for t in self._enabled_tools if t.name in allowed]

    @staticmethod
    def _should_use_structured_output(
        decision: ClarificationDecision,
        is_shopping_query: bool,
        tools_used: list[str],
    ) -> bool:
        """Invoke structured_model ONLY in Mode C Gray Zone when no tool ran."""
        if not is_shopping_query:
            return False
        if (
            "search_products" in tools_used
            or "recommend_size" in tools_used
            or "suggest_outfit" in tools_used
        ):
            return False
        return decision.action == "GRAY_ZONE"

    @staticmethod
    def _latest_human_text(messages: list[BaseMessage]) -> str:
        """Return trimmed content of the latest HumanMessage."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content or "").strip()
        return ""

    @classmethod
    def _user_requested_questions(cls, messages: list[BaseMessage]) -> bool:
        """Return True when customer explicitly asked the bot to ask questions."""
        lower = cls._latest_human_text(messages).lower()
        return any(
            cue in lower
            for cue in (
                "hỏi tôi",
                "hỏi mình",
                "hỏi em",
                "hỏi anh",
                "hỏi chị",
                "hỏi thêm",
            )
        )

    @staticmethod
    def _llm_tried_to_clarify(text: str) -> bool:
        """Detect when an AI reply asks a clarifying question instead of answering."""
        if not text:
            return False
        lower_reply = text.lower()
        return "?" in text or any(
            cue in lower_reply
            for cue in (
                "đối tượng nào",
                "nam hay nữ",
                "cho mình biết",
                "cho shop biết",
                "cần biết thêm",
                "tư vấn chính xác",
                "cung cấp thêm",
                "một số thông tin",
                "thông tin thêm",
                "thông tin chi tiết",
                "vài thông tin",
            )
        )

    @staticmethod
    def _llm_deferred_tool_promise(text: str) -> bool:
        """Detect when an AI reply promises to search later without product IDs."""
        if not text or _extract_product_ids_from_text(text):
            return False
        lower_reply = text.lower()
        return any(
            cue in lower_reply
            for cue in (
                "mình sẽ tìm kiếm",
                "tôi sẽ tìm kiếm",
                "để mình tìm kiếm",
                "sẽ tìm kiếm các sản phẩm",
                "mình sẽ tìm các sản phẩm",
            )
        )

    @staticmethod
    def _has_phantom_form_reference(text: str) -> bool:
        """Detect replies inviting user to click a form when clarification_form=None."""
        if not text:
            return False
        lower_reply = text.lower()
        return any(
            cue in lower_reply
            for cue in (
                "tích chọn",
                "bảng dưới đây",
                "bảng bên dưới",
                "form bên dưới",
                "form dưới đây",
                "chọn nhanh vài tiêu chí",
            )
        )

    @classmethod
    def _should_prefetch(
        cls,
        *,
        mode: ClarificationMode,
        intent: TurnIntent,
        decision: ClarificationDecision,
        policy: TurnPolicy,
        slots: ConversationSlots,
        messages: list[BaseMessage],
    ) -> bool:
        """Prefetch catalog/outfit items before agent_node on high-risk PROCEED."""
        del mode
        if decision.action != "PROCEED" or policy.gate_search:
            return False
        latest_user = cls._latest_human_text(messages)
        if intent == TurnIntent.SIZE_ADVICE:
            return latest_user.startswith("Mình chọn:") and (
                slots.has_complete_measurements(slots.value_of("category"))
            )
        if intent not in (TurnIntent.PRODUCT_SEARCH, TurnIntent.OUTFIT_ADVICE):
            return False
        if slots.has("product_id"):
            return False
        if cls._user_requested_questions(messages):
            return False
        if latest_user.startswith("Mình chọn:"):
            return True
        if (
            intent == TurnIntent.PRODUCT_SEARCH
            and slots.status_of("category") == "PROVIDED"
            and any(isinstance(m, AIMessage) for m in messages)
        ):
            return True
        if (
            intent == TurnIntent.OUTFIT_ADVICE
            and slots.status_of("occasion") == "PROVIDED"
            and any(isinstance(m, AIMessage) for m in messages)
        ):
            return True
        if decision.use_fallback:
            return True
        return False

    @classmethod
    def _build_grounded_fallback_reply(
        cls,
        intent: TurnIntent,
        slots: ConversationSlots,
        outfits: list[OutfitSuggestion],
        products: list[ChatRecommendedProduct],
    ) -> str:
        """Format already-fetched outfits or products without running a second tool."""
        if outfits:
            return cls._format_immediate_outfit_fallback_reply(outfits, slots)
        if products:
            return cls._format_immediate_fallback_reply(products, slots, intent=intent)
        if intent == TurnIntent.OUTFIT_ADVICE:
            return (
                "Hiện tại mình chưa tìm thấy bộ phối đồ nào khớp hoàn toàn "
                "với tiêu chí này trong cửa hàng. Bạn có thể thử dịp mặc "
                "hoặc màu sắc khác để mình gợi ý lại nhé!"
            )
        return (
            "Hiện tại mình chưa tìm thấy sản phẩm nào khớp hoàn toàn "
            "với tiêu chí này trong cửa hàng. Bạn có thể nới rộng "
            "khoảng giá hoặc mô tả kiểu đồ khác để mình tìm lại nhé!"
        )

    def _build_debug_meta(
        self,
        *,
        mode: ClarificationMode,
        executed_phase: str,
        next_phase_value: str,
        gated_turns: int,
        clarification_turns: int,
        topic_switched: bool,
        decision_source: str | None,
        decision_reason: str | None,
        reason_codes: list[str] | None,
        impact_signals: dict[str, Any] | None,
        evidence_rule_ids: list[str] | None,
        asked_slots: list[str],
        gate_search: bool,
        active_profile: UserProfile | None,
        slots: ConversationSlots,
        intent: TurnIntent,
        context_messages_count: int,
        latency_ms: int,
        usage_meta: Any,
        runs: list[Any],
        user_id: int | None = None,
    ) -> TurnDebugMeta | None:
        """Build TurnDebugMeta capturing executed phase, decision provenance, traces."""
        if not self.settings.DEBUG_CHAT_META:
            return None
        active_tools_names = [
            t.name
            for t in self._get_active_tools_for_turn(
                intent, gate_search, user_id=user_id
            )
        ]
        state_snapshot = StateSnapshotDebug(
            clarify_phase=executed_phase,
            next_phase=next_phase_value,
            gated_turns=gated_turns,
            clarification_turns=clarification_turns,
            topic_switched=topic_switched,
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
            clarification_turns=clarification_turns,
            decision_source=decision_source,
            decision_reason=decision_reason,
            reason_codes=list(reason_codes or []),
            impact_signals=dict(impact_signals or {}),
            evidence_rule_ids=list(evidence_rule_ids or []),
            asked_slots=asked_slots,
            tools_available=active_tools_names,
            latency_ms=latency_ms,
            usage_metadata=usage_meta,
            runs=runs,
            state_snapshot=state_snapshot,
        )

    def _resolve_state_decision(
        self,
        state: AgentState,
        mode: ClarificationMode,
        intent: TurnIntent,
        slots: ConversationSlots,
    ) -> ClarificationDecision:
        raw_dec = state.get("policy_decision")
        if raw_dec:
            return ClarificationDecision.model_validate(raw_dec)
        return evaluate_clarification_policy(
            mode=mode,
            intent=intent,
            slots=slots,
            clarification_turns=state.get("clarification_turns", 0),
            gray_zone_evaluator=self.gray_zone_evaluator,
        )

    def _create_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph workflow with tool execution and memory."""
        builder = StateGraph(AgentState)

        async def prefetch_node(
            state: AgentState, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            slots, intent = extract_conversation_slots(
                state["messages"],
                state.get("user_profile"),
                declined_slots=set(state.get("declined_slots") or []),
                last_asked_slots=state.get("last_asked_slots") or [],
            )
            decision = self._resolve_state_decision(state, mode, intent, slots)
            policy = policy_to_turn_policy(decision, mode, intent)

            if not self._should_prefetch(
                mode=mode,
                intent=intent,
                decision=decision,
                policy=policy,
                slots=slots,
                messages=state["messages"],
            ):
                return {}

            tools_used = list(state.get("tools_used") or [])
            if (
                intent == TurnIntent.SIZE_ADVICE
                and "recommend_size" in self.tools_by_name
            ):
                size_tool = self.tools_by_name["recommend_size"]
                args = self._build_size_tool_args(slots)
                try:
                    tool_res = await size_tool.ainvoke(args, config=config)
                    if "recommend_size" not in tools_used:
                        tools_used.append("recommend_size")
                    combined_ctx = str(tool_res)
                    size_match = re.search(
                        r"Size đề xuất:\s*(XXL|XL|XS|[SML]|2[89]|3[0-6])\b",
                        combined_ctx,
                        re.IGNORECASE,
                    )
                    if size_match and "search_products" in self.tools_by_name:
                        primary_size = size_match.group(1).upper()
                        search_args = self._build_immediate_search_args(
                            slots, state["messages"]
                        )
                        search_args["size"] = primary_size
                        search_args["limit"] = 4
                        try:
                            search_res = await self.tools_by_name[
                                "search_products"
                            ].ainvoke(search_args, config=config)
                            if "search_products" not in tools_used:
                                tools_used.append("search_products")
                            combined_ctx = (
                                f"{combined_ctx}\n\n"
                                f"SẢN PHẨM GỢI Ý CÓ SẴN SIZE {primary_size}:\n"
                                f"{search_res}"
                            )
                        except Exception as search_exc:
                            logger.warning(
                                "Prefetch size search_products failed: %s", search_exc
                            )
                    return {
                        "tools_used": tools_used,
                        "prefetched_tool": "recommend_size",
                        "prefetched_context": combined_ctx,
                    }
                except Exception as exc:
                    logger.warning("Prefetch recommend_size failed: %s", exc)
                return {}

            if (
                intent == TurnIntent.OUTFIT_ADVICE
                and self.settings.OUTFIT_KG_ENABLED
                and "suggest_outfit" in self.tools_by_name
            ):
                outfit_tool = self.tools_by_name["suggest_outfit"]
                args = self._build_outfit_tool_args(
                    {},
                    slots,
                    state.get("user_id"),
                    use_casual_default=decision.use_fallback,
                )
                try:
                    tool_res = await outfit_tool.ainvoke(args, config=config)
                    if "suggest_outfit" not in tools_used:
                        tools_used.append("suggest_outfit")
                    return {
                        "tools_used": tools_used,
                        "prefetched_tool": "suggest_outfit",
                        "prefetched_context": str(tool_res),
                    }
                except Exception as exc:
                    logger.warning("Prefetch suggest_outfit failed: %s", exc)
                return {}

            search_tool = self.tools_by_name.get("search_products")
            if search_tool is not None:
                args = self._build_immediate_search_args(slots, state["messages"])
                try:
                    if not slots.has("target_gender"):
                        with balanced_gender_search_context(
                            occasion=slots.value_of("occasion")
                        ):
                            tool_res = await search_tool.ainvoke(args, config=config)
                    else:
                        tool_res = await search_tool.ainvoke(args, config=config)
                    if "search_products" not in tools_used:
                        tools_used.append("search_products")
                    return {
                        "tools_used": tools_used,
                        "prefetched_tool": "search_products",
                        "prefetched_context": str(tool_res),
                    }
                except Exception as exc:
                    logger.warning("Prefetch search_products failed: %s", exc)
            return {}

        async def agent_node(
            state: AgentState, config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            slots, intent = extract_conversation_slots(
                state["messages"],
                state.get("user_profile"),
                declined_slots=set(state.get("declined_slots") or []),
                last_asked_slots=state.get("last_asked_slots") or [],
            )
            decision = self._resolve_state_decision(state, mode, intent, slots)
            policy = policy_to_turn_policy(decision, mode, intent)
            is_shopping_query = intent != TurnIntent.NON_SHOPPING

            if policy.gate_search:
                bound_model = self.chat_model
                slot_summary = format_slots_for_prompt(
                    slots, intent, include_missing=False
                )
                sys_prompt, prompt_debug = build_interview_system_prompt_with_debug(
                    user_profile=state.get("user_profile"),
                    slot_summary=slot_summary,
                )
            else:
                active_tools = self._get_active_tools_for_turn(
                    intent, policy.gate_search, user_id=state.get("user_id")
                )
                try:
                    bound_model = self.chat_model.bind_tools(active_tools)
                except (NotImplementedError, AttributeError):
                    bound_model = self.chat_model

                include_missing = mode != ClarificationMode.IMMEDIATE_ANSWER
                slot_summary = (
                    format_slots_for_prompt(
                        slots, intent, include_missing=include_missing
                    )
                    if is_shopping_query
                    else ""
                )
                sys_prompt, prompt_debug = build_system_prompt_with_debug(
                    user_profile=state.get("user_profile"),
                    prompt_key=policy.prompt_key,
                    slot_summary=slot_summary,
                    clarification_mode=mode,
                )
                prefetched_ctx = str(state.get("prefetched_context") or "").strip()
                if prefetched_ctx:
                    if state.get("prefetched_tool") == "recommend_size":
                        sys_prompt = (
                            f"{sys_prompt}\n\n"
                            "KẾT QUẢ TƯ VẤN SIZE ĐÃ TÍNH SẴN TỪ BẢNG SIZE CHUẨN "
                            "(hãy dùng trực tiếp thông tin dưới đây để tư vấn size "
                            "rõ ràng, tự nhiên cho khách, KHÔNG cần gọi lại tool "
                            "`recommend_size` và KHÔNG hỏi lại số đo):\n"
                            f"{prefetched_ctx}"
                        )
                    else:
                        sys_prompt = (
                            f"{sys_prompt}\n\n"
                            "DỮ LIỆU SẢN PHẨM THỰC TẾ ĐÃ TRUY XUẤT SẴN CHO LƯỢT NÀY "
                            "(hãy dùng trực tiếp thông tin dưới đây để tư vấn tự nhiên "
                            "cho khách, KHÔNG cần gọi lại tool tìm kiếm tương tự và "
                            "tuyệt đối KHÔNG tự bịa thêm mã [ID] hay giá tiền ngoài "
                            "danh sách):\n"
                            f"{prefetched_ctx}"
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

            invoke_config = (
                _with_prompt_debug_metadata(config, prompt_debug)
                if self.settings.DEBUG_CHAT_META
                else config
            )
            response = await bound_model.ainvoke(full_prompt_msgs, config=invoke_config)
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
            turn_slots, turn_intent = extract_conversation_slots(
                state["messages"],
                state.get("user_profile"),
                declined_slots=set(state.get("declined_slots") or []),
                last_asked_slots=state.get("last_asked_slots") or [],
            )
            decision = self._resolve_state_decision(
                state, mode, turn_intent, turn_slots
            )
            policy = policy_to_turn_policy(decision, mode, turn_intent)

            active_tool_names = (
                set() if policy.gate_search else {t.name for t in self._enabled_tools}
            )
            needs_balanced_gender = not turn_slots.has("target_gender") and (
                mode == ClarificationMode.IMMEDIATE_ANSWER or decision.use_fallback
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
                        t_args,
                        turn_slots,
                        state.get("user_id"),
                        use_casual_default=decision.use_fallback,
                    )
                elif t_name == "suggest_outfit":
                    t_args = self._build_outfit_tool_args(
                        t_args,
                        turn_slots,
                        state.get("user_id"),
                        use_casual_default=decision.use_fallback,
                    )

                if t_name not in active_tool_names:
                    logger.warning(
                        "Tool '%s' is gated in turn, returning polite error",
                        t_name,
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
                            if t_name == state.get("prefetched_tool") and state.get(
                                "prefetched_context"
                            ):
                                t_result = state["prefetched_context"]
                            elif t_name == "search_products" and needs_balanced_gender:
                                effective_args = self._sanitize_mode_a_balanced_args(
                                    t_args, turn_slots, state["messages"]
                                )
                                with balanced_gender_search_context(
                                    occasion=turn_slots.value_of("occasion")
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
                state["messages"],
                state.get("user_profile"),
                declined_slots=set(state.get("declined_slots") or []),
                last_asked_slots=state.get("last_asked_slots") or [],
            )
            is_shopping_query = intent != TurnIntent.NON_SHOPPING
            tools_used = list(state.get("tools_used") or [])
            mode = state.get(
                "clarification_mode", ClarificationMode.CONDITIONAL_CLARIFICATION
            )
            phase = state.get("clarify_phase", ClarifyPhase.INTERVIEW)
            decision = self._resolve_state_decision(state, mode, intent, slots)

            allow_clarification = is_shopping_query and decision.action in (
                "ASK",
                "GRAY_ZONE",
            )
            use_structured_output = self._should_use_structured_output(
                decision=decision,
                is_shopping_query=is_shopping_query,
                tools_used=tools_used,
            )
            prior_messages = _get_prior_turn_messages(state["messages"])
            grounded_refs = list(prior_messages)
            prefetched_ctx = str(state.get("prefetched_context") or "").strip()
            if prefetched_ctx:
                grounded_refs.append(
                    ToolMessage(content=prefetched_ctx, tool_call_id="prefetch")
                )
            executed_react_tool = any(
                isinstance(m, ToolMessage)
                for m in state["messages"][len(prior_messages) :]
            )
            can_check_ungrounded = not executed_react_tool

            agent_reply = await self._generate_agent_reply(
                state["messages"],
                last_reply_text=last_text,
                is_shopping_query=is_shopping_query,
                tools_used=tools_used,
                allow_clarification=allow_clarification,
                use_structured_output=use_structured_output,
                prior_messages=grounded_refs,
                config=config,
            )

            outfits, products = await self._collect_turn_outputs(
                tools_used=tools_used,
                product_ids=agent_reply.product_ids,
                reply_text=last_text or agent_reply.reply,
            )
            agent_reply.product_ids = [str(p.id) for p in products]

            decision_source = decision.decision_source
            decision_reason = decision.decision_reason
            asked_slots: list[str] = []

            if not is_shopping_query:
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None
                decision_source = "non_shopping_bypass"
            elif mode == ClarificationMode.IMMEDIATE_ANSWER:
                (
                    outfits,
                    products,
                    asked_slots,
                    decision_source,
                    decision_reason,
                ) = await self._finalize_mode_a(
                    intent=intent,
                    slots=slots,
                    decision=decision,
                    agent_reply=agent_reply,
                    last_text=last_text,
                    tools_used=tools_used,
                    can_check_ungrounded=can_check_ungrounded,
                    grounded_refs=grounded_refs,
                    outfits=outfits,
                    products=products,
                    messages=state["messages"],
                    user_id=state.get("user_id"),
                    prefetched_tool=state.get("prefetched_tool"),
                    config=config,
                )
            elif mode == ClarificationMode.ALWAYS_CLARIFY:
                (
                    outfits,
                    products,
                    asked_slots,
                    decision_source,
                    decision_reason,
                ) = await self._finalize_mode_b(
                    intent=intent,
                    slots=slots,
                    decision=decision,
                    agent_reply=agent_reply,
                    last_text=last_text,
                    tools_used=tools_used,
                    can_check_ungrounded=can_check_ungrounded,
                    grounded_refs=grounded_refs,
                    outfits=outfits,
                    products=products,
                    messages=state["messages"],
                    user_id=state.get("user_id"),
                    clarification_turns=state.get("clarification_turns", 0),
                    prefetched_tool=state.get("prefetched_tool"),
                    config=config,
                )
            else:
                (
                    outfits,
                    products,
                    asked_slots,
                    decision_source,
                    decision_reason,
                ) = await self._finalize_mode_c(
                    intent=intent,
                    slots=slots,
                    decision=decision,
                    agent_reply=agent_reply,
                    last_text=last_text,
                    tools_used=tools_used,
                    can_check_ungrounded=can_check_ungrounded,
                    grounded_refs=grounded_refs,
                    outfits=outfits,
                    products=products,
                    messages=state["messages"],
                    user_id=state.get("user_id"),
                    prefetched_tool=state.get("prefetched_tool"),
                    config=config,
                )

            agent_reply.reply = agent_reply.reply.strip()

            session_id = state["session_id"]
            sess_mgr = self.session_manager or get_session_manager(
                ttl_seconds=self.settings.SESSION_TTL_SECONDS,
                default_max_turns=self.settings.MAX_SESSION_HISTORY_TURNS,
            )
            sess = sess_mgr.get_session(session_id)

            new_phase = phase
            if is_shopping_query:
                if not agent_reply.needs_clarification:
                    new_phase = ClarifyPhase.OPEN
                elif mode == ClarificationMode.ALWAYS_CLARIFY:
                    next_clar_turns = (
                        (sess.clarification_turns + 1)
                        if sess is not None
                        else state.get("clarification_turns", 0) + 1
                    )
                    new_phase = (
                        ClarifyPhase.OPEN
                        if next_clar_turns >= MAX_MODE_B_TURNS
                        else ClarifyPhase.INTERVIEW
                    )

            if sess:
                sess.clarification_mode = str(
                    mode.value if hasattr(mode, "value") else mode
                )
                sess.clarify_phase = new_phase
                if agent_reply.needs_clarification and asked_slots:
                    sess.clarification_turns += 1
                    sess.last_asked_slots = list(asked_slots)
                else:
                    sess.last_asked_slots = []
                if outfits:
                    sess.record_shown_outfits(outfits)
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
                "decision_source": decision_source,
                "decision_reason": decision_reason,
                "asked_slots": asked_slots,
            }

        builder.add_node("prefetch", prefetch_node)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)
        builder.add_node("finalize", finalize_node)

        builder.add_edge(START, "prefetch")
        builder.add_edge("prefetch", "agent")
        builder.add_conditional_edges(
            "agent", route_tools, {"tools": "tools", "finalize": "finalize"}
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=MemorySaver())

    @staticmethod
    def _enrich_recommended_from_outfit_item(
        cached: ChatRecommendedProduct, item: OutfitItem
    ) -> ChatRecommendedProduct:
        """Override default product preview with the chosen outfit variant data."""
        eff_price = item.price if item.price > 0 else cached.price
        orig_price = cached.original_price
        discount_pct = cached.discount_percent
        if orig_price is not None and orig_price > eff_price:
            discount_pct = round(((orig_price - eff_price) / orig_price) * 100)
        elif orig_price is not None and orig_price <= eff_price:
            orig_price = None
            discount_pct = None

        return cached.model_copy(
            update={
                "image_url": item.image_url or cached.image_url,
                "variant_id": item.variant_id,
                "color": item.color,
                "size": item.size,
                "price": eff_price,
                "original_price": orig_price,
                "discount_percent": discount_pct,
            }
        )

    async def _collect_turn_outputs(
        self,
        *,
        tools_used: list[str],
        product_ids: list[str],
        reply_text: str,
    ) -> tuple[list[OutfitSuggestion], list[ChatRecommendedProduct]]:
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
                        outfit_products.append(
                            self._enrich_recommended_from_outfit_item(cached, item)
                        )

        if "search_products" in tools_used or "get_product_detail" in tools_used:
            searched_ids = (
                get_latest_searched_product_ids()
                if "search_products" in tools_used
                else None
            )
            resolved_catalog = await self._resolve_products(
                product_ids,
                reply_text=reply_text,
                fallback_product_ids=searched_ids,
            )
        else:
            resolved_catalog = []

        if outfit_products:
            seen_pids = {p.id for p in outfit_products}
            products = (
                outfit_products + [p for p in resolved_catalog if p.id not in seen_pids]
            )[:MAX_RECOMMENDED_PRODUCTS]
        else:
            products = resolved_catalog[:MAX_RECOMMENDED_PRODUCTS]
        return outfits, products

    async def _attach_or_build_fallback(
        self,
        *,
        intent: TurnIntent,
        slots: ConversationSlots,
        messages: list[BaseMessage],
        user_id: int | None,
        tools_used: list[str],
        use_casual_default: bool,
        outfits: list[OutfitSuggestion],
        products: list[ChatRecommendedProduct],
        replace_reply: bool,
        prefetched_tool: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[list[OutfitSuggestion], list[ChatRecommendedProduct], str | None]:
        """Reuse prefetched items or fetch fallback; replace reply only on violation."""
        if outfits or products or prefetched_tool is not None:
            fb_reply = (
                self._build_grounded_fallback_reply(intent, slots, outfits, products)
                if replace_reply
                else None
            )
            return outfits, products, fb_reply

        fb_outfits, fb_products, fb_reply = await self._execute_grounded_fallback(
            intent=intent,
            slots=slots,
            messages=messages,
            user_id=user_id,
            tools_used=tools_used,
            use_casual_default=use_casual_default,
            config=config,
        )
        return fb_outfits, fb_products, (fb_reply if replace_reply else None)

    async def _finalize_mode_a(
        self,
        *,
        intent: TurnIntent,
        slots: ConversationSlots,
        decision: ClarificationDecision,
        agent_reply: AgentReply,
        last_text: str,
        tools_used: list[str],
        can_check_ungrounded: bool,
        grounded_refs: list[BaseMessage],
        outfits: list[OutfitSuggestion],
        products: list[ChatRecommendedProduct],
        messages: list[BaseMessage],
        user_id: int | None,
        prefetched_tool: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[
        list[OutfitSuggestion],
        list[ChatRecommendedProduct],
        list[str],
        str,
        str,
    ]:
        """Mode A (IMMEDIATE_ANSWER): strict zero-clarification & anti-fabrication."""
        agent_reply.needs_clarification = False
        agent_reply.clarification_form = None

        if intent == TurnIntent.SIZE_ADVICE and not slots.has_complete_measurements(
            slots.value_of("category")
        ):
            agent_reply.reply = SizeRecommender(
                settings=self.settings
            ).format_size_chart(
                category=slots.value_of("category"),
                gender=slots.value_of("target_gender"),
            )
            return (
                outfits,
                products,
                [],
                decision.decision_source,
                decision.decision_reason,
            )

        if intent in (
            TurnIntent.PRODUCT_SEARCH,
            TurnIntent.OUTFIT_ADVICE,
            TurnIntent.BODY_FIT_ADVICE,
        ):
            raw_text = last_text or agent_reply.reply
            has_ungrounded = can_check_ungrounded and (
                self._has_ungrounded_product_listing(
                    raw_text, prior_messages=grounded_refs
                )
            )
            cited_ids = set(_extract_product_ids_from_text(raw_text))
            grounded_ids = {str(p.id) for p in products}
            has_cited_grounded = bool(cited_ids & grounded_ids)
            tried_clarify = (
                self._llm_tried_to_clarify(raw_text) and not has_cited_grounded
            )
            has_phantom_form = self._has_phantom_form_reference(raw_text)
            if has_ungrounded or tried_clarify or has_phantom_form:
                eff_prefetched = prefetched_tool or (
                    tools_used[0] if not can_check_ungrounded and tools_used else None
                )
                outfits, products, maybe_reply = await self._attach_or_build_fallback(
                    intent=intent,
                    slots=slots,
                    messages=messages,
                    user_id=user_id,
                    tools_used=tools_used,
                    use_casual_default=decision.use_fallback,
                    outfits=outfits,
                    products=products,
                    replace_reply=True,
                    prefetched_tool=eff_prefetched,
                    config=config,
                )
                agent_reply.product_ids = [str(p.id) for p in products]
                if maybe_reply:
                    agent_reply.reply = maybe_reply

        return outfits, products, [], decision.decision_source, decision.decision_reason

    async def _finalize_mode_b(
        self,
        *,
        intent: TurnIntent,
        slots: ConversationSlots,
        decision: ClarificationDecision,
        agent_reply: AgentReply,
        last_text: str,
        tools_used: list[str],
        can_check_ungrounded: bool,
        grounded_refs: list[BaseMessage],
        outfits: list[OutfitSuggestion],
        products: list[ChatRecommendedProduct],
        messages: list[BaseMessage],
        user_id: int | None,
        clarification_turns: int,
        prefetched_tool: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[
        list[OutfitSuggestion],
        list[ChatRecommendedProduct],
        list[str],
        str,
        str,
    ]:
        """Mode B (ALWAYS_CLARIFY): 2-turn interview form & card-safe PROCEED."""
        if decision.action == "ASK":
            agent_reply.needs_clarification = True
            allow_reconfirm = any(
                rc
                in ("UNFULFILLABLE_HARD_CONSTRAINT", "UNFULFILLABLE_COLOR_CONSTRAINT")
                for rc in decision.reason_codes
            )
            form = build_clarification_form(
                slots,
                intent,
                max_questions=decision.max_questions or 3,
                target_slots=decision.target_slots,
                allow_provided=allow_reconfirm,
            )
            asked_slots: list[str] = []
            if form.questions:
                agent_reply.clarification_form = form
                agent_reply.reply = planner_align_reply(
                    agent_reply.reply, form, slots=slots, intent=intent
                )
                asked_slots = [
                    _SLOT_ID_ALIASES.get(q.id.strip().lower(), q.id.strip().lower())
                    for q in form.questions
                ]
            else:
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None
            return (
                outfits,
                products,
                asked_slots,
                decision.decision_source,
                decision.decision_reason,
            )

        agent_reply.needs_clarification = False
        agent_reply.clarification_form = None

        if (
            intent == TurnIntent.SIZE_ADVICE
            and not slots.has_complete_measurements(slots.value_of("category"))
            and (decision.use_fallback or "recommend_size" not in tools_used)
        ):
            agent_reply.reply = SizeRecommender(
                settings=self.settings
            ).format_size_chart(
                category=slots.value_of("category"),
                gender=slots.value_of("target_gender"),
            )
            return (
                outfits,
                products,
                [],
                decision.decision_source,
                decision.decision_reason,
            )

        if (
            intent
            in (
                TurnIntent.PRODUCT_SEARCH,
                TurnIntent.OUTFIT_ADVICE,
                TurnIntent.BODY_FIT_ADVICE,
            )
            and can_check_ungrounded
        ):
            raw_text = last_text or agent_reply.reply
            has_ungrounded = self._has_ungrounded_product_listing(
                raw_text, prior_messages=grounded_refs
            )
            is_clarify_loop = (
                not self._user_requested_questions(messages)
                and not outfits
                and not products
                and (
                    self._llm_tried_to_clarify(raw_text)
                    or self._llm_deferred_tool_promise(raw_text)
                )
            )
            has_phantom_form = self._has_phantom_form_reference(raw_text)
            if has_ungrounded or is_clarify_loop or has_phantom_form:
                outfits, products, maybe_reply = await self._attach_or_build_fallback(
                    intent=intent,
                    slots=slots,
                    messages=messages,
                    user_id=user_id,
                    tools_used=tools_used,
                    use_casual_default=decision.use_fallback,
                    outfits=outfits,
                    products=products,
                    replace_reply=True,
                    prefetched_tool=prefetched_tool,
                    config=config,
                )
                agent_reply.product_ids = [str(p.id) for p in products]
                if maybe_reply:
                    agent_reply.reply = maybe_reply
            elif (
                decision.use_fallback
                and not outfits
                and not products
                and prefetched_tool is None
            ):
                # Attach cards only; preserve the LLM reply verbatim
                outfits, products, _ = await self._attach_or_build_fallback(
                    intent=intent,
                    slots=slots,
                    messages=messages,
                    user_id=user_id,
                    tools_used=tools_used,
                    use_casual_default=True,
                    outfits=outfits,
                    products=products,
                    replace_reply=False,
                    prefetched_tool=prefetched_tool,
                    config=config,
                )
                agent_reply.product_ids = [str(p.id) for p in products]

        return outfits, products, [], decision.decision_source, decision.decision_reason

    async def _finalize_mode_c(
        self,
        *,
        intent: TurnIntent,
        slots: ConversationSlots,
        decision: ClarificationDecision,
        agent_reply: AgentReply,
        last_text: str,
        tools_used: list[str],
        can_check_ungrounded: bool,
        grounded_refs: list[BaseMessage],
        outfits: list[OutfitSuggestion],
        products: list[ChatRecommendedProduct],
        messages: list[BaseMessage],
        user_id: int | None,
        prefetched_tool: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[
        list[OutfitSuggestion],
        list[ChatRecommendedProduct],
        list[str],
        str,
        str,
    ]:
        """Mode C (CONDITIONAL_CLARIFICATION): Guard / Counterfactual / Gray Zone."""
        decision_source = decision.decision_source
        decision_reason = decision.decision_reason

        if decision.action == "ASK":
            agent_reply.needs_clarification = True
            allow_reconfirm = any(
                rc
                in ("UNFULFILLABLE_HARD_CONSTRAINT", "UNFULFILLABLE_COLOR_CONSTRAINT")
                for rc in decision.reason_codes
            )
            form = build_clarification_form(
                slots,
                intent,
                max_questions=decision.max_questions or 2,
                target_slots=decision.target_slots,
                allow_provided=allow_reconfirm,
            )
            asked_slots: list[str] = []
            if form.questions:
                agent_reply.clarification_form = form
                agent_reply.reply = planner_align_reply(
                    agent_reply.reply, form, slots=slots, intent=intent
                )
                asked_slots = [
                    _SLOT_ID_ALIASES.get(q.id.strip().lower(), q.id.strip().lower())
                    for q in form.questions
                ]
            else:
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None
            return outfits, products, asked_slots, decision_source, decision_reason

        if decision.action == "PROCEED":
            agent_reply.needs_clarification = False
            agent_reply.clarification_form = None

            if (
                intent == TurnIntent.SIZE_ADVICE
                and not slots.has_complete_measurements(slots.value_of("category"))
                and (decision.use_fallback or "recommend_size" not in tools_used)
            ):
                agent_reply.reply = SizeRecommender(
                    settings=self.settings
                ).format_size_chart(
                    category=slots.value_of("category"),
                    gender=slots.value_of("target_gender"),
                )
                return outfits, products, [], decision_source, decision_reason

            if (
                intent
                in (
                    TurnIntent.PRODUCT_SEARCH,
                    TurnIntent.OUTFIT_ADVICE,
                    TurnIntent.BODY_FIT_ADVICE,
                )
                and can_check_ungrounded
            ):
                raw_text = last_text or agent_reply.reply
                is_clarify_loop = (
                    not self._user_requested_questions(messages)
                    and not outfits
                    and not products
                    and (
                        self._llm_tried_to_clarify(raw_text)
                        or self._llm_deferred_tool_promise(raw_text)
                    )
                )
                if (
                    self._has_ungrounded_product_listing(
                        raw_text, prior_messages=grounded_refs
                    )
                    or is_clarify_loop
                    or self._has_phantom_form_reference(raw_text)
                ):
                    (
                        outfits,
                        products,
                        maybe_reply,
                    ) = await self._attach_or_build_fallback(
                        intent=intent,
                        slots=slots,
                        messages=messages,
                        user_id=user_id,
                        tools_used=tools_used,
                        use_casual_default=decision.use_fallback,
                        outfits=outfits,
                        products=products,
                        replace_reply=True,
                        prefetched_tool=prefetched_tool,
                        config=config,
                    )
                    agent_reply.product_ids = [str(p.id) for p in products]
                    if maybe_reply:
                        agent_reply.reply = maybe_reply
                elif (
                    decision.use_fallback
                    and not outfits
                    and not products
                    and prefetched_tool is None
                ):
                    outfits, products, _ = await self._attach_or_build_fallback(
                        intent=intent,
                        slots=slots,
                        messages=messages,
                        user_id=user_id,
                        tools_used=tools_used,
                        use_casual_default=True,
                        outfits=outfits,
                        products=products,
                        replace_reply=False,
                        prefetched_tool=prefetched_tool,
                        config=config,
                    )
                    agent_reply.product_ids = [str(p.id) for p in products]
            return outfits, products, [], decision_source, decision_reason

        # Mode C Gray Zone
        asked_slots = []
        if agent_reply.needs_clarification:
            validated_form = planner_validate_or_repair(
                agent_reply.clarification_form,
                slots=slots,
                intent=intent,
                force_planner=False,
                max_fallback_questions=MAX_MODE_C_QUESTIONS,
                max_questions=MAX_MODE_C_QUESTIONS,
                target_slots=decision.target_slots,
            )
            if validated_form.questions:
                agent_reply.clarification_form = validated_form
                agent_reply.reply = planner_align_reply(
                    agent_reply.reply, validated_form, slots=slots, intent=intent
                )
                asked_slots = [
                    _SLOT_ID_ALIASES.get(q.id.strip().lower(), q.id.strip().lower())
                    for q in validated_form.questions
                ]
                decision_source = "mode_c_llm"
                decision_reason = (
                    agent_reply.decision_reason
                    or "LLM requested high-impact clarification in gray zone."
                )
            else:
                agent_reply.needs_clarification = False
                agent_reply.clarification_form = None

        if not agent_reply.needs_clarification:
            decision_source = "mode_c_llm"
            decision_reason = agent_reply.decision_reason or (
                "LLM answered directly using available context/anchors."
            )
            raw_gray_text = last_text or agent_reply.reply
            if (
                intent
                in (
                    TurnIntent.PRODUCT_SEARCH,
                    TurnIntent.OUTFIT_ADVICE,
                    TurnIntent.BODY_FIT_ADVICE,
                )
                and can_check_ungrounded
                and (
                    self._has_ungrounded_product_listing(
                        raw_gray_text, prior_messages=grounded_refs
                    )
                    or self._has_phantom_form_reference(raw_gray_text)
                )
            ):
                outfits, products, maybe_reply = await self._attach_or_build_fallback(
                    intent=intent,
                    slots=slots,
                    messages=messages,
                    user_id=user_id,
                    tools_used=tools_used,
                    use_casual_default=False,
                    outfits=outfits,
                    products=products,
                    replace_reply=True,
                    prefetched_tool=prefetched_tool,
                    config=config,
                )
                agent_reply.product_ids = [str(p.id) for p in products]
                if maybe_reply:
                    agent_reply.reply = maybe_reply

        return outfits, products, asked_slots, decision_source, decision_reason

    async def _execute_grounded_fallback(
        self,
        *,
        intent: TurnIntent,
        slots: ConversationSlots,
        messages: list[BaseMessage],
        user_id: int | None,
        tools_used: list[str],
        use_casual_default: bool,
        config: RunnableConfig | None = None,
    ) -> tuple[list[OutfitSuggestion], list[ChatRecommendedProduct], str]:
        if (
            intent == TurnIntent.OUTFIT_ADVICE
            and self.settings.OUTFIT_KG_ENABLED
            and "suggest_outfit" in self.tools_by_name
        ):
            outfits, products, fb_reply = await self._run_verified_fallback_outfit(
                slots=slots,
                user_id=user_id,
                tools_used=tools_used,
                use_casual_default=use_casual_default,
                config=config,
            )
            if not fb_reply:
                fb_reply = (
                    "Hiện tại mình chưa tìm thấy bộ phối đồ nào khớp hoàn toàn "
                    "với tiêu chí này trong cửa hàng. Bạn có thể thử dịp mặc "
                    "hoặc màu sắc khác để mình gợi ý lại nhé!"
                )
            return outfits, products, fb_reply

        products = await self._run_verified_fallback_search(
            slots=slots,
            messages=messages,
            tools_used=tools_used,
            config=config,
        )
        if products:
            reply = self._format_immediate_fallback_reply(
                products, slots, intent=intent
            )
        else:
            reply = (
                "Hiện tại mình chưa tìm thấy sản phẩm nào khớp hoàn toàn "
                "với tiêu chí này trong cửa hàng. Bạn có thể nới rộng "
                "khoảng giá hoặc mô tả kiểu đồ khác để mình tìm lại nhé!"
            )
        return [], products, reply

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
        ClarificationDecision,
        TurnPolicy,
        ConversationSlots,
        TurnIntent,
        bool,
    ]:
        """Resolve session, profile, lifecycle slots, and ClarificationDecision."""
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

        slots, intent, topic_switched = extract_conversation_slots_with_lifecycle(
            session.messages,
            session.active_profile,
            declined_slots=session.declined_slots,
            last_asked_slots=session.last_asked_slots,
        )
        if topic_switched:
            session.reset_task_clarification()

        for slot_name in STANDARD_SLOTS:
            if slots.is_declined(slot_name):
                session.declined_slots.add(slot_name)
            else:
                raw_s = slots._raw_slot(slot_name)
                if raw_s is not None and raw_s.status == "PROVIDED":
                    session.declined_slots.discard(slot_name)

        profile_updates = extract_session_profile_updates(slots)
        if profile_updates:
            session.update_active_profile(profile_updates)

        is_shopping_query = intent != TurnIntent.NON_SHOPPING
        decision = evaluate_clarification_policy(
            mode=mode,
            intent=intent,
            slots=slots,
            clarification_turns=session.clarification_turns,
            gray_zone_evaluator=self.gray_zone_evaluator,
        )
        policy = policy_to_turn_policy(decision, mode, intent)

        if policy.gate_search:
            session.clarify_phase = ClarifyPhase.INTERVIEW
            session.gated_turns += 1
        elif is_shopping_query:
            session.clarify_phase = ClarifyPhase.OPEN

        if is_shopping_query:
            session.last_shopping_intent = intent.value

        return (
            session_id,
            session_mgr,
            session,
            mode,
            decision,
            policy,
            slots,
            intent,
            topic_switched,
        )

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
        """Strip LLM-hallucinated gender/category biases for balanced multi-gender."""
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
    def _build_size_tool_args(slots: ConversationSlots) -> dict[str, Any]:
        """Build deterministic recommend_size arguments from extracted slots."""
        cat_raw = slots.value_of("category")
        if cat_raw == "dress":
            size_cat = "dresses"
        elif cat_raw in ("pants", "jeans", "kaki", "short", "skirt"):
            size_cat = "bottoms"
        else:
            size_cat = "tops"

        args: dict[str, Any] = {
            "category": size_cat,
            "gender": str(slots.value_of("target_gender") or "UNISEX"),
            "fit_preference": str(slots.value_of("fit") or "REGULAR"),
        }
        meas = slots.value_of("measurements")
        if meas is not None:
            if getattr(meas, "height_cm", None) is not None:
                args["height_cm"] = meas.height_cm
            if getattr(meas, "weight_kg", None) is not None:
                args["weight_kg"] = meas.weight_kg
            if getattr(meas, "waist_cm", None) is not None:
                args["waist_cm"] = meas.waist_cm
        if slots.has("product_id"):
            args["product_id"] = slots.value_of("product_id")
        return args

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
        occ_val = slots.value_of("occasion")
        if cat_val in exact_cat_map:
            args["category"] = exact_cat_map[cat_val]
        elif cat_val == "pants":
            args["category"] = (
                "Quần âu, Quần kaki" if occ_val in ("office", "party") else "Quần"
            )
        elif cat_val == "tops":
            args["category"] = (
                "Áo sơ mi, Áo polo, Áo khoác"
                if occ_val in ("office", "party")
                else "Áo"
            )
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
    def _has_ungrounded_product_listing(
        text: str,
        prior_messages: list[BaseMessage] | None = None,
    ) -> bool:
        """Detect fabricated product IDs or catalog prices unseen in prior turns."""
        if not text:
            return False

        prior_ids: set[str] = set()
        prior_prices: set[str] = set()
        for msg in prior_messages or []:
            content = str(msg.content or "")
            if not content:
                continue
            if isinstance(msg, (AIMessage, ToolMessage)):
                prior_ids.update(_extract_product_ids_from_text(content))
            prior_prices.update(_extract_normalized_prices(content))

        for pid in _extract_product_ids_from_text(text):
            if pid not in prior_ids:
                return True

        for match in _PRODUCT_PRICE_PATTERN.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if digits in prior_prices:
                continue

            prefix = text[max(0, match.start() - 40) : match.start()].lower()
            suffix = text[match.end() : match.end() + 20].lower()

            if re.search(
                r"(?:giá(?:\s+từ)?\s*:|\*\*giá|—\s*giá|\|\s*\*\*?giá)",
                prefix,
            ):
                return True

            is_advisory_range = bool(
                re.search(
                    r"\b(?:khoảng|tầm|dao\s+động|ngân\s+sách|dưới|trên|từ)\b",
                    prefix,
                )
                or re.search(
                    r"(?:đ|vnđ|vnd)\s*(?:-|–|~|đến|tới)\s*$",
                    prefix,
                )
                or re.search(r"^\s*(?:-|–|~|đến|tới)\s*\d", suffix)
            )
            if not is_advisory_range:
                return True

        return False

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
        *,
        use_casual_default: bool = False,
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
            "style",
            "gender",
            "user_id",
            "favorite_colors",
            "disliked_colors",
            "min_price",
            "max_price",
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
        if not args.get("disliked_colors") and slots.has("disliked_colors"):
            args["disliked_colors"] = slots.value_of("disliked_colors")
        if args.get("product_id") is None and slots.has("product_id"):
            args["product_id"] = slots.value_of("product_id")

        if not args.get("occasion"):
            if slots.has("occasion"):
                occ_map = {
                    "office": "đi làm",
                    "meeting": "đi họp",
                    "party": "dự tiệc",
                    "date": "hẹn hò",
                    "casual": "đi chơi",
                    "sport": "thể thao",
                }
                occ_val = str(slots.value_of("occasion"))
                args["occasion"] = occ_map.get(occ_val, occ_val)
            elif use_casual_default and not (
                args.get("top_category")
                or args.get("bottom_category")
                or args.get("product_id")
            ):
                args["occasion"] = "đi chơi"

        if not args.get("style") and slots.has("style"):
            args["style"] = slots.value_of("style")
        if slots.has("budget"):
            b_val = slots.value_of("budget")
            if (
                args.get("min_price") is None
                and getattr(b_val, "min_price", None) is not None
            ):
                args["min_price"] = b_val.min_price
            if (
                args.get("max_price") is None
                and getattr(b_val, "max_price", None) is not None
            ):
                args["max_price"] = b_val.max_price

        if not args.get("gender") and slots.has("target_gender"):
            args["gender"] = slots.value_of("target_gender")
        if args.get("user_id") is None and user_id is not None:
            args["user_id"] = user_id

        return args

    @staticmethod
    def _format_immediate_outfit_fallback_reply(
        outfits: list[OutfitSuggestion], slots: ConversationSlots
    ) -> str:
        """Format complete top + bottom combos from Outfit KG for fallback paths."""
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
        *,
        use_casual_default: bool = False,
        config: RunnableConfig | None = None,
    ) -> tuple[list[OutfitSuggestion], list[ChatRecommendedProduct], str]:
        """Execute suggest_outfit fallback and resolve combo products & reply."""
        outfit_tool = self.tools_by_name.get("suggest_outfit")
        if outfit_tool is None:
            return [], [], ""
        fallback_args = self._build_outfit_tool_args(
            {}, slots, user_id, use_casual_default=use_casual_default
        )
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
                        products.append(
                            self._enrich_recommended_from_outfit_item(cached, item)
                        )
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
        products: list[ChatRecommendedProduct],
        slots: ConversationSlots,
        *,
        intent: TurnIntent = TurnIntent.PRODUCT_SEARCH,
    ) -> str:
        """Format a direct fact-first recommendation list with clear assumption."""
        top_products = products[:MAX_RECOMMENDED_PRODUCTS]
        if intent == TurnIntent.BODY_FIT_ADVICE and not slots.has("body_concern"):
            intro = (
                "Vì chưa có thông tin đặc điểm vóc dáng cụ thể nên mình gợi ý "
                "các mẫu form Regular dễ mặc, tôn dáng trung tính:"
            )
            lines = [intro]
            for p in top_products:
                lines.append(ShopAssistantAgent._format_product_bullet(p))
            return "\n".join(lines)

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
        prior_messages: list[BaseMessage] | None = None,
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
                extractor_prompt, extractor_debug = (
                    build_extractor_system_prompt_with_debug()
                )
                sys_extractor = SystemMessage(content=extractor_prompt)
                eval_msgs: list[BaseMessage] = (
                    [sys_extractor, *recent_msgs] if recent_msgs else turn_messages
                )

                extractor_config = (
                    _with_prompt_debug_metadata(config, extractor_debug)
                    if self.settings.DEBUG_CHAT_META
                    else config
                )
                result = await self.structured_model.ainvoke(
                    eval_msgs, config=extractor_config
                )
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
                        and not self._has_ungrounded_product_listing(
                            last_reply_text, prior_messages=prior_messages
                        )
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
            and not self._has_ungrounded_product_listing(
                text, prior_messages=prior_messages
            )
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
            candidate_ids = [
                str(pid) for pid in fallback_product_ids[:MAX_RECOMMENDED_PRODUCTS]
            ]

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

        if not resolved and fallback_product_ids:
            for pid in fallback_product_ids[:MAX_RECOMMENDED_PRODUCTS]:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                prod = get_cached_product(pid)
                if prod:
                    resolved.append(
                        prod
                        if isinstance(prod, ChatRecommendedProduct)
                        else product_schema_to_recommended(prod)
                    )

        return resolved[:MAX_RECOMMENDED_PRODUCTS]

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
                decision,
                policy,
                slots,
                intent,
                topic_switched,
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
                "clarification_turns": session.clarification_turns,
                "declined_slots": sorted(session.declined_slots),
                "last_asked_slots": list(session.last_asked_slots),
                "topic_switched": topic_switched,
                "turn_intent": intent.value,
                "is_shopping_query": is_shopping_query,
                "policy_decision": decision.model_dump(),
                "decision_source": decision.decision_source,
                "decision_reason": decision.decision_reason,
                "asked_slots": [],
                "tools_used": [],
                "reply": "",
                "needs_clarification": False,
                "clarification_form": None,
                "product_ids": [],
                "products": [],
                "outfits": [],
                "prefetched_tool": None,
                "prefetched_context": "",
            }

            debug_handler = LangChainDebugCallbackHandler()
            config: RunnableConfig = {
                "configurable": {"thread_id": session_id},
                "callbacks": [debug_handler],
            }
            rotation_seed = (
                session_id if self.settings.OUTFIT_SESSION_ROTATION else None
            )
            with (
                product_turn_context(),
                outfit_diversity_context(
                    exclude_product_ids=list(session.shown_outfit_product_ids),
                    exclude_pairs=list(session.shown_outfit_pairs),
                    rotation_seed=rotation_seed,
                ),
            ):
                result_state = await self.graph.ainvoke(initial_state, config=config)

            latency_ms = int((time.time() - start_time) * 1000)
            final_asked = bool(result_state.get("needs_clarification", False))
            final_form: ClarificationForm | None = result_state.get(
                "clarification_form"
            )
            question_count = len(final_form.questions) if final_form else 0
            asked_slots = list(result_state.get("asked_slots") or [])
            final_source = result_state.get("decision_source", decision.decision_source)
            final_reason = result_state.get("decision_reason", decision.decision_reason)
            tools_used = list(result_state.get("tools_used", []))

            logger.info(
                "CHAT_LANGGRAPH_METRICS session_id=%s mode=%s phase=%s "
                "next_phase=%s decision_source=%s tools=%s needs_clarification=%s "
                "products_count=%d latency_ms=%d",
                session_id[:8],
                mode.value if hasattr(mode, "value") else mode,
                executed_phase,
                session.clarify_phase.value,
                final_source,
                tools_used,
                final_asked,
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
                clarification_turns=session.clarification_turns,
                topic_switched=topic_switched,
                decision_source=final_source,
                decision_reason=final_reason,
                reason_codes=decision.reason_codes,
                impact_signals=decision.impact_signals,
                evidence_rule_ids=decision.evidence_rule_ids,
                asked_slots=asked_slots,
                gate_search=policy.gate_search,
                active_profile=session.active_profile,
                slots=slots,
                intent=intent,
                context_messages_count=len(session_mgr.get_messages(session_id)),
                latency_ms=latency_ms,
                usage_meta=usage_meta,
                runs=runs,
                user_id=user_id,
            )

            turn_id = (len(session.messages) // 2) + 1
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            await log_chat_turn(
                session_id=session_id,
                turn_id=turn_id,
                user_message=request.message,
                assistant_message=result_state.get("reply", ""),
                latency_ms=latency_ms,
                token_usage=usage_meta,
                error=None,
                system_prompt_version=SYSTEM_PROMPT_VERSION,
                clarification_mode=mode_val,
                turn_intent=intent.value,
                clarification_asked=final_asked,
                question_count=question_count,
                asked_slots=asked_slots,
                slot_states=slots.to_slot_states_dict(intent),
                decision_source=final_source,
                decision_reason=final_reason,
                tools_used=tools_used,
                reason_codes=decision.reason_codes or None,
                impact_signals=decision.impact_signals or None,
                evidence_rule_ids=decision.evidence_rule_ids or None,
            )

            return ChatResponse(
                reply=result_state.get("reply", ""),
                session_id=session_id,
                intent=intent.value,
                clarification_asked=final_asked,
                clarification_form=final_form,
                tools_used=tools_used,
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
                system_prompt_version=SYSTEM_PROMPT_VERSION,
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
            decision,
            policy,
            slots,
            intent,
            topic_switched,
        ) = await self._resolve_turn_session(
            request=request, user_id=user_id, auth_token=auth_token
        )

        history_msgs = session_mgr.get_messages(
            session_id, max_turns=self.settings.MAX_SESSION_HISTORY_TURNS
        )
        is_shopping_query = intent != TurnIntent.NON_SHOPPING
        executed_phase = session.clarify_phase.value
        buffer_tokens_for_size_chart = (
            intent == TurnIntent.SIZE_ADVICE
            and not slots.has_complete_measurements(slots.value_of("category"))
            and (
                mode == ClarificationMode.IMMEDIATE_ANSWER
                or (decision.action == "PROCEED" and decision.use_fallback)
            )
        )

        initial_state: AgentState = {
            "messages": history_msgs,
            "session_id": session_id,
            "user_id": user_id,
            "user_profile": session.active_profile,
            "clarification_mode": mode,
            "clarify_phase": session.clarify_phase,
            "gated_turns": session.gated_turns,
            "clarification_turns": session.clarification_turns,
            "declined_slots": sorted(session.declined_slots),
            "last_asked_slots": list(session.last_asked_slots),
            "topic_switched": topic_switched,
            "turn_intent": intent.value,
            "is_shopping_query": is_shopping_query,
            "policy_decision": decision.model_dump(),
            "decision_source": decision.decision_source,
            "decision_reason": decision.decision_reason,
            "asked_slots": [],
            "tools_used": [],
            "reply": "",
            "needs_clarification": False,
            "clarification_form": None,
            "product_ids": [],
            "products": [],
            "outfits": [],
            "prefetched_tool": None,
            "prefetched_context": "",
        }

        debug_handler = LangChainDebugCallbackHandler()
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": [debug_handler],
        }
        tools_used: list[str] = []
        full_reply_parts: list[str] = []

        rotation_seed = session_id if self.settings.OUTFIT_SESSION_ROTATION else None
        try:
            with (
                product_turn_context(),
                outfit_diversity_context(
                    exclude_product_ids=list(session.shown_outfit_product_ids),
                    exclude_pairs=list(session.shown_outfit_pairs),
                    rotation_seed=rotation_seed,
                ),
            ):
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
                            if (
                                not policy.gate_search
                                and not buffer_tokens_for_size_chart
                            ):
                                yield {
                                    "type": "token",
                                    "content": text,
                                    "session_id": session_id,
                                }
                    elif chunk_mode == "updates":
                        prefetch_data = payload.get("prefetch")
                        if isinstance(prefetch_data, dict):
                            for t in prefetch_data.get("tools_used", []):
                                if t not in tools_used:
                                    tools_used.append(t)
                                    yield {
                                        "type": "tool_start",
                                        "tool": t,
                                        "session_id": session_id,
                                    }
                        tools_data = payload.get("tools")
                        if isinstance(tools_data, dict):
                            new_tools = tools_data.get("tools_used", [])
                            for t in new_tools:
                                if t not in tools_used:
                                    tools_used.append(t)
                                    yield {
                                        "type": "tool_start",
                                        "tool": t,
                                        "session_id": session_id,
                                    }
                        fin_data = payload.get("finalize")
                        if isinstance(fin_data, dict):
                            for t in fin_data.get("tools_used", []):
                                if t not in tools_used:
                                    tools_used.append(t)
                            final_reply = (
                                fin_data.get("reply")
                                or "".join(full_reply_parts).strip()
                            ).strip()
                            final_asked = bool(
                                fin_data.get("needs_clarification", False)
                            )
                            final_form = fin_data.get("clarification_form")
                            question_count = (
                                len(final_form.questions) if final_form else 0
                            )
                            asked_slots = list(fin_data.get("asked_slots") or [])
                            final_source = fin_data.get(
                                "decision_source", decision.decision_source
                            )
                            final_reason = fin_data.get(
                                "decision_reason", decision.decision_reason
                            )

                            latency_ms = int((time.time() - start_time) * 1000)
                            runs = debug_handler.get_runs()
                            usage_meta = debug_handler.get_aggregated_usage()
                            debug_meta_obj = self._build_debug_meta(
                                mode=mode,
                                executed_phase=executed_phase,
                                next_phase_value=session.clarify_phase.value,
                                gated_turns=session.gated_turns,
                                clarification_turns=session.clarification_turns,
                                topic_switched=topic_switched,
                                decision_source=final_source,
                                decision_reason=final_reason,
                                reason_codes=decision.reason_codes,
                                impact_signals=decision.impact_signals,
                                evidence_rule_ids=decision.evidence_rule_ids,
                                asked_slots=asked_slots,
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
                                user_id=user_id,
                            )
                            debug_meta = (
                                debug_meta_obj.model_dump() if debug_meta_obj else None
                            )

                            turn_id = (len(session.messages) // 2) + 1
                            mode_val = (
                                mode.value if hasattr(mode, "value") else str(mode)
                            )
                            await log_chat_turn(
                                session_id=session_id,
                                turn_id=turn_id,
                                user_message=request.message,
                                assistant_message=final_reply,
                                latency_ms=latency_ms,
                                token_usage=usage_meta,
                                error=None,
                                system_prompt_version=SYSTEM_PROMPT_VERSION,
                                clarification_mode=mode_val,
                                turn_intent=intent.value,
                                clarification_asked=final_asked,
                                question_count=question_count,
                                asked_slots=asked_slots,
                                slot_states=slots.to_slot_states_dict(intent),
                                decision_source=final_source,
                                decision_reason=final_reason,
                                tools_used=tools_used,
                                reason_codes=decision.reason_codes or None,
                                impact_signals=decision.impact_signals or None,
                                evidence_rule_ids=decision.evidence_rule_ids or None,
                            )

                            yield {
                                "type": "done",
                                "reply": final_reply,
                                "session_id": session_id,
                                "intent": intent.value,
                                "clarification_asked": final_asked,
                                "clarification_form": (
                                    final_form.model_dump() if final_form else None
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
            mode_val = mode.value if hasattr(mode, "value") else str(mode)
            await log_chat_turn(
                session_id=session_id,
                turn_id=turn_id,
                user_message=request.message,
                assistant_message=None,
                latency_ms=latency_ms,
                token_usage=None,
                error=str(e),
                system_prompt_version=SYSTEM_PROMPT_VERSION,
                clarification_mode=mode_val,
                turn_intent=intent.value,
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
