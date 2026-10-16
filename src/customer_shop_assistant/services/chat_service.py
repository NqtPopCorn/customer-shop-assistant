from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from customer_shop_assistant.config import Settings, get_settings
from customer_shop_assistant.llm.factory import get_chat_model


class ChatService:
    """Service handling conversational turn generation and message formatting."""

    def __init__(
        self,
        settings: Settings | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.system_prompt = system_prompt or (
            "Bạn là trợ lý tư vấn khách hàng của shop thời trang. "
            "Trả lời ngắn gọn, trực diện, đúng sự thật về sản phẩm và chính sách."
        )

    def _get_llm(self) -> BaseChatModel:
        return get_chat_model(self.settings)

    def _build_messages(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        custom_system_prompt: str | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = [
            SystemMessage(content=custom_system_prompt or self.system_prompt)
        ]
        if history:
            for item in history:
                role = item.get("role")
                content = str(item.get("content", ""))
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

        messages.append(HumanMessage(content=message))
        return messages

    async def generate_response(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        custom_system_prompt: str | None = None,
    ) -> str:
        llm = self._get_llm()
        messages = self._build_messages(message, history, custom_system_prompt)
        response = await llm.ainvoke(messages)
        return str(response.content)

    async def stream_response(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        custom_system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        llm = self._get_llm()
        messages = self._build_messages(message, history, custom_system_prompt)
        async for chunk in llm.astream(messages):
            yield str(chunk.content)
