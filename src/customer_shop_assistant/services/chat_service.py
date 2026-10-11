from typing import AsyncIterator, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from customer_shop_assistant.core.config import settings

class ChatService:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.system_prompt = system_prompt or (
            "You are a friendly, knowledgeable, and helpful customer shop assistant. "
            "Your role is to assist customers with product recommendations, store information, "
            "pricing, availability, and general support. "
            "Always be polite, clear, and concise."
        )

    def _get_llm(self) -> ChatOllama:
        return ChatOllama(
            model=self.model,
            base_url=self.base_url,
        )

    def _build_messages(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        messages: List[BaseMessage] = [
            SystemMessage(content=custom_system_prompt or self.system_prompt)
        ]
        if history:
            for item in history:
                role = item.get("role")
                content = item.get("content", "")
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
        history: Optional[List[dict]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> str:
        llm = self._get_llm()
        messages = self._build_messages(message, history, custom_system_prompt)
        response = await llm.ainvoke(messages)
        return str(response.content)

    async def stream_response(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        custom_system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        llm = self._get_llm()
        messages = self._build_messages(message, history, custom_system_prompt)
        async for chunk in llm.astream(messages):
            yield str(chunk.content)


chat_service = ChatService()
