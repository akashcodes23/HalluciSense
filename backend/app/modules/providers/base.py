"""
Abstract LLM Provider Protocol.
Enforces the interface for all external AI models (Gemini, OpenAI, Anthropic).
"""
from typing import AsyncGenerator, List, Protocol
from app.modules.providers.schemas import ProviderResponseChunk

class MessageDict(Protocol):
    role: str
    content: str

class AbstractLLMProvider(Protocol):
    """
    Protocol defining the contract for any LLM provider integration.
    """
    
    @property
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'Google Gemini')."""
        ...

    @property
    def model_slug(self) -> str:
        """The specific model ID being used (e.g., 'gemini-2.0-flash')."""
        ...

    async def stream_chat(self, messages: List[dict], system_prompt: str | None = None) -> AsyncGenerator[ProviderResponseChunk, None]:
        """
        Stream a chat completion response from the model.
        Must yield ProviderResponseChunk objects containing text and optionally logits.
        """
        ...
        # Workaround for Python AST parser issue with protocols containing yield
        yield ProviderResponseChunk(text="", is_done=True) # type: ignore
