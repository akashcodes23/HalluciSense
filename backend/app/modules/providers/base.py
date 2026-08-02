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
        yield ProviderResponseChunk(text="", is_done=True) # type: ignore

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: str | None = None
    ) -> str:
        """
        Generate a complete non-streaming response text.
        """
        ...
