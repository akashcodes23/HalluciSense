"""
Ollama Provider Mock.
"""
from typing import AsyncGenerator, List
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk
from app.modules.providers.gemini import GeminiProvider

class OllamaProvider(AbstractLLMProvider):
    """
    Mock implementation of Ollama Provider.
    For demonstration, this passes through to Gemini or returns a simulated stream.
    """
    def __init__(self, model_slug: str):
        self._model_slug = model_slug
        self._underlying_engine = GeminiProvider(model_slug="gemini-1.5-pro")

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    async def stream_chat(self, messages: List[dict], system_prompt: str | None = None) -> AsyncGenerator[ProviderResponseChunk, None]:
        # Proxy request through available Gemini provider
        async for chunk in self._underlying_engine.stream_chat(messages, system_prompt):
            yield chunk
