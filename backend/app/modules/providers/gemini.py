"""
Google Gemini provider implementation.
"""
from typing import AsyncGenerator, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk, TokenLogit

class GeminiProvider(AbstractLLMProvider):
    """
    Integration with Google Gemini models.
    """

    def __init__(self, model_slug: str = "gemini-2.0-flash"):
        self._model_slug = model_slug
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Note: system instructions can be passed at model initialization or request time.
        self._client = genai.GenerativeModel(model_name=self._model_slug)

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    async def stream_chat(self, messages: List[dict], system_prompt: str | None = None) -> AsyncGenerator[ProviderResponseChunk, None]:
        """
        Stream the chat from Gemini.
        Messages should be formatted into Google's required roles ('user', 'model').
        """
        # Convert generic standard roles (user, assistant, system) to Gemini roles (user, model).
        formatted_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_messages.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Apply system prompt if available via client initialization (GenerativeModel allows system_instruction)
        # We will override the client if a system prompt is provided.
        client = self._client
        if system_prompt:
            client = genai.GenerativeModel(
                model_name=self._model_slug,
                system_instruction=system_prompt
            )

        # In google-generativeai 0.8.x, generate_content_async supports streaming.
        response = await client.generate_content_async(
            contents=formatted_messages,
            stream=True
        )

        async for chunk in response:
            text = chunk.text if hasattr(chunk, "text") else ""
            
            # Logprobs extraction (if supported by the specific model and API version)
            # Many Gemini versions do not return logprobs natively yet, so we safely handle it.
            logits = None 
            
            yield ProviderResponseChunk(
                text=text,
                logits=logits,
                is_done=False
            )

        # Final yield to indicate completion
        yield ProviderResponseChunk(text="", is_done=True)
