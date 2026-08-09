"""
HalluciSense Ollama Provider.

Provides local LLM inference through the Ollama HTTP API.
No external API key or cloud provider is required.
"""

import json
from typing import AsyncGenerator, List, Optional

import httpx

from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk
from app.core.http_client import get_shared_async_client


class OllamaProvider(AbstractLLMProvider):
    """
    Local Ollama LLM provider.

    Communicates with the Ollama server through its HTTP API
    and exposes responses using the HalluciSense provider contract.
    """

    BASE_URL = "http://127.0.0.1:11434"

    def __init__(self, model_slug: str = "llama3:latest"):
        # Convert generic frontend names into an installed Ollama model.
        normalized = (model_slug or "").strip().lower()

        aliases = {
            "ollama": "llama3:latest",
            "ollama-llama3": "llama3:latest",
            "llama3": "llama3:latest",
            "llama3:latest": "llama3:latest",
        }

        self._model_slug = aliases.get(normalized, model_slug)

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    async def stream_chat(
        self,
        messages: List[dict],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[ProviderResponseChunk, None]:
        """
        Stream a chat completion from the local Ollama server.
        """

        formatted_messages = []

        if system_prompt:
            formatted_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        for message in messages:
            role = message.get("role", "user")

            if role not in {"system", "user", "assistant"}:
                role = "user"

            formatted_messages.append(
                {
                    "role": role,
                    "content": str(message.get("content", "")),
                }
            )

        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "stream": True,
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=30.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/api/chat",
                json=payload,
            ) as response:

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = data.get("message") or {}
                    text = message.get("content", "")

                    if text:
                        yield ProviderResponseChunk(
                            text=text,
                            logits=None,
                            is_done=False,
                        )

                    if data.get("done") is True:
                        break

        yield ProviderResponseChunk(
            text="",
            logits=None,
            is_done=True,
        )

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a complete non-streaming response text from Ollama using specified temperature.
        """
        formatted_messages = []

        if system_prompt:
            formatted_messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        for message in messages:
            role = message.get("role", "user")

            if role not in {"system", "user", "assistant"}:
                role = "user"

            formatted_messages.append(
                {
                    "role": role,
                    "content": str(message.get("content", "")),
                }
            )

        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        client = get_shared_async_client()
        response = await client.post(
            f"{self.BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        message_obj = data.get("message") or {}
        return message_obj.get("content", "") or ""
