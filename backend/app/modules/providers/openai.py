"""
OpenAI Provider implementation via HTTPX.
"""
import json
import traceback
import httpx
from typing import AsyncGenerator, List, Optional
import structlog

from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk, TokenLogit
from app.core.config import settings

logger = structlog.get_logger(__name__)


class OpenAIProvider(AbstractLLMProvider):
    """
    Integration with OpenAI models using direct REST API.
    """

    def __init__(self, model_slug: str = "gpt-4o"):
        self._model_slug = model_slug
        self.api_key = settings.OPENAI_API_KEY

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    async def stream_chat(
        self, messages: List[dict], system_prompt: Optional[str] = None
    ) -> AsyncGenerator[ProviderResponseChunk, None]:
        if not self.api_key or not self.api_key.strip():
            logger.error("openai_api_key_not_configured")
            raise ValueError("OpenAI API key is not configured in settings.")

        formatted_messages = []
        if system_prompt and system_prompt.strip():
            formatted_messages.append({"role": "system", "content": system_prompt.strip()})

        for msg in messages:
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            role = msg.get("role", "user")
            formatted_messages.append({"role": role, "content": content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 5,
        }

        logger.info(
            "openai_stream_chat_started",
            model=self._model_slug,
            messages_count=len(formatted_messages),
        )

        tokens_yielded = 0
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")

                                logits = []
                                if choice.get("logprobs") and choice["logprobs"].get("content"):
                                    for lp in choice["logprobs"]["content"]:
                                        logits.append(
                                            TokenLogit(token=lp["token"], logprob=lp["logprob"])
                                        )

                                if content or logits:
                                    tokens_yielded += 1
                                    yield ProviderResponseChunk(
                                        text=content or "",
                                        logits=logits if logits else None,
                                        is_done=False,
                                    )
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass

            logger.info("openai_stream_chat_completed", model=self._model_slug, tokens=tokens_yielded)
            yield ProviderResponseChunk(text="", logits=None, is_done=True)

        except Exception as e:
            logger.error("openai_stream_chat_failed", model=self._model_slug, error=str(e), traceback=traceback.format_exc())
            raise e

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.api_key or not self.api_key.strip():
            logger.error("openai_api_key_not_configured")
            raise ValueError("OpenAI API key is not configured in settings.")

        formatted_messages = []
        if system_prompt and system_prompt.strip():
            formatted_messages.append({"role": "system", "content": system_prompt.strip()})

        for msg in messages:
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            role = msg.get("role", "user")
            formatted_messages.append({"role": role, "content": content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": False,
        }

        timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or ""
            return ""
