"""
OpenAI Provider implementation via HTTPX.
"""
import json
import httpx
from typing import AsyncGenerator, List
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.schemas import ProviderResponseChunk, TokenLogit
from app.core.config import settings

class OpenAIProvider(AbstractLLMProvider):
    """
    Integration with OpenAI models using direct REST API to avoid additional heavy dependencies.
    """
    def __init__(self, model_slug: str):
        self._model_slug = model_slug
        self.api_key = settings.OPENAI_API_KEY
        
    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_slug(self) -> str:
        return self._model_slug

    async def stream_chat(self, messages: List[dict], system_prompt: str | None = None) -> AsyncGenerator[ProviderResponseChunk, None]:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", "https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30.0) as response:
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
                                    logits.append(TokenLogit(token=lp["token"], logprob=lp["logprob"]))
                                    
                            if content or logits:
                                yield ProviderResponseChunk(text=content or "", logits=logits if logits else None, is_done=False)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
                            
        yield ProviderResponseChunk(text="", is_done=True)

    async def generate_response(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a complete non-streaming response text from OpenAI using specified temperature.
        """
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self._model_slug,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": False,
        }

        timeout = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or ""
            return ""
