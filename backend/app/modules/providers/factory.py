"""
HalluciSense LLM Provider Factory.

Maps frontend model identifiers to valid provider-specific
API model names.
"""

from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.gemini import GeminiProvider
from app.modules.providers.openai import OpenAIProvider
from app.modules.providers.ollama import OllamaProvider


OPENAI_MODEL_MAP = {
    "openai": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "openai-gpt-4o": "gpt-4o",
    "openai (gpt-4o)": "gpt-4o",

    "gpt-4o-mini": "gpt-4o-mini",
    "openai-gpt-4o-mini": "gpt-4o-mini",
    "openai (gpt-4o-mini)": "gpt-4o-mini",
}


GEMINI_MODEL_MAP = {
    "gemini": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "google-gemini": "gemini-2.0-flash",
    "google (gemini)": "gemini-2.0-flash",
}


def get_provider(model_slug: str) -> AbstractLLMProvider:
    """
    Resolve a HalluciSense/frontend model identifier into
    the corresponding LLM provider.

    Raises:
        ValueError: If the model identifier is unsupported.
    """

    if not model_slug:
        raise ValueError("model_slug cannot be empty")

    normalized = model_slug.strip().lower()

    # ---------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------

    if normalized in OPENAI_MODEL_MAP:
        return OpenAIProvider(
            model_slug=OPENAI_MODEL_MAP[normalized]
        )

    if "openai" in normalized or "gpt" in normalized:
        return OpenAIProvider(
            model_slug="gpt-4o"
        )

    # ---------------------------------------------------------
    # Google Gemini
    # ---------------------------------------------------------

    if normalized in GEMINI_MODEL_MAP:
        return GeminiProvider(
            model_slug=GEMINI_MODEL_MAP[normalized]
        )

    if "gemini" in normalized:
        return GeminiProvider(
            model_slug="gemini-2.0-flash"
        )

    # ---------------------------------------------------------
    # Local Ollama models
    # ---------------------------------------------------------

    if (
        "ollama" in normalized
        or "llama" in normalized
        or "mistral" in normalized
        or "qwen" in normalized
        or "deepseek" in normalized
    ):
        return OllamaProvider(
            model_slug=model_slug
        )

    # ---------------------------------------------------------
    # Unknown model
    # ---------------------------------------------------------

    raise ValueError(
        f"Unsupported HalluciSense model: {model_slug}"
    )