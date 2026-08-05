"""
HalluciSense LLM Provider Factory.

Maps frontend model identifiers to valid provider-specific
API model names.
"""

import structlog
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.gemini import GeminiProvider
from app.modules.providers.openai import OpenAIProvider
from app.modules.providers.ollama import OllamaProvider

logger = structlog.get_logger(__name__)

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
    "gemini": "gemini-2.5-flash",
    "gemini-3.1-pro": "gemini-2.5-pro",
    "gemini-3.1-flash": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
}


def get_provider(model_slug: str) -> AbstractLLMProvider:
    """
    Resolve a HalluciSense/frontend model identifier into
    the corresponding LLM provider.

    Raises:
        ValueError: If the model identifier is unsupported.
    """

    if not model_slug:
        logger.warning("empty_model_slug_provided_defaulting_to_gemini")
        return GeminiProvider(model_slug="gemini-1.5-flash")

    normalized = model_slug.strip().lower()
    logger.info("provider_factory_resolving", model_slug=model_slug, normalized=normalized)

    # ---------------------------------------------------------
    # OpenAI
    # ---------------------------------------------------------

    if normalized in OPENAI_MODEL_MAP:
        provider = OpenAIProvider(model_slug=OPENAI_MODEL_MAP[normalized])
        logger.info("provider_factory_resolved", provider=provider.provider_name, model=provider.model_slug)
        return provider

    if "openai" in normalized or "gpt" in normalized:
        provider = OpenAIProvider(model_slug="gpt-4o")
        logger.info("provider_factory_resolved", provider=provider.provider_name, model=provider.model_slug)
        return provider

    # ---------------------------------------------------------
    # Google Gemini
    # ---------------------------------------------------------

    if normalized in GEMINI_MODEL_MAP:
        provider = GeminiProvider(model_slug=GEMINI_MODEL_MAP[normalized])
        logger.info("provider_factory_resolved", provider=provider.provider_name, model=provider.model_slug)
        return provider

    if "gemini" in normalized:
        provider = GeminiProvider(model_slug="gemini-1.5-flash")
        logger.info("provider_factory_resolved", provider=provider.provider_name, model=provider.model_slug)
        return provider

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
        provider = OllamaProvider(model_slug=model_slug)
        logger.info("provider_factory_resolved", provider=provider.provider_name, model=provider.model_slug)
        return provider

    # Fallback to Gemini if unknown
    logger.warning("unknown_model_slug_falling_back_to_gemini", model_slug=model_slug)
    return GeminiProvider(model_slug="gemini-1.5-flash")