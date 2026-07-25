"""
Provider Factory.
Instantiates the correct LLM provider based on the model slug.
"""
from app.modules.providers.base import AbstractLLMProvider
from app.modules.providers.gemini import GeminiProvider
from app.modules.providers.openai import OpenAIProvider
from app.modules.providers.ollama import OllamaProvider

def get_provider(model_slug: str) -> AbstractLLMProvider:
    """
    Factory function to get the appropriate LLM provider for a given model.
    """
    if "gemini" in model_slug.lower():
        return GeminiProvider(model_slug=model_slug)
    if "openai" in model_slug.lower() or "gpt" in model_slug.lower():
        return OpenAIProvider(model_slug=model_slug)
    if "ollama" in model_slug.lower() or "llama" in model_slug.lower() or "mistral" in model_slug.lower():
        return OllamaProvider(model_slug=model_slug)
    
    # Default fallback (could throw NotImplementedError for unknown models)
    return GeminiProvider(model_slug="gemini-2.0-flash")
