# ADR-002: LLM Provider Abstraction & Orchestration

## Status
Accepted

## Context
The AI landscape moves rapidly. Hardcoding our application to the OpenAI or Gemini SDK means we cannot easily adapt when a newer, cheaper, or faster model is released by a competitor. Furthermore, provider outages (like OpenAI API going down) would cause total system failure for HalluciSense users.

## Decision
We will use a **Provider Factory** and an **LLM Orchestrator** pattern.
1. **AbstractLLMProvider (Protocol)**: Defines the required interface for all AI models (e.g., `generate`, `stream`, `get_logits`).
2. **Provider Implementations**: Dedicated classes (e.g., `GeminiProvider`, `OpenAIProvider`) that implement the protocol using provider-specific SDKs.
3. **LLMOrchestrator**: A middleware service that sits between the `MessageService` and the providers. It is responsible for:
   - Initializing the requested primary provider.
   - Using `tenacity` to automatically retry transient connection errors.
   - Handling fallback logic (if Gemini fails completely, fallback to a secondary model like OpenAI GPT-4o-mini).

## Consequences
- **Positive**: Zero vendor lock-in. We can add local Llama 3 models tomorrow simply by creating an `OllamaProvider`. High availability due to the Orchestrator's fallback logic.
- **Negative**: We are constrained by the lowest common denominator of features across providers (e.g., if one provider doesn't support token logprobs natively, we must build complex workarounds like entropy estimation).
