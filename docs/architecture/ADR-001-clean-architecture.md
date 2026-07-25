# ADR-001: Clean Architecture Adherence

## Status
Accepted

## Context
HalluciSense is a complex AI-driven SaaS application requiring robust separation of concerns. We need to ensure that the core domain logic (the Hallucination Engine) is completely independent of frameworks, databases, and UI layers.

## Decision
We will strictly adhere to Robert C. Martin's Clean Architecture principles.
- **Domain Layer (`app/core/engine`)**: Contains pure Python logic for the tri-pillar hallucination detection. It must NEVER import from FastAPI, SQLAlchemy, or Celery.
- **Use Case Layer (`app/modules/*/service.py`)**: Orchestrates the application logic and acts as the boundary between the web layer and the domain layer.
- **Interface Adapters (`app/repositories`, `app/modules/*/router.py`)**: Converts data between the external formats (HTTP/DB) and the internal use case formats.
- **Frameworks (`app/main.py`, `app/database`)**: The outermost layer containing FastAPI setup, database connections, and Celery worker configurations.

## Consequences
- **Positive**: The core verification engine is completely unit-testable without mocking databases or HTTP requests. We can theoretically swap FastAPI for Django or SQLAlchemy for Tortoise ORM without touching the engine.
- **Negative**: Adds initial boilerplate (e.g., mapping ORM models to Pydantic schemas or Domain entities), which slightly slows down initial feature velocity but pays off long-term.
