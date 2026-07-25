# ADR-004: Event-Driven Asynchronous Verification Pipeline

## Status
Accepted

## Context
The HalluciSense Tri-Pillar engine is computationally expensive. It requires querying external APIs (Wikipedia), running dense vector searches (FAISS), processing sentence embeddings, and executing cross-encoder models. If we run this synchronously, the user would wait 5-15 seconds before seeing the AI's response, resulting in a terrible User Experience (UX).

## Decision
We will adopt an **Event-Driven Asynchronous** architecture for verification.
1. **Immediate Streaming**: When the user sends a prompt, the FastAPI server streams the raw LLM tokens back to the user via WebSockets *immediately*.
2. **Background Dispatch**: Once the stream completes, the completed message text and token logits are saved to the database, and a Celery background task (`verify_response_task`) is dispatched.
3. **Asynchronous Analysis**: Celery workers pick up the task and run the heavy Tri-Pillar analysis in the background.
4. **Real-time Notification**: Upon completion, the Celery worker saves the report to the database and publishes a `verification_complete` event to a Redis Pub/Sub channel specific to that user.
5. **UI Update**: The frontend, which is listening to this channel via a global WebSocket, receives the event and triggers a re-render of the message, applying the color-coded annotations and H-Score badges.

## Consequences
- **Positive**: Incredible UX. The user sees the AI typing immediately (similar to ChatGPT), and then magically sees the text get highlighted a few seconds later (similar to Perplexity AI).
- **Negative**: Substantial infrastructure complexity. We now must manage a Redis broker, Celery worker fleet, and WebSocket state, making local development and deployment harder.
