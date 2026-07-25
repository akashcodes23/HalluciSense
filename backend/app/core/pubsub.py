"""
Redis Pub/Sub utilities for broadcasting messages from Celery workers to FastAPI WebSockets.
"""
import json
import structlog
from typing import AsyncGenerator
from redis.asyncio import Redis

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Use a connection pool for publisher
_redis_pool = None

async def get_redis() -> Redis:
    global _redis_pool
    if _redis_pool is None:
        import redis.asyncio as redis_async
        _redis_pool = redis_async.ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)
    return Redis(connection_pool=_redis_pool)

async def publish_message(channel: str, message: dict):
    """
    Publish a JSON message to a Redis channel.
    """
    try:
        redis = await get_redis()
        await redis.publish(channel, json.dumps(message))
        logger.debug("pubsub_published", channel=channel, message=message)
    except Exception as e:
        logger.error("pubsub_publish_failed", channel=channel, error=str(e))

async def subscribe(channel: str) -> AsyncGenerator[dict, None]:
    """
    Subscribe to a Redis channel and yield JSON-decoded messages.
    """
    redis = await get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    logger.info("pubsub_subscribed", channel=channel)
    
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    yield data
                except json.JSONDecodeError:
                    logger.warning("pubsub_decode_error", channel=channel, data=message['data'])
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
