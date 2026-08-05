"""
Redis Pub/Sub utilities for broadcasting messages from Celery workers to FastAPI WebSockets.
"""
import asyncio
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
        _redis_pool = redis_async.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
    return Redis(connection_pool=_redis_pool)

async def publish_message(channel: str, message: dict):
    """
    Publish a JSON message to a Redis channel with a 3.0s timeout limit.
    """
    try:
        redis = await get_redis()
        await asyncio.wait_for(redis.publish(channel, json.dumps(message)), timeout=3.0)
        logger.debug("pubsub_published", channel=channel, message=message)
    except Exception as e:
        logger.warning("pubsub_publish_failed", channel=channel, error=str(e))

async def subscribe(channel: str) -> AsyncGenerator[dict, None]:
    """
    Subscribe to a Redis channel and yield JSON-decoded messages.
    Handles Redis connection errors and server shutdown gracefully.
    """
    try:
        redis = await get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        logger.info("pubsub_subscribed", channel=channel)
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    yield data
                except json.JSONDecodeError:
                    logger.warning("pubsub_decode_error", channel=channel, data=message['data'])
    except asyncio.CancelledError:
        logger.info("pubsub_subscribe_cancelled", channel=channel)
        raise
    except Exception as e:
        logger.warning("pubsub_subscribe_error", channel=channel, error=str(e))
    finally:
        try:
            if 'pubsub' in locals():
                await pubsub.unsubscribe(channel)
                await pubsub.close()
        except Exception:
            pass
