"""Database module for SABER episode event storage.

Redis is the storage backend for episode events and transcripts.

This module provides:
- RedisManager: Connection management for redis.asyncio
- EpisodeEventRepository: CRUD operations for episode events
"""

from .event_repository import EpisodeEventRepository
from .redis_manager import RedisManager

__all__ = ["RedisManager", "EpisodeEventRepository"]
