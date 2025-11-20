"""
memory/short_term.py - Session Memory Management for AIOS Brain

Manages short-term context and conversation memory using Redis.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    session_id: str
    content: str
    timestamp: float
    metadata: Dict

class ShortTermMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_seconds: int = 3600):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.client: Optional[aioredis.Redis] = None
    
    async def connect(self):
        self.client = await aioredis.from_url(self.redis_url, decode_responses=True)
        logger.info("Connected to Redis for short-term memory")
    
    async def store(self, session_id: str, content: str, metadata: Dict = None):
        if not self.client:
            await self.connect()
        
        item = MemoryItem(
            session_id=session_id,
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata or {}
        )
        
        key = f"session:{session_id}:memory"
        await self.client.lpush(key, json.dumps(asdict(item)))
        await self.client.expire(key, self.ttl_seconds)
        logger.debug(f"Stored memory for session {session_id}")
    
    async def retrieve(self, session_id: str, limit: int = 10) -> List[MemoryItem]:
        if not self.client:
            await self.connect()
        
        key = f"session:{session_id}:memory"
        items = await self.client.lrange(key, 0, limit - 1)
        
        memories = []
        for item_json in items:
            data = json.loads(item_json)
            memories.append(MemoryItem(**data))
        
        logger.debug(f"Retrieved {len(memories)} memories for session {session_id}")
        return memories
    
    async def clear_session(self, session_id: str):
        if not self.client:
            await self.connect()
        
        key = f"session:{session_id}:memory"
        await self.client.delete(key)
        logger.info(f"Cleared memory for session {session_id}")
    
    async def get_context_window(self, session_id: str, max_tokens: int = 4000) -> str:
        memories = await self.retrieve(session_id, limit=20)
        
        context_parts = []
        total_tokens = 0
        
        for memory in memories:
            estimated_tokens = len(memory.content.split()) * 1.3
            if total_tokens + estimated_tokens > max_tokens:
                break
            context_parts.append(memory.content)
            total_tokens += estimated_tokens
        
        return "\n".join(reversed(context_parts))
    
    async def close(self):
        if self.client:
            await self.client.close()
            logger.info("Closed Redis connection")
