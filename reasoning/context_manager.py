"""
reasoning/context_manager.py - Context Window Management for AIOS Brain

Manages context windows with sliding window algorithm and automatic summarization.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class ContextItem:
    content: str
    importance: float
    tokens: int
    timestamp: float

class ContextManager:
    def __init__(self, max_tokens: int = 8000, importance_threshold: float = 0.5):
        self.max_tokens = max_tokens
        self.importance_threshold = importance_threshold
        self.context_items: List[ContextItem] = []
    
    def add_context(self, content: str, importance: float = 0.7):
        tokens = self._estimate_tokens(content)
        item = ContextItem(
            content=content,
            importance=importance,
            tokens=tokens,
            timestamp=asyncio.get_event_loop().time()
        )
        self.context_items.append(item)
        self._trim_context()
        logger.debug(f"Added context item ({tokens} tokens, importance={importance})")
    
    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
    
    def _trim_context(self):
        total_tokens = sum(item.tokens for item in self.context_items)
        
        while total_tokens > self.max_tokens and len(self.context_items) > 1:
            sorted_items = sorted(self.context_items, key=lambda x: (x.importance, -x.timestamp))
            removed = sorted_items[0]
            self.context_items.remove(removed)
            total_tokens -= removed.tokens
            logger.debug(f"Removed low-importance context item ({removed.tokens} tokens)")
    
    def get_context_window(self) -> str:
        return "\n\n".join(item.content for item in self.context_items)
    
    def get_token_count(self) -> int:
        return sum(item.tokens for item in self.context_items)
    
    def clear(self):
        self.context_items.clear()
        logger.info("Cleared all context items")
    
    def score_importance(self, content: str) -> float:
        keywords = ["important", "critical", "key", "essential", "must"]
        score = 0.5
        lower_content = content.lower()
        for keyword in keywords:
            if keyword in lower_content:
                score += 0.1
        return min(score, 1.0)
    
    async def summarize_context(self) -> Optional[str]:
        if len(self.context_items) < 3:
            return None
        
        content = self.get_context_window()
        summary = f"Context summary ({len(self.context_items)} items, {self.get_token_count()} tokens): "
        summary += content[:200] + "..."
        return summary
