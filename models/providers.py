"""
models/providers.py - API Provider Clients for AIOS Brain

Production-grade async clients for AI model providers.
"""

import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import httpx
from enum import Enum

logger = logging.getLogger(__name__)

class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass
class ModelResponse:
    content: str
    model: str
    provider: str
    tokens_used: int
    latency_ms: float

class ProviderClient:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
    
    async def generate(self, prompt: str, model: str, max_tokens: int = 2048) -> ModelResponse:
        raise NotImplementedError
    
    async def close(self):
        await self.client.aclose()

class OpenAIProvider(ProviderClient):
    async def generate(self, prompt: str, model: str, max_tokens: int = 2048) -> ModelResponse:
        import time
        start = time.time()
        response = await self.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        )
        data = response.json()
        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            provider="openai",
            tokens_used=data["usage"]["total_tokens"],
            latency_ms=(time.time()-start)*1000
        )

class AnthropicProvider(ProviderClient):
    async def generate(self, prompt: str, model: str, max_tokens: int = 2048) -> ModelResponse:
        import time
        start = time.time()
        response = await self.client.post(
            "/v1/messages",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        )
        data = response.json()
        return ModelResponse(
            content=data["content"][0]["text"],
            model=model,
            provider="anthropic",
            tokens_used=data["usage"]["input_tokens"]+data["usage"]["output_tokens"],
            latency_ms=(time.time()-start)*1000
        )

class ProviderManager:
    def __init__(self):
        self.providers: Dict[str, ProviderClient] = {}
    
    def register_provider(self, name: str, client: ProviderClient):
        self.providers[name] = client
    
    async def generate(self, provider: str, prompt: str, model: str, **kwargs) -> ModelResponse:
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        return await self.providers[provider].generate(prompt, model, **kwargs)
    
    async def close_all(self):
        for provider in self.providers.values():
            await provider.close()
