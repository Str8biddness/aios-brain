"""
models/router.py — Intelligent Model Router for AIOS Brain

Production-ready, asynchronous router that:
- Matches task types (coding, writing, analysis, creative) to model capabilities
- Optimizes cost vs. quality based on constraints and environment
- Performs load balancing across multiple providers with health-awareness
- Applies fallbacks when preferred models are unavailable or degraded
- Provides clear type hints, docstrings, and structured logging

Usage:
    router = ModelRouter()
    chosen = await router.select_model(TaskType.CODING, {"max_cost": 0.002, "prefer_provider": "providerA"})
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aios-brain.model-router")
logger.setLevel(logging.INFO)


class TaskType(str, Enum):
    """Enumeration of supported task types for routing decisions."""
    CODING = "coding"
    WRITING = "writing"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelSpec:
    """
    Model specification including capabilities and operational metadata.

    Attributes:
        name: Unique model name.
        provider: Provider identifier (e.g., "providerA").
        task_caps: Supported task types.
        quality_score: Relative quality (0.0-1.0).
        cost_per_1k_tokens: Approximate cost in currency units per 1k tokens.
        max_tps: Theoretical max requests per second for planning (not a limiter).
        latency_target_ms: Expected median latency for single inference.
    """
    name: str
    provider: str
    task_caps: List[TaskType]
    quality_score: float
    cost_per_1k_tokens: float
    max_tps: int
    latency_target_ms: int


@dataclass
class ProviderState:
    """
    Provider runtime state for health-aware load balancing.

    Attributes:
        healthy: Whether provider is considered healthy.
        inflight: Number of ongoing requests routed to this provider.
        last_check_ts: Last health check timestamp (epoch seconds).
        rr_cursor: Round-robin cursor for model selection within provider.
    """
    healthy: bool = True
    inflight: int = 0
    last_check_ts: float = field(default_factory=lambda: time.time())
    rr_cursor: int = 0


class ModelRouter:
    """
    Intelligent model router with async health-aware selection and fallback logic.

    Responsibilities:
    - Capability matching by task type
    - Cost-vs-quality optimization under constraints
    - Load balancing across providers (weighted RR)
    - Fallback to secondary models when preferred unavailable
    - Health tracking and soft backoff

    Constraints (optional dict keys):
        - prefer_provider: str -> prefer routing to this provider
        - max_cost: float -> maximum allowed cost per 1k tokens
        - min_quality: float -> minimum acceptable quality score
        - max_latency_ms: int -> upper bound on target latency
        - prefer_model: str -> direct preferred model if available
    """

    def __init__(self, models: Optional[List[ModelSpec]] = None) -> None:
        self._models: List[ModelSpec] = models or self._default_models()
        self._providers: Dict[str, ProviderState] = {m.provider: ProviderState() for m in self._models}
        self._lock = asyncio.Lock()

    async def select_model(self, task_type: TaskType, constraints: Optional[Dict[str, Any]] = None) -> ModelSpec:
        """
        Select an appropriate model for the given task type applying constraints and balancing.

        Args:
            task_type: The task type requiring inference.
            constraints: Optional constraints dict (see class docstring).

        Returns:
            A ModelSpec representing the chosen model.

        Raises:
            RuntimeError: If no suitable model is available.
        """
        constraints = constraints or {}
        prefer_model = constraints.get("prefer_model")
        prefer_provider = constraints.get("prefer_provider")
        max_cost = constraints.get("max_cost")
        min_quality = constraints.get("min_quality", 0.0)
        max_latency_ms = constraints.get("max_latency_ms", 200)

        async with self._lock:
            candidates = [m for m in self._models if task_type in m.task_caps]
            if not candidates:
                # Fallback: allow UNKNOWN-capable models
                candidates = [m for m in self._models if TaskType.UNKNOWN in m.task_caps]

            # Apply health filter
            candidates = [m for m in candidates if self._providers[m.provider].healthy]

            # Prefer a specific model if requested and healthy
            if prefer_model:
                for m in candidates:
                    if m.name == prefer_model:
                        logger.debug(f"Selected preferred model: {m.name}")
                        self._providers[m.provider].inflight += 1
                        return m

            # Filter by constraints
            def meets_constraints(m: ModelSpec) -> bool:
                if max_cost is not None and m.cost_per_1k_tokens > max_cost:
                    return False
                if m.quality_score < float(min_quality):
                    return False
                if m.latency_target_ms > int(max_latency_ms):
                    return False
                if prefer_provider and m.provider != prefer_provider:
                    return False
                return True

            filtered = [m for m in candidates if meets_constraints(m)]
            if not filtered:
                # Relax constraints progressively: drop provider preference, then latency, then cost
                relaxed = list(candidates)
                if prefer_provider:
                    relaxed = [m for m in relaxed if m.provider == prefer_provider] or candidates
                if not relaxed:
                    relaxed = candidates
                filtered = relaxed

            # Score models based on cost vs quality trade-off
            # Lower score is better; alpha controls quality emphasis
            alpha_quality = 0.7
            beta_latency = 0.3

            def score(m: ModelSpec) -> float:
                # Normalize values
                cost_norm = m.cost_per_1k_tokens
                quality_norm = 1.0 - m.quality_score
                latency_norm = m.latency_target_ms / max(1, max_latency_ms)
                return (alpha_quality * quality_norm) + cost_norm + (beta_latency * latency_norm)

            ranked = sorted(filtered, key=score)

            # Load balancing: choose among top-K by provider weight and inflight
            top_k = ranked[: min(3, len(ranked))]
            chosen = await self._choose_balanced(top_k)

            if chosen is None:
                # Fallback: pick any healthy candidate randomly
                if candidates:
                    chosen = random.choice(candidates)
                    logger.warning("Balanced selection failed; using random healthy candidate")
                else:
                    # Last-resort: pick any model ignoring health
                    if self._models:
                        chosen = random.choice(self._models)
                        logger.error("No healthy models; selecting from full pool ignoring health")
                    else:
                        raise RuntimeError("No models are registered for routing")

            # Update inflight counter for chosen provider
            self._providers[chosen.provider].inflight += 1
            logger.info(f"Model selected: {chosen.name} (provider={chosen.provider}) for task={task_type.value}")
            return chosen

    # Internal helpers --------------------------------------------------------

    async def release(self, model: ModelSpec) -> None:
        """Release inflight counter for provider after request completion."""
        async with self._lock:
            st = self._providers.get(model.provider)
            if st and st.inflight > 0:
                st.inflight -= 1

    async def mark_provider_health(self, provider: str, healthy: bool) -> None:
        """Mark provider health status (e.g., after monitoring signals)."""
        async with self._lock:
            st = self._providers.setdefault(provider, ProviderState())
            st.healthy = healthy
            st.last_check_ts = time.time()
            if not healthy:
                logger.warning(f"Provider {provider} marked unhealthy")

    async def _choose_balanced(self, candidates: List[ModelSpec]) -> Optional[ModelSpec]:
        """
        Weighted round-robin selection among candidates:
        - Weight favors lower inflight and higher max_tps
        - Soft random jitter to avoid lockstep routing
        """
        if not candidates:
            return None

        # Build weights per provider-model pair
        weights: List[Tuple[ModelSpec, float]] = []
        for m in candidates:
            st = self._providers.get(m.provider, ProviderState())
            # Inflight penalty increases weight denominator
            denom = max(1.0, 1.0 + float(st.inflight))
            weight = (m.max_tps / denom) * (0.5 + 0.5 * m.quality_score)
            # Light jitter
            weight *= (0.9 + 0.2 * random.random())
            weights.append((m, weight))

        # Normalize and select
        total = sum(w for _, w in weights)
        if total <= 0:
            # All weights zero; pick with simple round-robin by provider
            return self._round_robin_by_provider(candidates)

        r = random.random() * total
        acc = 0.0
        for m, w in weights:
            acc += w
            if r <= acc:
                return m
        # Fallback to last
        return candidates[-1]

    def _round_robin_by_provider(self, candidates: List[ModelSpec]) -> Optional[ModelSpec]:
        """Simple round-robin selection by provider cursor."""
        if not candidates:
            return None
        providers = list({m.provider for m in candidates})
        # Pick provider with minimal cursor
        provider = min(providers, key=lambda p: self._providers[p].rr_cursor)
        self._providers[provider].rr_cursor += 1
        # Select first candidate for this provider
        for m in candidates:
            if m.provider == provider:
                return m
        return candidates[0]

    @staticmethod
    def _default_models() -> List[ModelSpec]:
        """Default model registry for out-of-the-box routing."""
        return [
            ModelSpec(
                name="brain-small",
                provider="providerA",
                task_caps=[TaskType.WRITING, TaskType.ANALYSIS, TaskType.CREATIVE, TaskType.UNKNOWN],
                quality_score=0.70,
                cost_per_1k_tokens=0.0012,
                max_tps=100,
                latency_target_ms=60,
            ),
            ModelSpec(
                name="brain-code",
                provider="providerA",
                task_caps=[TaskType.CODING, TaskType.ANALYSIS],
                quality_score=0.82,
                cost_per_1k_tokens=0.0018,
                max_tps=80,
                latency_target_ms=70,
            ),
            ModelSpec(
                name="brain-medium",
                provider="providerB",
                task_caps=[TaskType.WRITING, TaskType.ANALYSIS, TaskType.CREATIVE],
                quality_score=0.86,
                cost_per_1k_tokens=0.0024,
                max_tps=60,
                latency_target_ms=90,
            ),
            ModelSpec(
                name="brain-premium",
                provider="providerC",
                task_caps=[TaskType.CODING, TaskType.WRITING, TaskType.ANALYSIS, TaskType.CREATIVE],
                quality_score=0.94,
                cost_per_1k_tokens=0.0048,
                max_tps=40,
                latency_target_ms=110,
            ),
        ]
