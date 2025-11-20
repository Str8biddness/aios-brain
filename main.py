"""
main.py — AIOS Brain reasoning engine (FastAPI)

Production-grade FastAPI application providing:
- Async endpoints for health, reasoning, model routing, memory management
- Prometheus metrics (/metrics)
- JWT authentication middleware with role-based route protection
- CORS configuration
- Comprehensive error handling and structured logging
- Graceful startup/shutdown with resource cleanup
- Environment-based configuration via Pydantic
- Concurrency-ready design (100+ concurrent requests)
- Response time optimization (<100ms target) with lightweight handlers and metrics

Run:
    uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4

Notes:
- Ensure environment variables are set appropriately (see AppSettings).
- Use async-safe operations; avoid blocking calls within endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, BaseSettings, Field, ValidationError
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, Gauge, generate_latest
import jwt  # PyJWT
from jwt import InvalidTokenError

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Environment-based application settings."""
    app_name: str = Field(default="AIOS Brain")
    environment: str = Field(default="production")  # production, staging, development
    debug: bool = Field(default=False)
    jwt_secret: str = Field(default="change_me")
    jwt_algorithm: str = Field(default="HS256")
    cors_origins: str = Field(default="*")  # comma-separated list
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: str = Field(default="GET,POST,OPTIONS")
    cors_allow_headers: str = Field(default="Authorization,Content-Type")
    request_timeout_ms: int = Field(default=80)  # helps target <100ms by design (soft timeout)
    max_memory_items: int = Field(default=10000)
    metrics_namespace: str = Field(default="aios_brain")
    reasoning_max_tokens: int = Field(default=2048)
    routing_default_model: str = Field(default="brain-small")
    shutdown_grace_period_s: int = Field(default=5)

    class Config:
        env_prefix = "AIOS_"
        case_sensitive = False


settings = AppSettings()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.DEBUG if settings.debug else logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("aios-brain")

# -----------------------------------------------------------------------------
# Metrics (Prometheus)
# -----------------------------------------------------------------------------

registry = CollectorRegistry()
REQ_COUNTER = Counter(
    f"{settings.metrics_namespace}_http_requests_total",
    "Total HTTP requests processed",
    ["path", "method", "status"],
    registry=registry,
)
LATENCY_HIST = Histogram(
    f"{settings.metrics_namespace}_request_latency_ms",
    "Request latency in milliseconds",
    ["path", "method"],
    buckets=(5, 10, 25, 50, 75, 100, 200, 400, 800, 1600),
    registry=registry,
)
INFLIGHT_GAUGE = Gauge(
    f"{settings.metrics_namespace}_inflight_requests",
    "In-flight requests",
    registry=registry,
)
REASONINGS_COUNTER = Counter(
    f"{settings.metrics_namespace}_reasonings_total",
    "Total reasoning requests",
    ["model", "result"],
    registry=registry,
)

# -----------------------------------------------------------------------------
# Security / JWT
# -----------------------------------------------------------------------------

class TokenPayload(BaseModel):
    """Decoded JWT payload schema."""
    sub: str
    role: str = "user"
    exp: Optional[int] = None  # epoch seconds


async def jwt_auth_dependency(request: Request) -> TokenPayload:
    """Dependency to authenticate request via JWT Bearer token."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        decoded = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        payload = TokenPayload(**decoded)
        return payload
    except (InvalidTokenError, ValidationError) as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def require_role(*roles: str):
    """Role-based access decorator-like dependency."""
    async def _dep(payload: TokenPayload = Depends(jwt_auth_dependency)) -> TokenPayload:
        if payload.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return payload
    return _dep

# -----------------------------------------------------------------------------
# Models / Schemas
# -----------------------------------------------------------------------------

class ReasoningRequest(BaseModel):
    """Input schema for reasoning requests."""
    input: str = Field(min_length=1, max_length=10000)
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    context_ids: Optional[list[str]] = None


class ReasoningResponse(BaseModel):
    """Output schema for reasoning responses."""
    output: str
    model: str
    latency_ms: int


class RouteModelRequest(BaseModel):
    """Model routing request to select best model for a task."""
    task: str
    constraints: Optional[Dict[str, Any]] = None


class RouteModelResponse(BaseModel):
    """Model routing response."""
    model: str
    reason: str


class MemoryItem(BaseModel):
    """Memory item structure."""
    id: str
    data: Dict[str, Any]
    created_at: float


class MemoryUpsertRequest(BaseModel):
    """Request to upsert memory items."""
    items: list[MemoryItem]


class MemoryQueryRequest(BaseModel):
    """Request to query memory by IDs."""
    ids: list[str]


class MemoryQueryResponse(BaseModel):
    """Response with found memory items."""
    items: list[MemoryItem]

# -----------------------------------------------------------------------------
# In-memory store (async-friendly)
# -----------------------------------------------------------------------------

class MemoryStore:
    """Async-friendly in-memory store with bounded size."""
    def __init__(self, max_items: int) -> None:
        self._items: Dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
        self._max = max_items

    async def upsert(self, items: list[MemoryItem]) -> int:
        async with self._lock:
            for it in items:
                if len(self._items) >= self._max and it.id not in self._items:
                    # Simple eviction strategy: drop oldest
                    oldest_id = min(self._items, key=lambda k: self._items[k].created_at)
                    self._items.pop(oldest_id, None)
                self._items[it.id] = it
            return len(items)

    async def get_many(self, ids: list[str]) -> list[MemoryItem]:
        async with self._lock:
            return [self._items[i] for i in ids if i in self._items]

    async def size(self) -> int:
        async with self._lock:
            return len(self._items)

memory_store = MemoryStore(settings.max_memory_items)

# -----------------------------------------------------------------------------
# Reasoning / Model routing (stubs optimized for latency)
# -----------------------------------------------------------------------------

async def select_model(task: str, constraints: Optional[Dict[str, Any]] = None) -> str:
    """Select an appropriate model based on task/constraints (fast heuristic)."""
    # Minimal heuristic to avoid blocking: choose default model or constraint-specified.
    if constraints and isinstance(constraints.get("prefer"), str):
        return constraints["prefer"]
    if "long" in task.lower():
        return "brain-medium"
    if "code" in task.lower():
        return "brain-code"
    return settings.routing_default_model

async def run_reasoning(model: str, input_text: str, max_tokens: int, temperature: float) -> str:
    """
    Simulated non-blocking reasoning function.
    In production, this should dispatch to async inference backends (CUDA/OpenCL/NPU).
    Designed to be sub-50ms by using minimal processing here.
    """
    # Non-blocking tiny await to yield the event loop; emulate fast compute.
    await asyncio.sleep(0)  # yields control; keep endpoint responsive
    # Echo-completion stub (replace with real engine call)
    return f"[{model}] {input_text[:max_tokens]}"

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

app = FastAPI(title=settings.app_name, debug=settings.debug)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=[m.strip() for m in settings.cors_allow_methods.split(",")],
    allow_headers=[h.strip() for h in settings.cors_allow_headers.split(",")],
)

# -----------------------------------------------------------------------------
# Middleware: Metrics + Timeout
# -----------------------------------------------------------------------------

@app.middleware("http")
async def metrics_and_timeout(request: Request, call_next):
    """Collect latency metrics and enforce soft request timeout."""
    start = time.perf_counter()
    path = request.url.path
    method = request.method
    INFLIGHT_GAUGE.inc()
    try:
        # Soft timeout using wait_for; don't kill server task, handle gracefully.
        response: Response = await asyncio.wait_for(call_next(request), timeout=settings.request_timeout_ms / 1000.0)
    except asyncio.TimeoutError:
        LATENCY_HIST.labels(path=path, method=method).observe((time.perf_counter() - start) * 1000)
        REQ_COUNTER.labels(path=path, method=method, status=str(status.HTTP_504_GATEWAY_TIMEOUT)).inc()
        INFLIGHT_GAUGE.dec()
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "Request timed out", "target_ms": settings.request_timeout_ms},
        )
    except Exception as e:
        logger.exception(f"Unhandled error during request: {e}")
        LATENCY_HIST.labels(path=path, method=method).observe((time.perf_counter() - start) * 1000)
        REQ_COUNTER.labels(path=path, method=method, status=str(status.HTTP_500_INTERNAL_SERVER_ERROR)).inc()
        INFLIGHT_GAUGE.dec()
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal server error"})
    latency_ms = int((time.perf_counter() - start) * 1000)
    LATENCY_HIST.labels(path=path, method=method).observe(latency_ms)
    REQ_COUNTER.labels(path=path, method=method, status=str(response.status_code)).inc()
    INFLIGHT_GAUGE.dec()
    return response

# -----------------------------------------------------------------------------
# Routers
# -----------------------------------------------------------------------------

health_router = APIRouter()
api_router = APIRouter(prefix="/api")

@health_router.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    """Liveness/readiness probe."""
    # Quick async checks (e.g., memory store size)
    size = await memory_store.size()
    return f"ok:{size}"

@health_router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@api_router.post("/route/model", response_model=RouteModelResponse, dependencies=[Depends(require_role("user", "admin"))])
async def route_model(req: RouteModelRequest) -> RouteModelResponse:
    """Route to the most suitable model based on task and constraints."""
    model = await select_model(req.task, req.constraints)
    return RouteModelResponse(model=model, reason="heuristic-selection")

@api_router.post("/reason", response_model=ReasoningResponse, dependencies=[Depends(require_role("user", "admin"))])
async def reason(req: ReasoningRequest) -> ReasoningResponse:
    """Execute a reasoning request using the routed or specified model."""
    model = req.model or await select_model(req.input, {"prefer": req.model} if req.model else None)
    max_tokens = min(req.max_tokens or settings.reasoning_max_tokens, settings.reasoning_max_tokens)
    t0 = time.perf_counter()
    output = await run_reasoning(model=model, input_text=req.input, max_tokens=max_tokens, temperature=req.temperature or 0.2)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    REASONINGS_COUNTER.labels(model=model, result="ok").inc()
    return ReasoningResponse(output=output, model=model, latency_ms=latency_ms)

@api_router.post("/memory/upsert", dependencies=[Depends(require_role("admin"))])
async def memory_upsert(req: MemoryUpsertRequest) -> Dict[str, Any]:
    """Upsert memory items (admin-only)."""
    count = await memory_store.upsert(req.items)
    return {"upserted": count}

@api_router.post("/memory/query", response_model=MemoryQueryResponse, dependencies=[Depends(require_role("user", "admin"))])
async def memory_query(req: MemoryQueryRequest) -> MemoryQueryResponse:
    """Query memory items by IDs."""
    items = await memory_store.get_many(req.ids)
    return MemoryQueryResponse(items=items)

app.include_router(health_router)
app.include_router(api_router)

# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.info(f"HTTPException {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": "Validation failed", "details": exc.errors()})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal server error"})

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------

_shutdown_event = asyncio.Event()

async def _signal_handler():
    logger.info("Shutdown signal received, starting graceful shutdown...")
    _shutdown_event.set()

def _install_signal_handlers():
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_signal_handler()))
        except NotImplementedError:
            # Windows may not support add_signal_handler; rely on server shutdown hooks
            logger.debug(f"Signal handler not supported for {sig}")

@app.on_event("startup")
async def on_startup():
    logger.info(f"Starting {settings.app_name} (env={settings.environment}, debug={settings.debug})")
    _install_signal_handlers()
    # Pre-warm lightweight components if needed to reduce cold-start latency
    await memory_store.upsert([
        MemoryItem(id="__warm__", data={"status": "ready"}, created_at=time.time())
    ])
    logger.info("Startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Initiating shutdown sequence...")
    # Wait for graceful shutdown signal or time out
    try:
        await asyncio.wait_for(_shutdown_event.wait(), timeout=settings.shutdown_grace_period_s)
    except asyncio.TimeoutError:
        logger.info("Shutdown grace period elapsed, proceeding with teardown")
    # Cleanup resources (close pools, release connections, flush metrics if needed)
    logger.info("Shutdown complete")

# -----------------------------------------------------------------------------
# Concurrency and performance notes
# -----------------------------------------------------------------------------
# - Endpoints are fully async and avoid blocking operations.
# - Use multiple workers (e.g., --workers 4) to handle 100+ concurrent requests.
# - Keep request_timeout_ms tuned to avoid tail-latency explosions.
# - Replace run_reasoning with real async inference backends for production.
# - Enable HTTP keep-alive and configure client-side timeouts appropriately.
# -----------------------------------------------------------------------------
