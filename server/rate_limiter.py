"""
Rate Limiting Middleware for QNA-Auth API

Provides protection against brute-force attacks on authentication endpoints.
Uses in-memory storage (suitable for single-instance deployment).
For multi-instance, use Redis backend.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RateLimitEntry:
    """Tracks request counts for a single client."""
    count: int = 0
    window_start: float = field(default_factory=time.time)
    
    def reset_if_expired(self, window_seconds: int) -> None:
        """Reset counter if window has expired."""
        now = time.time()
        if now - self.window_start >= window_seconds:
            self.count = 0
            self.window_start = now
    
    def increment(self) -> int:
        """Increment and return new count."""
        self.count += 1
        return self.count


class RateLimiter:
    """In-memory rate limiter with per-endpoint configuration."""
    
    def __init__(
        self,
        default_limit: int = 60,
        default_window: int = 60,
        endpoint_limits: Optional[Dict[str, tuple]] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            default_limit: Default requests per window
            default_window: Default window in seconds
            endpoint_limits: Dict of endpoint pattern -> (limit, window)
        """
        self.default_limit = default_limit
        self.default_window = default_window
        self.endpoint_limits = endpoint_limits or {}
        
        # Storage: {client_key: {endpoint: RateLimitEntry}}
        self._storage: Dict[str, Dict[str, RateLimitEntry]] = defaultdict(dict)
        
        # Lockout tracking for auth endpoints
        self._lockouts: Dict[str, float] = {}
    
    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier."""
        # Use X-Forwarded-For if behind proxy, otherwise client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _get_endpoint_key(self, path: str) -> str:
        """Normalize endpoint path for rate limiting."""
        # Group similar endpoints
        if "/authenticate" in path:
            return "authenticate"
        if "/enroll" in path:
            return "enroll"
        if "/challenge" in path or "/verify" in path:
            return "challenge"
        if "/devices" in path:
            return "devices"
        return "default"
    
    def _get_limits(self, endpoint_key: str) -> tuple:
        """Get (limit, window) for endpoint."""
        return self.endpoint_limits.get(
            endpoint_key, 
            (self.default_limit, self.default_window)
        )
    
    def is_locked_out(self, client_key: str) -> Optional[float]:
        """Check if client is locked out. Returns remaining lockout time or None."""
        if client_key not in self._lockouts:
            return None
        
        lockout_until = self._lockouts[client_key]
        remaining = lockout_until - time.time()
        
        if remaining <= 0:
            del self._lockouts[client_key]
            return None
        
        return remaining
    
    def add_lockout(self, client_key: str, duration: int = 300) -> None:
        """Lock out a client for specified duration (default 5 minutes)."""
        self._lockouts[client_key] = time.time() + duration
        logger.warning(f"Client {client_key} locked out for {duration}s")
    
    def check_rate_limit(self, request: Request) -> tuple:
        """
        Check if request is within rate limits.
        
        Returns:
            (allowed: bool, limit: int, remaining: int, reset_time: float)
        """
        client_key = self._get_client_key(request)
        endpoint_key = self._get_endpoint_key(request.url.path)
        
        # Check lockout
        lockout_remaining = self.is_locked_out(client_key)
        if lockout_remaining:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Retry after {int(lockout_remaining)} seconds.",
                headers={"Retry-After": str(int(lockout_remaining))}
            )
        
        limit, window = self._get_limits(endpoint_key)
        
        # Get or create entry
        if endpoint_key not in self._storage[client_key]:
            self._storage[client_key][endpoint_key] = RateLimitEntry()
        
        entry = self._storage[client_key][endpoint_key]
        entry.reset_if_expired(window)
        
        # Check limit
        if entry.count >= limit:
            reset_time = entry.window_start + window
            remaining = int(reset_time - time.time())
            
            logger.warning(
                f"Rate limit exceeded: {client_key} on {endpoint_key} "
                f"({entry.count}/{limit})"
            )
            
            return False, limit, 0, reset_time
        
        # Increment and allow
        entry.increment()
        remaining = limit - entry.count
        reset_time = entry.window_start + window
        
        return True, limit, remaining, reset_time
    
    def record_auth_failure(self, request: Request, max_failures: int = 5) -> None:
        """Record authentication failure. Lock out after max_failures."""
        client_key = self._get_client_key(request)
        
        if "auth_failures" not in self._storage[client_key]:
            self._storage[client_key]["auth_failures"] = RateLimitEntry()
        
        entry = self._storage[client_key]["auth_failures"]
        entry.reset_if_expired(300)  # 5 minute window for failures
        
        count = entry.increment()
        
        if count >= max_failures:
            self.add_lockout(client_key, duration=300)
            # Reset failure counter
            entry.count = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number cleaned."""
        now = time.time()
        cleaned = 0
        
        # Clean lockouts
        expired_lockouts = [
            k for k, v in self._lockouts.items() 
            if v < now
        ]
        for k in expired_lockouts:
            del self._lockouts[k]
            cleaned += 1
        
        # Clean old entries (older than 1 hour)
        stale_threshold = now - 3600
        clients_to_remove = []
        
        for client_key, endpoints in self._storage.items():
            endpoints_to_remove = []
            for endpoint_key, entry in endpoints.items():
                if entry.window_start < stale_threshold:
                    endpoints_to_remove.append(endpoint_key)
            
            for ep in endpoints_to_remove:
                del endpoints[ep]
                cleaned += 1
            
            if not endpoints:
                clients_to_remove.append(client_key)
        
        for ck in clients_to_remove:
            del self._storage[ck]
        
        return cleaned


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate_limiter: RateLimiter, enabled: bool = True):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.enabled = enabled
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks and docs
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
        if any(request.url.path.startswith(p) for p in skip_paths):
            return await call_next(request)
        
        try:
            allowed, limit, remaining, reset_time = self.rate_limiter.check_rate_limit(request)
        except HTTPException:
            raise
        
        if not allowed:
            retry_after = int(reset_time - time.time())
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_time))
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response


# Default configuration for QNA-Auth
def create_default_rate_limiter() -> RateLimiter:
    """Create rate limiter with default QNA-Auth configuration."""
    return RateLimiter(
        default_limit=60,  # 60 requests per minute default
        default_window=60,
        endpoint_limits={
            "authenticate": (5, 60),   # 5 auth attempts per minute
            "enroll": (3, 60),         # 3 enrollments per minute
            "challenge": (10, 60),     # 10 challenges per minute
            "devices": (20, 60),       # 20 device operations per minute
        }
    )
