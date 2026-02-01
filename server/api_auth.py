"""
API Key Authentication Middleware for QNA-Auth

Provides optional API key authentication for securing endpoints.
Enable by setting API_KEY_ENABLED=true and API_KEY in environment.
"""

from __future__ import annotations

import secrets
import logging
from typing import Optional
from functools import wraps

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# API key header and query parameter names
API_KEY_HEADER = "X-API-Key"
API_KEY_QUERY = "api_key"

# Security schemes for OpenAPI docs
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_QUERY, auto_error=False)


class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: bool = False,
        exempt_paths: Optional[list] = None
    ):
        """
        Initialize API key auth.
        
        Args:
            api_key: The valid API key
            enabled: Whether auth is enabled
            exempt_paths: Paths that don't require auth
        """
        self.api_key = api_key
        self.enabled = enabled
        self.exempt_paths = exempt_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/"
        ]
    
    def is_exempt(self, path: str) -> bool:
        """Check if path is exempt from authentication."""
        return any(path.startswith(p) or path == p for p in self.exempt_paths)
    
    def validate_key(self, key: Optional[str]) -> bool:
        """Validate an API key."""
        if not self.api_key:
            return False
        if not key:
            return False
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(key, self.api_key)
    
    async def get_api_key(
        self,
        request: Request,
        api_key_header: str = None,
        api_key_query: str = None
    ) -> Optional[str]:
        """Extract API key from request."""
        # Try header first, then query parameter
        key = api_key_header or api_key_query
        
        if not key:
            # Also check request object
            key = request.headers.get(API_KEY_HEADER)
            if not key:
                key = request.query_params.get(API_KEY_QUERY)
        
        return key
    
    def require_auth(self, request: Request) -> None:
        """
        Validate authentication for a request.
        
        Raises:
            HTTPException: If auth fails
        """
        if not self.enabled:
            return
        
        if self.is_exempt(request.url.path):
            return
        
        # Extract API key
        key = request.headers.get(API_KEY_HEADER)
        if not key:
            key = request.query_params.get(API_KEY_QUERY)
        
        if not key:
            logger.warning(f"Missing API key for {request.url.path}")
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        if not self.validate_key(key):
            logger.warning(f"Invalid API key for {request.url.path}")
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API key authentication."""
    
    def __init__(self, app, auth: APIKeyAuth):
        super().__init__(app)
        self.auth = auth
    
    async def dispatch(self, request: Request, call_next):
        try:
            self.auth.require_auth(request)
        except HTTPException:
            raise
        
        return await call_next(request)


def get_api_key_dependency(auth: APIKeyAuth):
    """
    Create a FastAPI dependency for API key auth.
    
    Usage:
        auth = APIKeyAuth(api_key="secret", enabled=True)
        
        @app.get("/protected")
        async def protected(api_key: str = Depends(get_api_key_dependency(auth))):
            return {"message": "Authenticated!"}
    """
    async def verify_api_key(
        request: Request,
        header_key: str = Security(api_key_header),
        query_key: str = Security(api_key_query)
    ) -> str:
        if not auth.enabled:
            return "disabled"
        
        if auth.is_exempt(request.url.path):
            return "exempt"
        
        key = header_key or query_key
        
        if not key:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        if not auth.validate_key(key):
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        
        return key
    
    return verify_api_key


def generate_api_key(length: int = 32) -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(length)


# Helper to create auth from config
def create_api_key_auth_from_env() -> APIKeyAuth:
    """Create APIKeyAuth from environment variables."""
    import os
    
    enabled = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
    api_key = os.getenv("API_KEY")
    
    return APIKeyAuth(
        api_key=api_key,
        enabled=enabled
    )
