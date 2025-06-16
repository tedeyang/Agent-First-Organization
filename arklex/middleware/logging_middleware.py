"""Request logging middleware for the Arklex framework.

This module provides middleware for logging HTTP requests and responses,
including request tracking, timing, and error handling with retry mechanisms.
"""

import time
import uuid
from typing import Callable, Any, Optional, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from tenacity import retry, stop_after_attempt, wait_exponential

from arklex.utils.logging_config import get_logger, log_with_context
from arklex.utils.exceptions import (
    RetryableError,
    NetworkError,
    TimeoutError,
    ServiceUnavailableError,
)

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses.

    This middleware provides:
    - Request ID generation and tracking
    - Request/response logging with context
    - Error handling with retry mechanisms
    - Performance timing
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware.

        Args:
            app: The ASGI application.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log relevant information.

        Args:
            request: The incoming request.
            call_next: The next middleware in the chain.

        Returns:
            The response from the application.

        Raises:
            RetryableError: If the request fails with a retryable error.
            Exception: For other types of errors.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Prepare request context
        request_context = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else None,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
        }

        # Log request start
        log_with_context(
            logger,
            "INFO",
            "Request started",
            context=request_context,
        )

        start_time = time.time()
        try:
            # Process the request with retry mechanism for retryable errors
            response = await self._process_request_with_retry(request, call_next)

            # Calculate processing time
            process_time = time.time() - start_time

            # Prepare response context
            response_context = {
                **request_context,
                "status_code": response.status_code,
                "process_time": process_time,
                "response_headers": dict(response.headers),
            }

            # Log request completion
            log_with_context(
                logger,
                "INFO",
                "Request completed",
                context=response_context,
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except RetryableError as e:
            # Log retryable error
            error_context = {
                **request_context,
                "error": str(e),
                "error_type": type(e).__name__,
                "max_retries": e.max_retries,
            }
            log_with_context(
                logger,
                "ERROR",
                f"Retryable error occurred: {str(e)}",
                context=error_context,
                exc_info=e,
            )
            raise

        except Exception as e:
            # Log unexpected error
            error_context = {
                **request_context,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            log_with_context(
                logger,
                "ERROR",
                f"Unexpected error occurred: {str(e)}",
                context=error_context,
                exc_info=e,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            lambda e: isinstance(
                e, (NetworkError, TimeoutError, ServiceUnavailableError)
            )
        ),
    )
    async def _process_request_with_retry(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process the request with retry mechanism for retryable errors.

        Args:
            request: The incoming request.
            call_next: The next middleware in the chain.

        Returns:
            The response from the application.

        Raises:
            RetryableError: If the request fails with a retryable error.
            Exception: For other types of errors.
        """
        try:
            return await call_next(request)
        except Exception as e:
            # Convert certain exceptions to retryable errors
            if isinstance(e, (ConnectionError, TimeoutError)):
                raise NetworkError(str(e))
            elif isinstance(e, TimeoutError):
                raise TimeoutError(str(e))
            elif isinstance(e, ServiceUnavailableError):
                raise ServiceUnavailableError(str(e))
            raise
