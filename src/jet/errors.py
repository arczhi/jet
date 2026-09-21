"""Error taxonomy.

Every failure jet can produce has a named type here so callers can translate
errors at the boundary instead of inspecting message text. Provider errors carry
``retryable`` so the agent loop can make a cost-aware retry decision.
"""

from __future__ import annotations


class JetError(Exception):
    """Base class for all jet errors."""


class ConfigError(JetError):
    """Configuration is missing or invalid."""


class ProviderError(JetError):
    """A model provider failed."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ProviderAuthError(ProviderError):
    def __init__(self, message: str = "provider rejected credentials", status: int = 401):
        super().__init__(message, retryable=False, status=status)


class ProviderRateLimitError(ProviderError):
    def __init__(self, message: str = "provider rate limited", status: int = 429):
        super().__init__(message, retryable=True, status=status)


class ProviderUnavailableError(ProviderError):
    def __init__(self, message: str = "provider unavailable", status: int | None = None):
        super().__init__(message, retryable=True, status=status)


class ProviderBadResponseError(ProviderError):
    def __init__(self, message: str = "provider returned an invalid response"):
        super().__init__(message, retryable=False)


class ToolError(JetError):
    """A tool failed or was used incorrectly."""


class ToolNotFoundError(ToolError):
    pass


class ToolDeniedError(ToolError):
    """Policy refused the tool call."""


class ToolExecutionError(ToolError):
    pass


class PolicyDeniedError(JetError):
    """A policy rule denied an action outright."""


class BudgetExceededError(JetError):
    """A cost or token budget was exceeded."""
