"""Allowlisted provider diagnostics; never persist arbitrary provider text."""

SDK_CATEGORIES = {
    "BadRequestError": "request_rejected",
    "UnprocessableEntityError": "request_rejected",
    "AuthenticationError": "authentication_failed",
    "PermissionDeniedError": "permission_denied",
    "NotFoundError": "resource_not_found",
    "ConflictError": "request_conflict",
    "RateLimitError": "rate_limited",
    "InternalServerError": "provider_server_error",
    "APITimeoutError": "timeout",
    "APIConnectionError": "connection_failed",
    "APIStatusError": "unknown_http_error",
    "APIResponseValidationError": "provider_response_invalid",
    "APIError": "unknown_provider_error",
}
SAFE_EXCEPTION_NAMES = set(SDK_CATEGORIES) | {
    "RuntimeError", "ValueError", "TypeError", "KeyError", "OSError", "TimeoutError",
}
SAFE_CODES = {
    "invalid_request_error", "invalid_parameter", "invalid_value", "unsupported_parameter",
    "unsupported_value", "missing_required_parameter", "model_not_found", "model_not_available",
    "context_length_exceeded", "rate_limit_exceeded", "insufficient_quota", "invalid_api_key",
    "authentication_error", "permission_denied", "server_error", "internal_server_error",
    "service_unavailable", "overloaded_error", "content_policy_violation",
}
SAFE_PARAMETERS = {
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
    "response_format", "response_format.type", "thinking", "thinking.type", "stream",
    "tools", "tool_choice", "top_p", "n", "stop",
}


def _known(value, allowed):
    return value if isinstance(value, str) and value in allowed else None


def diagnose_provider_error(exc):
    """Return fixed labels, numeric HTTP status and exact allowlisted fields.

    Message hints are lossy, unverified provider claims, not a recovered root
    cause. No exception string, URL, request, headers or response text is read.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    status = status if type(status) is int and 100 <= status <= 599 else None
    category = SDK_CATEGORIES.get(name, "unexpected_client_error")
    basis = "sdk_exception_class" if name in SDK_CATEGORIES else "unknown_exception"
    if status is not None:
        basis = "http_status"
        category = {
            400: "request_rejected", 401: "authentication_failed", 403: "permission_denied",
            404: "resource_not_found", 408: "timeout", 409: "request_conflict",
            422: "request_rejected", 429: "rate_limited",
        }.get(status, "provider_server_error" if status >= 500 else "unknown_http_error")
    body = getattr(exc, "body", None)
    body = body if isinstance(body, dict) else {}
    error = body.get("error")
    fields = error if isinstance(error, dict) else body
    code = getattr(exc, "code", None)
    code = fields.get("code") if code is None else code
    param = getattr(exc, "param", None)
    param = fields.get("param") if param is None else param
    provider_type = getattr(exc, "type", None)
    provider_type = fields.get("type") if provider_type is None else provider_type
    # Only a small string is inspected in memory. Nothing copied from it is
    # returned; even unknown codes and parameter names are suppressed.
    message = fields.get("message")
    if not isinstance(message, str):
        message = getattr(exc, "message", None)
    message = message[:8192].lower() if isinstance(message, str) else ""
    hint_rules = {
        "unsupported_parameter_reported": ("unsupported parameter", "unrecognized request argument", "unknown parameter"),
        "invalid_parameter_reported": ("invalid parameter", "invalid value", "unsupported value"),
        "missing_parameter_reported": ("missing required", "required parameter"),
        "model_unavailable_reported": ("model not found", "model does not exist", "model is not available", "model unavailable"),
        "context_limit_reported": ("context length", "context_length_exceeded", "maximum context"),
        "quota_issue_reported": ("insufficient quota", "insufficient balance", "quota exceeded"),
        "json_format_requirement_reported": ("must contain the word 'json'", "must contain the word \"json\""),
    }
    hints = sorted(label for label, phrases in hint_rules.items() if any(p in message for p in phrases))
    return {
        "schema_version": "agentspectesting.v5-provider-error-diagnostics/v0.1",
        "exception_type": name if name in SAFE_EXCEPTION_NAMES else "OtherException",
        "category": category, "classification_basis": basis, "http_status": status,
        "provider_code": _known(code, SAFE_CODES),
        "provider_error_type": _known(provider_type, SAFE_CODES),
        "provider_parameter": _known(param, SAFE_PARAMETERS),
        "provider_message_hints": hints,
        "provider_claims_verified": False,
        "raw_message_retained": False,
        "redaction_policy": "fixed_labels_and_exact_allowlists_only",
    }
