"""Local guard for the configured endpoint's JSON-object message requirement."""


def validate_json_object_messages(messages):
    """Check text actually sent, never packet metadata or serialized field names.

    This narrow prerequisite does not guarantee provider acceptance or valid
    JSON output. Callers must invoke it before config loading / API dispatch.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("json_object requires nonempty text messages")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("json_object review messages must have text content")
    if not any("json" in message["content"].lower() for message in messages):
        raise ValueError("json_object requires an explicit JSON mention in outbound message content")
