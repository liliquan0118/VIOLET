"""Central LLM client factory.

Usage in src modules:
    from src.core.llm_client import get_llm_client
    client = get_llm_client()
    resp = client.chat.completions.create(model=model, messages=[...])

Usage in scripts (call once at startup):
    from src.core.llm_client import load_provider_env
    env_model = load_provider_env(args.provider)
    model = args.model or env_model
"""

from __future__ import annotations

import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROVIDER_DEFAULTS = {
    "openai":   "gpt-4o",
    "deepseek": "deepseek-chat",
    "claude":   "claude-sonnet-4-6",
    "qwen":     "qwen3.8-max",
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Aliyun Bailian / DashScope OpenAI-compatible endpoint; override with DASHSCOPE_BASE_URL in .env
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _parse_env_file(path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if v and v.lower() != "none":
                        result[k] = v
    except FileNotFoundError:
        pass
    return result


def _apply_provider(provider: str, env: dict, model_key: str) -> str:
    """Configure env vars for a provider and return the resolved model name."""
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(f"Unknown provider {provider!r}. Choose: openai, deepseek, claude, qwen")

    if provider == "openai":
        api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        model   = env.get(model_key) or env.get("OPENAI_MODEL") or _PROVIDER_DEFAULTS["openai"]
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.pop("OPENAI_BASE_URL", None)

    elif provider == "deepseek":
        api_key = env.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        model   = env.get(model_key) or env.get("DEEPSEEK_MODEL") or _PROVIDER_DEFAULTS["deepseek"]
        os.environ["OPENAI_API_KEY"]   = api_key
        os.environ["OPENAI_BASE_URL"]  = _DEEPSEEK_BASE_URL
        os.environ["DEEPSEEK_API_KEY"] = api_key  # for LiteLLM (used by tau2-bench)

    elif provider == "claude":
        api_key = env.get("CLAUDE_API_KEY") or os.environ.get("CLAUDE_API_KEY", "")
        model   = env.get(model_key) or env.get("CLAUDE_MODEL") or _PROVIDER_DEFAULTS["claude"]
        os.environ["ANTHROPIC_API_KEY"] = api_key

    elif provider == "qwen":
        # Qwen is reached through an OpenAI-compatible endpoint.  Deliberately do NOT
        # overwrite OPENAI_API_KEY / OPENAI_BASE_URL: the same process may still talk
        # to OpenAI (e.g. the tau2 user simulator).  Callers that need the key and
        # base url pass them per request via agent_litellm_config().
        api_key = env.get("DASHSCOPE_API_KEY") or env.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
        model   = env.get(model_key) or env.get("QWEN_MODEL") or _PROVIDER_DEFAULTS["qwen"]
        base = env.get("DASHSCOPE_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or _DASHSCOPE_BASE_URL
        os.environ["DASHSCOPE_API_KEY"]  = api_key
        os.environ["DASHSCOPE_BASE_URL"] = base   # read by our OpenAI-SDK client
        os.environ["DASHSCOPE_API_BASE"] = base   # read by litellm's dashscope provider

    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"]    = model
    return model


def load_provider_env(provider: str, env_file: str | None = None) -> str:
    """Legacy: configure a single provider and return model name.

    Reads TESTER_PROVIDER / TESTER_MODEL from .env when provider matches,
    otherwise uses the explicit provider arg.  Call once at script startup.
    """
    env_path = env_file or os.path.join(_PROJECT_ROOT, ".env")
    env = _parse_env_file(env_path)
    return _apply_provider(provider, env, "TESTER_MODEL")


def load_tester_env(env_file: str | None = None) -> tuple[str, str]:
    """Configure the *tester* side (user simulator, oracle, planner) from .env.

    Reads TESTER_PROVIDER and TESTER_MODEL.
    Returns (provider, model).
    """
    env_path = env_file or os.path.join(_PROJECT_ROOT, ".env")
    env = _parse_env_file(env_path)
    provider = env.get("TESTER_PROVIDER") or os.environ.get("TESTER_PROVIDER", "openai")
    model = _apply_provider(provider, env, "TESTER_MODEL")
    return provider, model


def load_agent_env(env_file: str | None = None) -> tuple[str, str]:
    """Return the *target agent* provider and model from .env (does NOT set env vars).

    These are passed to tau2-bench to select which LLM drives the agent under test.
    Returns (provider, model).
    """
    env_path = env_file or os.path.join(_PROJECT_ROOT, ".env")
    env = _parse_env_file(env_path)
    provider = env.get("AGENT_PROVIDER") or os.environ.get("AGENT_PROVIDER", "openai")
    if provider == "openai":
        model = env.get("AGENT_MODEL") or env.get("OPENAI_MODEL") or _PROVIDER_DEFAULTS["openai"]
    elif provider == "deepseek":
        model = env.get("AGENT_MODEL") or env.get("DEEPSEEK_MODEL") or _PROVIDER_DEFAULTS["deepseek"]
    elif provider == "claude":
        model = env.get("AGENT_MODEL") or env.get("CLAUDE_MODEL") or _PROVIDER_DEFAULTS["claude"]
    elif provider == "qwen":
        model = env.get("AGENT_MODEL") or env.get("QWEN_MODEL") or _PROVIDER_DEFAULTS["qwen"]
    else:
        raise ValueError(f"Unknown AGENT_PROVIDER={provider!r}")
    return provider, model


def agent_litellm_config(provider: str, model: str, env_file: str | None = None) -> tuple[str, dict]:
    """Return (litellm_model_name, extra_llm_args) for driving a tau2 agent with `model`.

    Qwen goes through litellm's native ``dashscope/`` provider, which reads
    DASHSCOPE_API_KEY / DASHSCOPE_API_BASE from the environment.  The key is therefore
    never placed in llm_args (tau2 persists llm_args into the result file) and the
    process-wide OPENAI_* env stays untouched for the user simulator.
    Other providers are passed through unchanged.
    """
    if provider != "qwen":
        return model, {}
    env = _parse_env_file(env_file or os.path.join(_PROJECT_ROOT, ".env"))
    api_key = env.get("DASHSCOPE_API_KEY") or env.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    api_base = env.get("DASHSCOPE_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or _DASHSCOPE_BASE_URL
    if not api_key:
        raise ValueError("qwen provider selected but DASHSCOPE_API_KEY / QWEN_API_KEY not set in .env")
    os.environ["DASHSCOPE_API_KEY"] = api_key
    os.environ["DASHSCOPE_API_BASE"] = api_base
    name = model if "/" in model else f"dashscope/{model}"
    return name, {}


def get_default_model() -> str:
    """Return the model name set by load_provider_env, or 'gpt-4o' as fallback."""
    return os.environ.get("LLM_MODEL", "gpt-4o")


def get_llm_client():
    """Return an OpenAI-compatible client for the current LLM_PROVIDER.

    For openai/deepseek: returns an openai.OpenAI instance.
    For claude: returns an adapter that wraps the anthropic SDK with the same interface.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai")

    if provider in ("openai", "deepseek"):
        from openai import OpenAI
        kwargs: dict = {}
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    if provider == "claude":
        return _AnthropicAdapter()

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Choose: openai, deepseek, claude")


# ── Anthropic adapter ─────────────────────────────────────────────────────────

class _AnthropicAdapter:
    """Wraps anthropic.Anthropic with the same interface as openai.OpenAI."""

    def __init__(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError as exc:
            raise ImportError("Run 'pip install anthropic' to use the claude provider") from exc
        # Mirror the openai attribute chain: client.chat.completions.create(...)
        self.chat = _ChatNamespace(self)

    def _create(self, model: str, messages: list, temperature: float = 0.0,
                max_tokens: int = 4096, **_: object) -> "_OAIResponse":
        import anthropic
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [{"role": m["role"], "content": m["content"]}
                         for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_parts)

        kwargs: dict = dict(
            model=model,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_text:
            kwargs["system"] = system_text

        resp = self._client.messages.create(**kwargs)
        return _OAIResponse(resp.content[0].text)


class _ChatNamespace:
    def __init__(self, adapter: _AnthropicAdapter) -> None:
        self.completions = _CompletionsNamespace(adapter)


class _CompletionsNamespace:
    def __init__(self, adapter: _AnthropicAdapter) -> None:
        self._adapter = adapter

    def create(self, **kwargs: object) -> "_OAIResponse":
        return self._adapter._create(**kwargs)  # type: ignore[arg-type]


class _OAIResponse:
    """Minimal OpenAI-response-compatible wrapper."""
    def __init__(self, text: str) -> None:
        self.choices = [_OAIChoice(text)]


class _OAIChoice:
    def __init__(self, text: str) -> None:
        self.message = _OAIMessage(text)


class _OAIMessage:
    def __init__(self, text: str) -> None:
        self.content = text
