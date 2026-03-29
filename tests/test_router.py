import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock

from core.classifier import Classification, ModelTier, TaskType
from core.config import AppConfig, ModelMap, ProviderConfig, RoutingConfig, Thresholds, TaskCapConfig
from providers.base import ProviderResponse
from router.proxy import Proxy, ProxyResult
from router.selector import ModelSelector


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def _make_config(
    default_provider: str = "groq_cloud",
    fallback_provider: str = "ollama_local",
) -> AppConfig:
    return AppConfig(
        providers={
            "groq_cloud": ProviderConfig(
                name="groq_cloud",
                type="groq",
                base_url="https://api.groq.com/openai/v1",
                api_key="test-groq-key",
                models=ModelMap(
                    small="llama-3.1-8b-instant",
                    medium="llama-3.3-70b-versatile",
                    large="llama-3.3-70b-versatile",
                ),
            ),
            "ollama_local": ProviderConfig(
                name="ollama_local",
                type="ollama",
                base_url="http://localhost:11434",
                api_key=None,
                models=ModelMap(
                    small="qwen2.5:3b",
                    medium="qwen2.5:7b",
                    large="qwen2.5:14b",
                ),
            ),
        },
        routing=RoutingConfig(
            default_provider=default_provider,
            fallback_provider=fallback_provider,
            fallback_model="medium",
        ),
        thresholds=Thresholds(
            low_confidence=0.5,
            misclassification_ratio=2.0,
            overprovisioned_ratio=0.3,
        ),
        task_caps={
            t.value: TaskCapConfig(base_cap=300, floor=80)
            for t in TaskType
        },
    )


def _classification(
    task_type: TaskType = TaskType.FACTUAL,
    tier: ModelTier = ModelTier.SMALL,
    max_tokens: int = 80,
    confidence: float = 0.88,
) -> Classification:
    return Classification(
        task_type=task_type,
        model_tier=tier,
        max_tokens=max_tokens,
        confidence=confidence,
        signals_matched=4,
        signals_checked=7,
    )


def _provider_response(
    content: str = "Hello",
    was_truncated: bool = False,
    output_tokens: int = 10,
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        input_tokens=5,
        output_tokens=output_tokens,
        model="llama-3.1-8b-instant",
        was_truncated=was_truncated,
    )


# ------------------------------------------------------------------ #
# ModelSelector                                                        #
# ------------------------------------------------------------------ #

def test_selector_picks_default_provider():
    config = _make_config()
    selector = ModelSelector(config)
    _, _, provider_name = selector.select(_classification())
    assert provider_name == "groq_cloud"


def test_selector_respects_force_provider():
    config = _make_config()
    selector = ModelSelector(config)
    _, _, provider_name = selector.select(_classification(), force_provider="ollama_local")
    assert provider_name == "ollama_local"


def test_selector_respects_force_model():
    config = _make_config()
    selector = ModelSelector(config)
    _, model, _ = selector.select(_classification(tier=ModelTier.SMALL), force_model="large")
    assert model == "llama-3.3-70b-versatile"


def test_selector_fallback_uses_fallback_provider():
    config = _make_config()
    selector = ModelSelector(config)
    _, _, provider_name = selector.fallback(_classification())
    assert provider_name == "ollama_local"


def test_selector_upgrade_small_to_medium():
    config = _make_config()
    selector = ModelSelector(config)
    result = selector.upgrade(_classification(tier=ModelTier.SMALL), "groq_cloud")
    assert result is not None
    _, model, _ = result
    assert model == "llama-3.3-70b-versatile"


def test_selector_upgrade_medium_to_large():
    config = _make_config()
    selector = ModelSelector(config)
    result = selector.upgrade(_classification(tier=ModelTier.MEDIUM), "groq_cloud")
    assert result is not None
    _, model, _ = result
    assert model == "llama-3.3-70b-versatile"


def test_selector_upgrade_large_returns_none():
    config = _make_config()
    selector = ModelSelector(config)
    result = selector.upgrade(_classification(tier=ModelTier.LARGE), "groq_cloud")
    assert result is None


# ------------------------------------------------------------------ #
# Proxy — happy path                                                   #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_proxy_happy_path():
    config = _make_config()
    proxy = Proxy(config)

    mock_provider = AsyncMock()
    mock_provider.complete.return_value = _provider_response()
    proxy._selector._providers["groq_cloud"] = mock_provider

    result = await proxy.route(
        prompt="What is Python?",
        classification=_classification(),
    )

    assert result.was_fallback is False
    assert result.fallback_reason is None
    assert result.was_upgraded is False
    assert result.response.content == "Hello"
    mock_provider.complete.assert_called_once()


# ------------------------------------------------------------------ #
# Proxy — provider error triggers fallback                             #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_proxy_falls_back_on_http_error():
    config = _make_config()
    proxy = Proxy(config)

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("error", request=request, response=response)

    primary = AsyncMock()
    primary.complete.side_effect = error
    fallback = AsyncMock()
    fallback.complete.return_value = _provider_response(content="fallback response")

    proxy._selector._providers["groq_cloud"] = primary
    proxy._selector._providers["ollama_local"] = fallback

    result = await proxy.route(
        prompt="What is Python?",
        classification=_classification(),
    )

    assert result.was_fallback is True
    assert result.provider_name == "ollama_local"
    assert result.response.content == "fallback response"


@pytest.mark.asyncio
async def test_proxy_falls_back_on_rate_limit():
    config = _make_config()
    proxy = Proxy(config)

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    error = httpx.HTTPStatusError("rate limited", request=request, response=response)

    primary = AsyncMock()
    primary.complete.side_effect = error
    fallback = AsyncMock()
    fallback.complete.return_value = _provider_response()

    proxy._selector._providers["groq_cloud"] = primary
    proxy._selector._providers["ollama_local"] = fallback

    result = await proxy.route(
        prompt="What is Python?",
        classification=_classification(),
    )

    assert result.was_fallback is True
    assert result.fallback_reason is not None
    assert "rate" in str(result.fallback_reason)


@pytest.mark.asyncio
async def test_proxy_falls_back_on_network_error():
    config = _make_config()
    proxy = Proxy(config)

    primary = AsyncMock()
    primary.complete.side_effect = httpx.ConnectError("connection refused")
    fallback = AsyncMock()
    fallback.complete.return_value = _provider_response()

    proxy._selector._providers["groq_cloud"] = primary
    proxy._selector._providers["ollama_local"] = fallback

    result = await proxy.route(
        prompt="What is Python?",
        classification=_classification(),
    )

    assert result.was_fallback is True


# ------------------------------------------------------------------ #
# Proxy — incomplete response retries                                  #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_proxy_retries_with_more_tokens_on_truncation():
    config = _make_config()
    proxy = Proxy(config)

    truncated = _provider_response(was_truncated=True, output_tokens=80)
    complete = _provider_response(was_truncated=False, output_tokens=60)

    mock_provider = AsyncMock()
    mock_provider.complete.side_effect = [truncated, complete]
    proxy._selector._providers["groq_cloud"] = mock_provider

    result = await proxy.route(
        prompt="Explain decorators",
        classification=_classification(max_tokens=80),
    )

    assert mock_provider.complete.call_count == 2
    # second call should have higher max_tokens
    second_call_kwargs = mock_provider.complete.call_args_list[1][1]
    assert second_call_kwargs["max_tokens"] == 120  # 80 * 1.5


@pytest.mark.asyncio
async def test_proxy_upgrades_tier_after_token_retry_fails():
    config = _make_config()
    proxy = Proxy(config)

    truncated = _provider_response(was_truncated=True)
    complete = _provider_response(was_truncated=False)

    mock_provider = AsyncMock()
    mock_provider.complete.side_effect = [truncated, truncated, complete]
    proxy._selector._providers["groq_cloud"] = mock_provider
    proxy._selector._providers["ollama_local"] = AsyncMock()

    result = await proxy.route(
        prompt="Explain decorators",
        classification=_classification(tier=ModelTier.SMALL, max_tokens=80),
    )

    assert result.was_upgraded is True


# ------------------------------------------------------------------ #
# Proxy — all retries exhausted raises                                 #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_proxy_raises_after_all_retries_fail():
    config = _make_config()
    proxy = Proxy(config)

    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("error", request=request, response=response)

    for name in ("groq_cloud", "ollama_local"):
        m = AsyncMock()
        m.complete.side_effect = error
        proxy._selector._providers[name] = m

    with pytest.raises(httpx.HTTPStatusError):
        await proxy.route(
            prompt="What is Python?",
            classification=_classification(),
        )


# ------------------------------------------------------------------ #
# Proxy — force overrides                                              #
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_proxy_respects_max_tokens_override():
    config = _make_config()
    proxy = Proxy(config)

    mock_provider = AsyncMock()
    mock_provider.complete.return_value = _provider_response()
    proxy._selector._providers["groq_cloud"] = mock_provider

    await proxy.route(
        prompt="What is Python?",
        classification=_classification(max_tokens=80),
        max_tokens_override=500,
    )

    call_kwargs = mock_provider.complete.call_args[1]
    assert call_kwargs["max_tokens"] == 500


@pytest.mark.asyncio
async def test_proxy_response_time_is_positive():
    config = _make_config()
    proxy = Proxy(config)

    mock_provider = AsyncMock()
    mock_provider.complete.return_value = _provider_response()
    proxy._selector._providers["groq_cloud"] = mock_provider

    result = await proxy.route(
        prompt="What is Python?",
        classification=_classification(),
    )

    assert result.response_time_ms > 0