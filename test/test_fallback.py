import httpx
import pytest

from core.classifier import Classification, ModelTier, TaskType
from router.fallback import (
    FallbackDecision,
    FallbackReason,
    needs_preemptive_fallback,
    on_incomplete_response,
    on_network_error,
    on_provider_error,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _classification(
    task_type: TaskType = TaskType.FACTUAL,
    tier: ModelTier = ModelTier.SMALL,
    max_tokens: int = 80,
    confidence: float = 0.85,
) -> Classification:
    return Classification(
        task_type=task_type,
        model_tier=tier,
        max_tokens=max_tokens,
        confidence=confidence,
        signals_matched=3,
        signals_checked=7,
    )


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


# ------------------------------------------------------------------ #
# needs_preemptive_fallback — failure mode 1 (low confidence)         #
# ------------------------------------------------------------------ #

def test_low_confidence_short_returns_decision():
    c = _classification(
        task_type=TaskType.UNKNOWN_SHORT,
        tier=ModelTier.MEDIUM,
        max_tokens=300,
        confidence=0.3,
    )
    result = needs_preemptive_fallback(c)
    assert result is not None
    assert result.reason == FallbackReason.LOW_CONFIDENCE_SHORT
    assert result.use_fallback_provider is False
    assert result.upgraded_tier is False


def test_low_confidence_long_returns_decision():
    c = _classification(
        task_type=TaskType.UNKNOWN_LONG,
        tier=ModelTier.LARGE,
        max_tokens=900,
        confidence=0.2,
    )
    result = needs_preemptive_fallback(c)
    assert result is not None
    assert result.reason == FallbackReason.LOW_CONFIDENCE_LONG
    assert result.use_fallback_provider is False


def test_high_confidence_returns_none():
    c = _classification(confidence=0.9)
    assert needs_preemptive_fallback(c) is None


def test_confidence_exactly_at_threshold_returns_none():
    # threshold is < 0.5, so 0.5 exactly should not trigger
    c = _classification(confidence=0.5)
    assert needs_preemptive_fallback(c) is None


# ------------------------------------------------------------------ #
# needs_preemptive_fallback — failure mode 5 (multilingual)           #
# ------------------------------------------------------------------ #

def test_multilingual_preemptive_fallback():
    c = _classification(
        task_type=TaskType.UNKNOWN_SHORT,
        tier=ModelTier.MEDIUM,
        max_tokens=400,
        confidence=0.6,
    )
    result = needs_preemptive_fallback(c)
    assert result is not None
    assert result.reason == FallbackReason.MULTILINGUAL
    assert result.use_fallback_provider is False


# ------------------------------------------------------------------ #
# on_provider_error — failure modes 2 and 4                           #
# ------------------------------------------------------------------ #

def test_429_triggers_rate_limit_fallback():
    decision = on_provider_error(_http_error(429))
    assert decision.reason == FallbackReason.RATE_LIMITED
    assert decision.use_fallback_provider is True
    assert decision.upgraded_max_tokens is None
    assert decision.upgraded_tier is False


def test_500_triggers_provider_unavailable_fallback():
    decision = on_provider_error(_http_error(500))
    assert decision.reason == FallbackReason.PROVIDER_UNAVAILABLE
    assert decision.use_fallback_provider is True


def test_503_triggers_provider_unavailable_fallback():
    decision = on_provider_error(_http_error(503))
    assert decision.reason == FallbackReason.PROVIDER_UNAVAILABLE
    assert decision.use_fallback_provider is True


def test_401_triggers_provider_unavailable_fallback():
    decision = on_provider_error(_http_error(401))
    assert decision.reason == FallbackReason.PROVIDER_UNAVAILABLE
    assert decision.use_fallback_provider is True


# ------------------------------------------------------------------ #
# on_network_error — failure mode 2                                    #
# ------------------------------------------------------------------ #

def test_network_error_triggers_provider_fallback():
    decision = on_network_error()
    assert decision.reason == FallbackReason.PROVIDER_UNAVAILABLE
    assert decision.use_fallback_provider is True
    assert decision.upgraded_max_tokens is None
    assert decision.upgraded_tier is False


# ------------------------------------------------------------------ #
# on_incomplete_response — failure mode 3                              #
# ------------------------------------------------------------------ #

def test_first_incomplete_bumps_tokens():
    decision = on_incomplete_response(current_max_tokens=200, already_retried_tokens=False)
    assert decision.upgraded_max_tokens == 300  # 200 * 1.5
    assert decision.upgraded_tier is False
    assert decision.use_fallback_provider is False
    assert decision.reason == FallbackReason.INCOMPLETE_RESPONSE


def test_token_bump_rounds_down():
    decision = on_incomplete_response(current_max_tokens=100, already_retried_tokens=False)
    assert decision.upgraded_max_tokens == 150


def test_second_incomplete_upgrades_tier():
    decision = on_incomplete_response(current_max_tokens=300, already_retried_tokens=True)
    assert decision.upgraded_tier is True
    assert decision.upgraded_max_tokens is None
    assert decision.use_fallback_provider is False
    assert decision.reason == FallbackReason.TIER_UPGRADED


# ------------------------------------------------------------------ #
# FallbackDecision shape                                               #
# ------------------------------------------------------------------ #

def test_fallback_decision_is_dataclass():
    d = on_network_error()
    assert isinstance(d, FallbackDecision)


def test_fallback_reason_is_str():
    d = on_network_error()
    assert isinstance(d.reason, str)