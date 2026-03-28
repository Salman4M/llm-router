from dataclasses import dataclass
from enum import StrEnum

import httpx

from core.classifier import Classification, ModelTier, TaskType
from providers.base import BaseProvider

class FallbackReason(StrEnum):
    LOW_CONFIDENCE_SHORT = "low_confidence_short"
    LOW_CONFIDENCE_LONG  = "low_confidence_long"
    MULTILINGUAL         = "multilingual_fallback"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RATE_LIMITED         = "rate_limited"
    INCOMPLETE_RESPONSE  = "incomplete_response"
    TIER_UPGRADED        = "tier_upgraded"


@dataclass
class FallbackDecision:
    use_fallback_provider: bool
    upgraded_max_tokens: int | None
    upgraded_tier: bool
    reason: FallbackReason
    log_event: str


def needs_preemptive_fallback(classification:Classification)  -> FallbackReason | None:
    #failure mode 5 - multilingual
    if classification.task_type == TaskType.UNKNOWN_SHORT and classification.max_tokens == 400:
        return FallbackDecision(
            use_fallback_provider=False,
            upgraded_max_tokens=None,
            upgraded_tier=False,
            reason=FallbackReason.MULTILINGUAL,
            log_event = "multilingual_fallback"
        )
    #failure mode 1 - low confidence
    if classification.confidence < 0.5:
        if classification.task_type == TaskType.UNKNOWN_SHORT:
            return FallbackDecision(
                use_fallback_provider=False,
                upgraded_max_tokens=None,
                upgraded_tier=False,
                reason=FallbackReason.LOW_CONFIDENCE_SHORT,
                log_event = "low_confidence_long"
            )
        return FallbackDecision(
            use_fallback_provider=False,
            upgraded_max_tokens=None,
            upgraded_tier=False,
            reason=FallbackReason.LOW_CONFIDENCE_LONG,
            log_event="low_confidence_long"
        )
    return None



def on_provider_error(exc:httpx.HTTPStatusError) -> FallbackDecision:
    #failure mode 2 (provider unavailable) and mode 4 (rate limited)
    if exc.response.status_code == 429:
        return FallbackDecision(
            use_fallback_provider=True,
            upgraded_max_tokens=None,
            upgraded_tier=False,
            reason=FallbackReason.RATE_LIMITED,
            log_event="rate_limit_fallback"
        )
    return FallbackDecision(
        use_fallback_provider=True,
        upgraded_max_tokens=None,
        upgraded_tier=False,
        reason=FallbackReason.PROVIDER_UNAVAILABLE,
        log_event="provider_unavailable_fallback"
    )


def on_network_error() -> FallbackDecision:
    #failure mode 2 - network/connection error (no http response at all)
    return FallbackDecision(
        use_fallback_provider=True,
        upgraded_max_tokens=None,
        upgraded_tier=False,
        reason=FallbackReason.PROVIDER_UNAVAILABLE,
        log_event="network_error_fallback"
    )


def on_incomplete_response(
        current_max_tokens: int,
        already_retried_tokens: bool,
)-> FallbackDecision:
    #Failure mode 3 - response was truncated
    if not already_retried_tokens:
        return FallbackDecision(
            use_fallback_provider=False,
            upgraded_max_tokens=int(current_max_tokens * 1.5),
            upgraded_tier=False,
            reason= FallbackReason.INCOMPLETE_RESPONSE,
            log_event="incomplete_response_token_retry"
        )
    return FallbackDecision(
        use_fallback_provider=False,
        upgraded_max_tokens=None,
        upgraded_tier=True,
        reason=FallbackReason.TIER_UPGRADED,
        log_event="incomplete_response_tier_upgrade"
    )