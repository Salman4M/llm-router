import time
from dataclasses import dataclass

import httpx

from core.classifier import Classification
from core.config import AppConfig
from providers.base import BaseProvider, ProviderResponse
from router.fallback import (
    FallbackDecision,
    FallbackReason,
    needs_preemptive_fallback,
    on_incomplete_response,
    on_network_error,
    on_provider_error
)

from router.selector import ModelSelector


@dataclass
class ProxyResult:
    response:ProviderResponse
    provider_name: str
    was_fallback:bool
    fallback_reason: FallbackReason | None
    was_upgraded: bool
    response_time_ms: float


_MAX_RETRIES = 2

class Proxy:
    def __init__(self,config: AppConfig)-> None:
        self._selector =  ModelSelector(config)

    async def route(
        self,
        prompt:str,
        classification: Classification,
        force_model: str | None = None,
        force_provider: str | None = None,
        max_tokens_override: int | None = None,
    )-> ProxyResult:
        max_tokens = max_tokens_override if max_tokens_override else classification.max_tokens
        was_fallback = False
        was_upgraded = False
        fallback_reason: FallbackReason | None = None
        token_retry_done = False

        #failure mode 1 and 5 (preemptive fallback check)
        preemptive = needs_preemptive_fallback(classification)
        if preemptive:
            was_fallback = True
            fallback_reason = preemptive.reason
        
        #selection of primary provider and model
        provider, model, provider_name = self._selector.select(
            classification,
            force_model=force_model,
            force_provider=force_provider
        )

        start = time.monotonic()

        for attempt in range(_MAX_RETRIES + 1):
            try: 
                response = await provider.complete(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens
                )
            except httpx.HTTPStatusError as exc:
                if attempt >= _MAX_RETRIES:
                    raise

                decision = on_provider_error(exc)
                provider,model, provider_name,was_fallback, fallback_reason = (
                    self._apply_provider_fallback(
                        decision, classification, force_model, was_fallback
                    )
                )
                continue
            except httpx.HTTPError:
                if attempt >= _MAX_RETRIES:
                    raise

                decision = on_network_error()
                provider, model, provider_name, was_fallback, fallback_reason = (
                    self._apply_provider_fallback(
                        decision, classification, force_model, was_fallback
                    )
                )
                continue

            #failure mode 3 - incomplete response
            if response.was_truncated:
                if attempt >= _MAX_RETRIES:
                    break

                decision = on_incomplete_response(
                    current_max_tokens=max_tokens,
                    already_retried_tokens=token_retry_done
                )

                if decision.upgraded_max_tokens is not None:
                    max_tokens = decision.upgraded_max_tokens
                    token_retry_done = True
                    fallback_reason = decision.reason

                elif decision.upgraded_tier:
                    upgraded = self._selector.upgrade(classification,provider_name)
                    if upgraded is None:
                        break # already at large, nothing to upgrade to
                    provider, model, provider_name = upgraded
                    was_upgraded = True
                    fallback_reason = decision.reason

                continue

            break
        elapsed_ms = (time.monotonic() - start) * 1000

        return ProxyResult(
            response=response,
            provider_name=provider_name,
            was_fallback=was_fallback,
            fallback_reason=fallback_reason,
            was_upgraded=was_upgraded,
            response_time_ms=round(elapsed_ms,2)
        )


    def _apply_provider_fallback(
            self,
            decision:FallbackDecision,
            classification: Classification,
            force_model:str | None,
            was_fallback: bool,
    )-> tuple[BaseProvider,str,str,bool,FallbackReason]:
        provider, model, provider_name = self._selector.fallback(
            classification, force_model=force_model
        )
        return provider, model , provider_name , True, decision.reason

