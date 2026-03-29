import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core.analyzer import analyze
from core.classifier import classify
from core.config import AppConfig
from core.recorder import Recorder
from router.proxy import Proxy

router = APIRouter()

_VALID_TIERS = {"small","medium","large"}



class RouteRequest(BaseModel):
    prompt: str
    force_model:str | None = Field(default=None, pattern="^(small|medium|large)$")
    force_provider:str | None = None
    max_tokens: int | None = Field(default=None, gt=0)


class RouteResponse(BaseModel):
    response:str
    model_used:str
    provider_used:str
    task_type:str
    routing_confidence:float
    max_tokens_set:int
    actual_input_tokens:int
    actual_output_tokens:int
    routing_time_ms:float
    was_fallback:bool
    fallback_reason:str | None


class StatsResponse(BaseModel):
    total_requests:int
    routing_accuracy:float
    model_distribution:dict[str,float]
    provider_distribution:dict[str,float]
    avg_tokens_saved_vs_always_large: float
    fallback_rate:float
    upgrade_rate:float
    most_misclassified_type: str | None
    avg_response_time_ms:dict[str,float]


#dependency helpers
def _get_config(request:Request) -> AppConfig:
    return request.app.state.config

def _get_proxy(request:Request) -> Proxy:
    return request.app.state.proxy


def _get_recorder(request:Request) -> Recorder:
    return request.app.state.recorder
    

@router.post("/route",response_model=RouteResponse)
async def route(
        body:RouteRequest,
        config: AppConfig = Depends(_get_config),
        proxy: Proxy = Depends(_get_proxy),
        recorder: Recorder = Depends(_get_recorder)
    )->RouteResponse:
    if body.force_provider and body.force_provider not in config.providers:
        raise HTTPException(
                status_code=400,
                detail=f"Unknown provider '{body.force_provider}'."
                       f"Available: {sorted(config.providers.keys())}"
        )
    
    routing_start = time.monotonic()
    signals = analyze(body.prompt)
    classification = classify(signals, raw_prompt=body.prompt)
    routing_time_ms = round((time.monotonic() - routing_start) * 1000, 3)

    try:
        result = await proxy.route(
            prompt=body.prompt,
            classification=classification,
            force_model=body.force_model,
            force_provider=body.force_provider,
            max_tokens_override=body.max_tokens
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=503, detail="all_providers_unavailable") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="all_providers_unavailable") from exc
    
    await recorder.record(
        prompt=body.prompt,
        classification=classification,
        result=result
    )

    return RouteResponse(
        response=result.response.content,
        model_used=result.response.model,
        provider_used=result.provider_name,
        task_type=classification.task_type,
        routing_confidence=round(classification.confidence, 4),
        max_tokens_set=result.response.output_tokens or classification.max_tokens,
        actual_input_tokens=result.response.input_tokens,
        actual_output_tokens=result.response.output_tokens,
        routing_time_ms=routing_time_ms,
        was_fallback=result.was_fallback,
        fallback_reason=result.fallback_reason
            
    )


#stats
@router.get("/stats",response_model=StatsResponse)
async def stats(
    recorder:Recorder = Depends(_get_recorder)
)->StatsResponse:
    data=await recorder.stats()
    return StatsResponse(**data)
    
