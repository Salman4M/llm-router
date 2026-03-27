import httpx

from providers.base import BaseProvider, ProviderResponse


_COMPLETION_URL = "https://api.anthropic.com/v1/messages"
_MODELS_URL = "https://api.anthropic.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"
_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

class AnthropicProvider(BaseProvider):
    def __init__(self,api_key:str)-> None:
        self._api_key = api_key

    def _headers(self)->dict[str,str]:
        return {
            "x-api-key":self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json"
        }
    
    async def complete(
        self, 
        prompt:str, 
        model:str, 
        max_tokens:int
        )->ProviderResponse:
        payload = {
            "model":model,
            "max_tokens":max_tokens,
            "messages":[{"role":"user","content":prompt}]
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                _COMPLETION_URL,
                headers=self._headers(),
                json=payload
            )
            r.raise_for_status()

        data = r.json()
        content_blocks: list[dict] = data.get("content",[])
        content = "".join(
            block.get("text","")
            for block in content_blocks
            if block.get("type") == "text"
        )

        stop_reason: str = data.get("stop_reason","end_turn")

        usage = data.get("usage",{})
        input_tokens: int = usage.get("input_tokens", 0)
        output_tokens: int = usage.get("output_tokens",0)

        return ProviderResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            was_truncated=stop_reason == "max_tokens"
        )
    
    async def is_available(self) -> bool:
        #no inference cost
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                r = await client.get(
                    _MODELS_URL,
                    headers=self._headers()
                )
                return r.status_code == 200
        except httpx.HTTPError:
            return False