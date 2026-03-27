import httpx

from providers.base import BaseProvider, ProviderResponse

_COMPLETION_PATH = "/api/chat"
_HEALTH_PATH = "/api/tags"

_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

class OllamaProvider(BaseProvider):
    def __init__(self,base_url: str)->None:
        self.base_url = base_url.rstrip("/")

    async def complete(
        self,
        prompt:str,
        model:str,
        max_tokens:int
        ) -> ProviderResponse:
        payload = {
            "model":model,
            "messages":[{"role":"user", "content":prompt}],
            "stream":False,
            "options": {
                "num_predict":max_tokens
            }
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}{_COMPLETION_PATH}",
                json=payload
            )
            response.raise_for_status()

        data = response.json()
        message = data["message"]
        content: str = message.get("content","")

        prompt_eval_count: int = data.get("prompt_eval_count",0)
        eval_count:int = data.get("eval_count",0)
        #ollama sets done_reason="length" when num_predict is exhausted
        was_truncated = data.get("done_reason") == "length"

        return ProviderResponse(
            content=content,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            model=model,
            was_truncated=was_truncated
        )