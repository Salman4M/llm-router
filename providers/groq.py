import httpx

from providers.base import BaseProvider, ProviderResponse

_COMPLETION_PATH = "/chat/completions"

_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0 , write=10.0 , pool=5.0)

class GroqProvider(BaseProvider):
    def __init__(self,base_url:str, api_key:str)->None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _headers(self) -> dict[str,str]:
        return {
            "Authorization":f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
    
    async def complete(
        self,
        prompt:str, 
        model:str, 
        max_tokens:int,
        )->ProviderResponse:
        payload = {
            "model":model,
            "messages":[{"role":"user","content":prompt}],
            "max_tokens": max_tokens
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                f"{self._base_url}{_COMPLETION_PATH}",
                headers=self._headers(),
                json=payload
            )
            response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]
        content: str = choice["message"]["content"] or ""
        finish_reason: str = choice.get("finish_reason","stop")

        usage = data.get("usage",{})
        input_tokens: int = usage.get("prompt_tokens", 0)
        output_tokens:int = usage.get("completion_tokens", 0)

        return ProviderResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            was_truncated=finish_reason == "length"
        )
    
    async def is_available(self)-> bool:
        #groq doesn't have dedicated health endpoint - so we attempt a minimal completion
        #with max_tokens=1 to verify API key and connectivity
        try:
            await self.complete(prompt="hi",model="llama-3.1-8b-instant",max_tokens=1)
            return True
        except httpx.HTTPStatusError as e:
            #429 means rate-limited but provider is up
            if e.response.status_code == 429:
                return True
            return False
        except httpx.HTTPError:
            return False
    