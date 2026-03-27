from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ProviderResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    was_truncated: bool


class BaseProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
    )-> ProviderResponse: ...

    @abstractmethod
    async def is_available(self) -> bool: ...



    