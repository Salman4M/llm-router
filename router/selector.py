from core.config import AppConfig, ProviderConfig
from core.classifier import Classification
from providers.base import BaseProvider
from providers.ollama import OllamaProvider
from providers.groq import GroqProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider

def _build_provider(cfg: ProviderConfig) -> BaseProvider:
    match cfg.type:
        case "ollama":
            if not cfg.base_url:
                raise ValueError(f"Provider '{cfg.name}': base_url is required for ollama")
            return OllamaProvider(base_url=cfg.base_url)
        case "groq":
            if not cfg.base_url or not cfg.api_key:
                raise ValueError(f"Provider '{cfg.name}'base_url and api_key are required for groq")
            return GroqProvider(base_url=cfg.base_url, api_key=cfg.api_key)    
        case "openai":
            if not cfg.base_url or not cfg.api_key:
                raise ValueError(f"Provider '{cfg.name}'base_url and api_key are required for openai")
            return OpenAIProvider(base_url=cfg.base_url, api_key=cfg.api_key) 
        case "anthropic":
            if not cfg.api_key:
                raise ValueError(f"Provider '{cfg.name}': api_key is required for anthropic")
            return AnthropicProvider(api_key=cfg.api_key) 
        case _:
            raise ValueError(f"Provider '{cfg.name}': unknown type '{cfg.type}'")


class ModelSelector:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._providers: dict[str, BaseProvider] = {
            name: _build_provider(cfg)
            for name,cfg in config.providers.items()
        }

    def select(
        self,
        classification: Classification,
        force_model: str | None = None,
        force_provider: str | None = None
    )-> tuple[BaseProvider, str,str]:
        
        tier = force_model if force_model else classification.model_tier.value
        provider_name = force_provider if force_provider else self._config.routing.default_provider

        provider_cfg = self._config.provider(provider_name)
        model_name = provider_cfg.models.get(tier)
        provider = self._providers[provider_name]

        return provider, model_name, provider_name
    
    def fallback(
        self,
        classification: Classification,
        force_model: str | None = None,
    )-> tuple[BaseProvider,str,str]:
        tier = force_model if force_model else classification.model_tier.value
        provider_name = self._config.routing.fallback_provider

        provider_cfg = self._config.provider(provider_name)
        model_name = provider_cfg.models.get(tier)
        provider = self._providers[provider_name]

        return provider, model_name, provider_name
    
    def upgrade(
        self,
        classification: Classification,
        current_provider_name:str,
    )->tuple[BaseProvider, str, str] | None:
        current_tier = classification.model_tier.value
        next_tier = {"small":"medium","medium":"large"}.get(current_tier)

        if next_tier is None:
            return None
        
        provider_cfg = self._config.provider(current_provider_name)
        model_name = provider_cfg.models.get(next_tier)
        provider = self._providers[current_provider_name]

        return provider, model_name, current_provider_name
