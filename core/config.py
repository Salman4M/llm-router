import os, yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelMap:
    small: str
    medium: str
    large: str

    def get(self,tier:str) ->str:
        return getattr(self, tier)


@dataclass
class ProviderConfig:
    name:str
    type:str
    base_url:str | None
    api_key: str | None
    models: ModelMap


@dataclass
class RoutingConfig:
    default_provider: str
    fallback_provider: str
    fallback_model: str


@dataclass
class Thresholds:
    low_confidence: float
    misclassification_ratio: float
    overprovisioned_ratio: float

@dataclass
class TaskCapConfig:
    base_cap: int
    floor: int


@dataclass
class AppConfig:
    providers: dict[str, ProviderConfig]
    routing: RoutingConfig
    thresholds: Thresholds
    task_caps: dict[str, TaskCapConfig]

    def provider(self,name: str) -> ProviderConfig:
        if name not in self.providers:
            raise KeyError(f"Provider '{name}' not found in config")
        return self.providers[name]
    
    def default_provider(self) -> ProviderConfig:
        return self.provider(self.routing.default_provider)
    
    def fallback_provider(self) -> ProviderConfig:
        return self.provider(self.routing.fallback_provider)
    
    def task_cap(self,task_type: str) -> TaskCapConfig:
        if task_type not in self.task_caps:
            raise KeyError(f"Task type '{task_type}' not found in task_caps")
        return self.task_caps[task_type]


#Parsing
_VALID_PROVIDER_TYPES = frozenset({"ollama","groq","openai","anthropic"})
_VALID_TIERS = frozenset({"small","medium","large"})

def _parse_provider(name:str, raw:dict) -> ProviderConfig:
    provider_type = raw.get("type")
    if provider_type not in _VALID_PROVIDER_TYPES:
        raise ValueError(
            f"Provider '{name}': invalid type '{provider_type}'"
            f"Must be one of: {sorted(_VALID_PROVIDER_TYPES)}"
        )
    models_raw = raw.get("models",{})
    missing_tiers = _VALID_TIERS - set(models_raw.keys())
    if missing_tiers:
        raise ValueError(
            f"Provider '{name}': missing model tiers: {sorted(missing_tiers)}"
        )
    
    api_key: str | None = None
    api_key_env = raw.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Provider '{name}': env var '{api_key_env}' is not set"
            )
        
    return ProviderConfig(
        name=name,
        type=provider_type,
        base_url=raw.get("base_url"),
        api_key=api_key,
        models=ModelMap(
            small=models_raw["small"],
            medium=models_raw["medium"],
            large=models_raw["large"]
        )
    )


def _parse_routing(raw: dict, provider_names: set[str]) -> RoutingConfig:
    default = raw.get("default_provider")
    fallback = raw.get("fallback_provider")
    fallback_model = raw.get("fallback_model","medium")

    if not default:
        raise ValueError("routing.default_provider is required")
    if not fallback:
        raise ValueError("routing.fallback_provider is required")
    if default not in provider_names:
        raise ValueError(f"routing.default_provider '{default}' is not a defined provider")
    if fallback not in provider_names:
        raise ValueError(f"routing.fallback_provider '{fallback}' is not a defined provider")
    if fallback_model not in _VALID_TIERS:
        raise ValueError(f"routing.fallback_odel '{fallback_model}' must be one of: {sorted(_VALID_TIERS)}")
    
    return RoutingConfig(
        default_provider=default,
        fallback_provider=fallback,
        fallback_model=fallback_model
    )


def _parse_thresholds(raw:dict) -> Thresholds:
    low_confidence = float(raw.get("low_confidence",0.5))
    misclassification_ratio = float(raw.get("misclassification_ratio", 2.0))
    overprovisioned_ratio = float(raw.get("overprovisioned_ratio",0.3))

    if not (0.0 < low_confidence < 1.0):
        raise ValueError(f"thresholds.low_confidence must be between 0 and 1, got {low_confidence}")
    if misclassification_ratio <=1.0:
        raise ValueError(f"thresholds.misclassification_ratio must be > 1.0, got {misclassification_ratio}")
    
    if not (0.0 < overprovisioned_ratio < 1.0):
        raise ValueError(f"thresholds.overprovisioned_ratio must be between 0 and 1, got {overprovisioned_ratio}")
    

    return Thresholds(
        low_confidence=low_confidence,
        misclassification_ratio=misclassification_ratio,
        overprovisioned_ratio=overprovisioned_ratio
    )


def _parse_task_caps(raw: dict) -> dict[str,TaskCapConfig]:
    caps: dict[str, TaskCapConfig] = {}
    for task_type, values in raw.items():
        base_cap = int(values.get("base_cap",0))
        floor = int(values.get("floor",0))
        if base_cap <=0:
            raise ValueError(f"task_caps.{task_type}.base_cap must be > 0")
        if floor < 0:
            raise ValueError(f"task_caps.{task_type}.floor must be >= 0")
        if floor >= base_cap:
            raise ValueError(
                f"task_caps.{task_type}: floor ({floor}) must be less than base_cap ({base_cap})"
            )
        caps[task_type]  = TaskCapConfig(base_cap=base_cap,floor=floor)

    return caps


#public loader

def load_config(path:Path | str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open() as f:
        raw = yaml.safe_load(f)
    
    if not isinstance(raw,dict):
        raise ValueError("config.yaml must be a YAML mapping at the top level")
    
    providers_raw = raw.get("providers",{})
    if not providers_raw:
        raise ValueError("config.yaml: 'providers' section is required and must be not empty")
    
    providers: dict[str,ProviderConfig] = {
        name: _parse_provider(name,cfg)
        for name, cfg in providers_raw.items()
    }

    routing = _parse_routing(raw.get("routing",{}), set(providers.keys()))
    thresholds = _parse_thresholds(raw.get("thresholds",{}))
    task_caps = _parse_task_caps(raw.get("task_caps", {}))

    return AppConfig(
        providers=providers,
        routing=routing,
        thresholds=thresholds,
        task_caps=task_caps
    )