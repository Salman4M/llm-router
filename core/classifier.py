from dataclasses import dataclass
from enum import StrEnum

from core.analyzer import Signals

class TaskType(StrEnum):
    FACTUAL = "factual"
    LIST = "list"
    CODE_SMALL = "code_small"
    CODE_LARGE = "code_large"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    UNKNOWN_SHORT = "unknown_short"
    UNKNOWN_LONG = "unknown_long"


class ModelTier(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_CODE_LARGE_KEYWORDS = frozenset({
    "class", "api", "system", " service", "server", "client", "framework",
    "architecture", "pipeline", "module", "library", "package", "database",
    "schema", "endpoint", "microservices", "cli", "daemon", "worker"
})


_TASK_CAPS:dict[TaskType, int] = {
    TaskType.FACTUAL:       80,
    TaskType.LIST:          250,
    TaskType.CODE_SMALL:    350,
    TaskType.CODE_LARGE:    800,
    TaskType.EXPLANATION:   500,
    TaskType.REASONING:     900,
    TaskType.UNKNOWN_SHORT: 300,
    TaskType.UNKNOWN_LONG:  900
}


_TASK_TIERS: dict[TaskType, ModelTier] = {
    TaskType.FACTUAL:       ModelTier.SMALL,
    TaskType.LIST:          ModelTier.SMALL,
    TaskType.CODE_SMALL:    ModelTier.MEDIUM,
    TaskType.CODE_LARGE:    ModelTier.LARGE,
    TaskType.EXPLANATION:   ModelTier.MEDIUM,
    TaskType.REASONING:     ModelTier.LARGE,
    TaskType.UNKNOWN_SHORT: ModelTier.MEDIUM,
    TaskType.UNKNOWN_LONG:  ModelTier.LARGE 
}


_TASK_MULTIPLIERS: dict[TaskType, float] = {
    TaskType.FACTUAL:       1.0,
    TaskType.LIST:          1.5,
    TaskType.CODE_SMALL:    2.0,
    TaskType.CODE_LARGE:    3.0,
    TaskType.EXPLANATION:   2.5,
    TaskType.REASONING:     3.0,
    TaskType.UNKNOWN_SHORT: 1.5,
    TaskType.UNKNOWN_LONG:  3.0 
}


_DEFLATOR_CAP = 80
_INFLATOR_CAP = 900
_MULTILINGUAL_CAP = 400

_LOW_CONFIDENCE_SHORT_CAP = 300
_LOW_CONFIDENCE_LONG_CAP = 900
_LOW_CONFIDENCE_THRESHOLD = 0.5
_LOW_CONFIDENCE_SHORT_WORLD_LIMIT = 10

@dataclass
class Classification:
    task_type: TaskType
    model_tier: ModelTier
    max_tokens: int
    confidence: float
    signals_matched: int
    signals_checked: int


def _count_signals(signals: Signals) -> tuple[int,int]:
    checks = [
        signals.is_factual,
        signals.is_list_request,
        signals.is_explanation_request,
        signals.is_code_request,
        signals.has_code_block,
        signals.has_code_keywords,
        signals.has_question
    ]
    matched = sum(1 for c in checks if c)
    return matched,len(checks)


def _has_code_large_scope(signals:Signals)-> bool:
    return signals.has_code_keywords and signals.word_count >=20



def _infer_task_type(signals: Signals, raw_prompt:str) ->TaskType:
    if signals.is_code_request or signals.has_code_block:
        prompt_lower = raw_prompt.lower()
        has_large_scope = any(kw in prompt_lower for kw in _CODE_LARGE_KEYWORDS) or has_large_scope(signals)
        
        return TaskType.CODE_LARGE if has_large_scope else TaskType.CODE_SMALL
    
    if signals.is_factual and signals.word_count < 15:
        return TaskType.FACTUAL
    
    if signals.is_list_request:
        return TaskType.LIST
    
    if signals.is_explanation_request:
        return TaskType.EXPLANATION

    if signals.word_count >= 30:
        return TaskType.REASONING
    
    if signals.word_count <10:
        return TaskType.UNKNOWN_SHORT
    
    return TaskType.UNKNOWN_LONG



def classify(signals: Signals, raw_prompt:str="")->Classification:
    signals_matched, signals_checked = _count_signals(signals)
    confidence = signals_matched / signals_checked if signals_checked else 0.0

    #tier 1 - output overrides (last 1-2 sentences, short-circiut cap)
    if signals.is_binary_question or signals.has_deflator or signals.is_confirmation:
        return Classification(
            task_type=TaskType.FACTUAL,
            model_tier=ModelTier.SMALL,
            max_tokens=_DEFLATOR_CAP,
            confidence=confidence,
            signals_matched=signals_matched,
            signals_checked=signals_checked
        )
    if signals.has_inflator:
        task_type = _infer_task_type(signals, raw_prompt)
        return Classification(
            task_type=task_type,
            model_tier=ModelTier.LARGE,
            max_tokens=_INFLATOR_CAP,
            confidence=confidence,
            signals_matched=signals_matched,
            signals_checked=signals_checked
        )

    #tier 2 - routing overrides
    if signals.has_multilingual:
        return Classification(
            task_type=TaskType.UNKNOWN_SHORT,
            model_tier=ModelTier.MEDIUM,
            max_tokens=_MULTILINGUAL_CAP,
            confidence=confidence,
            signals_matched=signals_matched,
            signals_checked=signals_checked
        )

    #tier 1 (confidence gate) - low confidence fallback
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        if signals.word_count < _LOW_CONFIDENCE_SHORT_WORLD_LIMIT:
            return Classification(
                task_type=TaskType.UNKNOWN_SHORT,
                model_tier=ModelTier.MEDIUM,
                max_tokens=_LOW_CONFIDENCE_SHORT_CAP,
                confidence=confidence,
                signals_matched=signals_matched,
                signals_checked=signals_checked
            )
        return Classification(
                task_type=TaskType.UNKNOWN_LONG,
                model_tier=ModelTier.LARGE,
                max_tokens=_LOW_CONFIDENCE_LONG_CAP,
                confidence=confidence,
                signals_matched=signals_matched,
                signals_checked=signals_checked
        )
    
    #tier 3 - normal classification
    task_type = _infer_task_type(signals,raw_prompt)
    base_cap = _TASK_CAPS[task_type]
    multiplier = _TASK_MULTIPLIERS[task_type]
    max_tokens = int(base_cap + signals.last_sentence_word_count * multiplier)
    max_tokens = min(max_tokens, _INFLATOR_CAP)

    return Classification(
        task_type=task_type,
        model_tier=_TASK_TIERS[task_type],
        max_tokens=max_tokens,
        confidence=confidence,
        signals_matched=signals_matched,
        signals_checked=signals_checked
    )
