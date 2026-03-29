import pytest

from core.analyzer import analyze
from core.classifier import (
    Classification,
    ModelTier,
    TaskType,
    classify,
)


# Helpers                                                              

def c(prompt: str) -> Classification:
    return classify(analyze(prompt), raw_prompt=prompt)


# Tier 1 — deflator / binary / confirmation → small + floor cap       

def test_binary_question_tail_overrides_long_prompt():
    result = c(
        "I wrote a 200-word explanation with lots of context and nuance. Is this correct?"
    )
    assert result.max_tokens == 80
    assert result.model_tier == ModelTier.SMALL
    assert result.task_type == TaskType.FACTUAL


def test_deflator_sets_floor_cap():
    result = c("Explain the history of computing. Briefly.")
    assert result.max_tokens == 80
    assert result.model_tier == ModelTier.SMALL


def test_confirmation_sets_floor_cap():
    result = c("Here is my quicksort. Does this make sense?")
    assert result.max_tokens == 80
    assert result.model_tier == ModelTier.SMALL


# Tier 1 — inflator → large + ceiling cap                             

def test_inflator_sets_ceiling_cap():
    result = c("Explain Python decorators step by step.")
    assert result.max_tokens == 900
    assert result.model_tier == ModelTier.LARGE


def test_inflator_overrides_factual_classification():
    result = c("What is Python? Explain everything in detail.")
    assert result.max_tokens == 900
    assert result.model_tier == ModelTier.LARGE


# Tier 2 — multilingual → medium + 400                                

def test_multilingual_routes_to_medium():
    result = c("Azərbaycan dilini izah et")
    assert result.model_tier == ModelTier.MEDIUM
    assert result.max_tokens == 400


def test_multilingual_overrides_code_request():
    result = c("implement bir funksiya yaz")
    assert result.model_tier == ModelTier.MEDIUM
    assert result.max_tokens == 400


# Low confidence fallback                                              

def test_low_confidence_short_prompt():
    # Gibberish short prompt — no signals should match → low confidence
    result = c("blorf zibble")
    assert result.model_tier == ModelTier.MEDIUM
    assert result.max_tokens == 300
    assert result.task_type == TaskType.UNKNOWN_SHORT
    assert result.confidence < 0.5


def test_low_confidence_long_prompt():
    result = c("blorf zibble " + "word " * 15)
    assert result.model_tier == ModelTier.LARGE
    assert result.max_tokens == 900
    assert result.task_type == TaskType.UNKNOWN_LONG
    assert result.confidence < 0.5


# Tier 3 — normal classification                                       

def test_factual_short():
    result = c("What is Python?")
    assert result.task_type == TaskType.FACTUAL
    assert result.model_tier == ModelTier.SMALL
    assert result.max_tokens <= 900


def test_list_request():
    result = c("List all planets in the solar system")
    assert result.task_type == TaskType.LIST
    assert result.model_tier == ModelTier.SMALL


def test_explanation_request():
    result = c("Explain how TCP/IP works")
    assert result.task_type == TaskType.EXPLANATION
    assert result.model_tier == ModelTier.MEDIUM


def test_code_small():
    result = c("Write a binary search function in Python")
    assert result.task_type == TaskType.CODE_SMALL
    assert result.model_tier == ModelTier.MEDIUM


def test_code_large_by_keyword():
    result = c("Build a REST API service with authentication and database schema")
    assert result.task_type == TaskType.CODE_LARGE
    assert result.model_tier == ModelTier.LARGE


def test_code_large_by_scope():
    result = c("Implement a class that manages a connection pool with retry logic and logging")
    assert result.task_type == TaskType.CODE_LARGE


def test_reasoning_long_prompt():
    prompt = "Consider the trade-offs between " + "various architectural patterns " * 5
    result = c(prompt)
    assert result.task_type == TaskType.REASONING
    assert result.model_tier == ModelTier.LARGE


# max_tokens formula                                                   

def test_max_tokens_uses_last_sentence_word_count():
    # Two prompts with same task type but different last sentence lengths
    short_tail = c("Explain how caching works. Why?")
    long_tail  = c("Explain how caching works. Why does it matter in distributed systems at scale?")
    assert long_tail.max_tokens > short_tail.max_tokens


def test_max_tokens_capped_at_900():
    # Inflator path already tested; ensure normal path never exceeds 900
    prompt = "Explain " + "everything in great depth " * 30
    result = c(prompt)
    assert result.max_tokens <= 900


# Confidence scoring                                                   

def test_high_confidence_factual():
    result = c("What is the capital of France?")
    assert result.confidence >= 0.5


def test_confidence_range():
    result = c("What is Python?")
    assert 0.0 <= result.confidence <= 1.0


def test_signals_matched_leq_signals_checked():
    result = c("Write a sorting algorithm")
    assert result.signals_matched <= result.signals_checked


# caller-visible max_tokens override (classify without override)       

def test_classify_returns_classification_type():
    result = c("What is Python?")
    assert isinstance(result, Classification)


def test_task_type_is_str():
    result = c("What is Python?")
    assert isinstance(result.task_type, str)


# Edge cases                                                           

def test_empty_prompt():
    result = c("")
    assert isinstance(result, Classification)
    assert result.max_tokens > 0


def test_single_word_prompt():
    result = c("Python")
    assert isinstance(result, Classification)


def test_code_block_with_no_request_words():
    result = c("```python\ndef foo(): pass\n```")
    assert result.task_type in (TaskType.CODE_SMALL, TaskType.CODE_LARGE)