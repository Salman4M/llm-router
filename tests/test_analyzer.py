import pytest

from core.analyzer import Signals, analyze


# Helpers                                                              

def a(prompt: str) -> Signals:
    return analyze(prompt)


# Basic signal extraction                                              

def test_word_and_char_count():
    s = a("Hello world")
    assert s.word_count == 2
    assert s.char_count == 11


def test_has_code_block_true():
    assert a("Here is code:\n```python\nprint('hi')\n```").has_code_block is True


def test_has_code_block_false():
    assert a("No code here").has_code_block is False


def test_has_code_keywords_true():
    assert a("def my_function(): pass").has_code_keywords is True


def test_has_code_keywords_false():
    assert a("What is the capital of France?").has_code_keywords is False


def test_has_question_true():
    assert a("What is Python?").has_question is True


def test_has_question_false():
    assert a("Explain Python").has_question is False


def test_is_factual_true():
    for starter in ("What", "Who", "When", "Where", "Which"):
        assert a(f"{starter} is the capital of France?").is_factual is True


def test_is_factual_false():
    assert a("Explain how Python works").is_factual is False


def test_is_list_request_true():
    for prompt in (
        "List all planets in the solar system",
        "Give me examples of sorting algorithms",
        "Enumerate the main Python data structures",
        "Name 5 design patterns",
        "What are the top 10 frameworks?",
    ):
        assert a(prompt).is_list_request is True, f"Expected is_list_request for: {prompt!r}"


def test_is_list_request_false():
    assert a("What is Python?").is_list_request is False


def test_is_explanation_request_true():
    for prompt in (
        "Explain how TCP/IP works",
        "How does garbage collection work?",
        "Why does Python use the GIL?",
        "Describe the OSI model",
    ):
        assert a(prompt).is_explanation_request is True, f"Expected is_explanation_request for: {prompt!r}"


def test_is_explanation_request_false():
    assert a("Write a sorting function").is_explanation_request is False


def test_is_code_request_true():
    for prompt in (
        "Write a binary search function",
        "Implement a stack in Python",
        "Create a REST API",
        "Build a CLI tool",
    ):
        assert a(prompt).is_code_request is True, f"Expected is_code_request for: {prompt!r}"


def test_is_code_request_false():
    assert a("What is recursion?").is_code_request is False


def test_sentence_count_single():
    assert a("What is Python?").sentence_count == 1


def test_sentence_count_multiple():
    s = a("What is Python? It is a language. I like it.")
    assert s.sentence_count == 3


def test_has_multilingual_true():
    assert a("Azərbaycan dili çox gözəldir").has_multilingual is True


def test_has_multilingual_false():
    assert a("Hello world").has_multilingual is False


# Tier 1 signals — operate on last 1-2 sentences only                 

def test_is_binary_question_in_tail():
    prompt = "I wrote this long function with many edge cases. Is this correct?"
    s = a(prompt)
    assert s.is_binary_question is True


def test_is_binary_question_not_in_body():
    # "is this" appears only in the middle, not the tail
    prompt = "Is this a good approach? I want you to write a comprehensive system with full error handling, retry logic, and logging."
    s = a(prompt)
    # tail is the last sentence about error handling — no binary signal there
    assert s.is_binary_question is False


def test_has_deflator_in_tail():
    prompt = "Can you describe the history of computing? Briefly."
    assert a(prompt).has_deflator is True


def test_has_deflator_not_triggered_by_body():
    prompt = "Briefly mention it. Now write a comprehensive guide with step by step details."
    s = a(prompt)
    assert s.has_deflator is False
    assert s.has_inflator is True


def test_is_confirmation_in_tail():
    prompt = "Here is my implementation of quicksort. Does this make sense?"
    assert a(prompt).is_confirmation is True


def test_has_inflator_in_tail():
    prompt = "Explain Python decorators step by step."
    assert a(prompt).has_inflator is True


def test_has_inflator_false():
    assert a("What is Python?").has_inflator is False


# last_sentence_word_count                                             


def test_last_sentence_word_count_single_sentence():
    s = a("What is Python?")
    assert s.last_sentence_word_count == 3


def test_last_sentence_word_count_multi_sentence():
    s = a("Explain Python. What is it used for?")
    assert s.last_sentence_word_count == 5


# Edge cases                                                           

def test_empty_prompt():
    s = a("")
    assert s.word_count == 0
    assert s.char_count == 0
    assert s.has_question is False
    assert s.is_factual is False
    assert s.sentence_count == 0
    assert s.last_sentence_word_count == 0


def test_single_word():
    s = a("Python")
    assert s.word_count == 1
    assert s.is_factual is False
    assert s.has_question is False


def test_non_ascii_only():
    s = a("مرحبا بالعالم")
    assert s.has_multilingual is True
    assert s.word_count == 2


def test_mixed_language():
    s = a("Explain Python. Azərbaycan dilindədir.")
    assert s.has_multilingual is True
    assert s.is_explanation_request is True


def test_prompt_with_only_code_block():
    s = a("```python\ndef foo(): pass\n```")
    assert s.has_code_block is True
    assert s.has_code_keywords is True


def test_very_long_prompt_word_count():
    prompt = " ".join(["word"] * 500)
    s = a(prompt)
    assert s.word_count == 500