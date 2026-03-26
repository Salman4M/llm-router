import re
from dataclasses import dataclass

@dataclass
class Signals:
    word_count: int
    char_count: int 
    has_code_block: bool
    has_code_keywords: bool
    has_question: bool
    is_factual: bool
    is_list_request: bool
    is_explanation_request: bool
    is_code_request: bool
    sentence_count:int
    has_multilingual: bool
    is_binary_question: bool
    has_deflator: bool
    is_confirmation: bool
    has_inflator: bool
    last_sentence_word_count: int


_CODE_KEYWORDS = frozenset({
    "def","class", "import","function","return","const", "let", "var",
    "if", "else", "for", "while", "async", "await", "lambda", "yield",
    "struct", "interface", "extends", "implements",
})

_FACTUAL_STARTERS = frozenset({
    "what", "who", "when", "where", "which",
})


_LIST_PATTERNS = re.compile(
    r"\b(list|enumarate|give me|examples of|name \d+|top \d+|what are)\b",
    re.IGNORECASE
)

_EXPLANATION_PATTERNS = re.compile(
    r"\b(explain|how does|how do|why does|why do|describe|walk me through|what is|what are)\b",
    re.IGNORECASE
)

_CODE_REQUEST_PATTERNS = re.compile(
    r"\b(write||implement|create|build|generate|make|code|program|script|develop)\b",
    re.IGNORECASE
)

_BINARY_PATTERNS = re.compile(
    r"\b(is this|does this|can this|should i|are these|is it|does it|will this|would this)\b",
    re.IGNORECASE
)

_DEFLATION_PATTERNS = re.compile(
    r"\b(briefly|in one sentence|yes or no|just tell me|short answer|in a word|tldr|tl;dr|summarize in)\b",
    re.IGNORECASE
)

_CONFIRMATION_PATTERNS = re.compile(
    r"\b(is this correct|am i right|does this make sense|is that right|is this right|is this ok|correct\?|right\?|ok\?)\b",
    re.IGNORECASE
)

_INFLATION_PATTERNS = re.compile(
    r"\b(step by step|in detail|explain everything|thoroughly|comprehensively|in depth|exhaustively|"
    r"walk me through every|cover all|don't skip| do not skip)\b"
    ,
    re.IGNORECASE
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text:str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s for s in sentences if s.strip()]


def _tail(sentences:list[str], n: int = 2) -> str:
    return " ".join(sentences[-n:]) if sentences else ""


def analyze(prompt:str) -> Signals:
    words = prompt.split()
    word_count = len(words)
    char_count = len(prompt)

    has_code_block = "```" in prompt

    prompt_lower_words = set(w.lower().strip(".,!;:") for w in words)
    has_code_keywords = bool(prompt_lower_words & _CODE_KEYWORDS)

    stripped = prompt.strip()
    has_question = stripped.endswith("?")

    first_word = words[0].lower().strip(".,!?;:") if words else ""
    is_factual = first_word in _FACTUAL_STARTERS

    is_list_request = bool(_LIST_PATTERNS.search(prompt))
    is_explanation_request = bool(_EXPLANATION_PATTERNS.search(prompt))
    is_code_request = bool(_CODE_REQUEST_PATTERNS.search(prompt))

    has_multilingual = bool(re.search(r"[^\x00-\x7F]",prompt))
    sentences = _split_sentences(prompt)
    sentence_count = len(sentences)
    tail = _tail(sentences, n=2)

    is_binary_question = bool(_BINARY_PATTERNS.search(tail))
    has_deflator = bool(_DEFLATION_PATTERNS.search(tail))
    is_confirmation = bool(_CODE_REQUEST_PATTERNS.search(tail))
    has_inflator = bool(_INFLATION_PATTERNS.search(tail))

    last_sentence = sentences[-1] if sentences else ""
    last_sentence_word_count = len(last_sentence.split())

    return Signals(
    word_count = word_count,
    char_count =  char_count,
    has_code_block = has_code_block,
    has_code_keywords = has_code_keywords,
    has_question = has_question,
    is_factual = is_factual,
    is_list_request = is_list_request,
    is_explanation_request = is_explanation_request,
    is_code_request = is_code_request,
    sentence_count = sentence_count,
    has_multilingual = has_multilingual,
    is_binary_question = is_binary_question,
    has_deflator = has_deflator,
    is_confirmation = is_confirmation,
    has_inflator = has_inflator,
    last_sentence_word_count = last_sentence_word_count,
    )

