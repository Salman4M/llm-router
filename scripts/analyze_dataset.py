from __future__ import annotations

import json, argparse, sys, tiktoken
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles
from datasets import load_dataset

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from core.analyzer import analyze
from core.classifier import classify

#dataset config

DATASETS: dict[str, dict[str, str]] = {
    "lmsys": {
        "name":"lmsys/lmsys-chat-1m",
        "split":"train",
        "conversation_key":"conversation",
    },
    "sharegpt": {
        "name":"anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split":"train",
        "conversation_key":"conversations"
    }
}

#tokenizer

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text:str) -> int:
    return len(_TOKENIZER.encode(text))


#conversation parsing

def _iter_lmsys(
        dataset_iter: Iterator[dict[str, object]],
) -> Iterator[tuple[str,str]]:
    for row in dataset_iter:
        turns = row.get("conversation",[])
        if not isinstance(turns,list) or len(turns) < 2:
            continue
        prompt = turns[0].get("content","") if isinstance(turns[0],dict) else ""
        response = turns[1].get("content","") if isinstance(turns[1],dict) else ""
        if prompt and response:
            yield str(prompt), str(response)


def _iter_sharegpt(
        dataset_iter: Iterator[dict[str, object]],
) -> Iterator[tuple[str,str]]:
    for row in dataset_iter:
        turns = row.get("conversations",[])
        if not isinstance(turns,list) or len(turns) < 2:
            continue
        prompt = turns[0].get("value","") if isinstance(turns[0],dict) else ""
        response = turns[1].get("value","") if isinstance(turns[1],dict) else ""
        if prompt and response:
            yield str(prompt), str(response)


def _iter_conversations(
        dataset_name:str,
        dataset_iter: Iterator[dict[str, object]],
)-> Iterator[tuple[str,str]]:
    if dataset_name == "lmsys":
        return _iter_lmsys(dataset_iter)
    return _iter_sharegpt(dataset_iter)
