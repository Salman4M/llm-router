from __future__ import annotations

import json, argparse, sys, tiktoken
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles
from datasets import load_dataset
from typing import Iterator

from core.analyzer import analyze
from core.classifier import classify

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
#dataset config

DATASETS: dict[str, dict[str, str]] = {
    "lmsys": {
        "name":"lmsys/lmsys-chat-1m",
        "split":"train",
        "conversation_key":"conversation",
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


def _iter_sharegpt_file(path:Path) -> Iterator[tuple[str,str]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data,list):
        raise ValueError(f"Excepted a JSON array in {path}, got {type(data).__name__}")

    for record in data:
        turns = record.get("conversations",[])
        if not isinstance(turns, list) or len(turns) < 2:
            continue
        prompt = turns[0].get("value","") if isinstance(turns[0], dict) else ""
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


#analysis

def analyze_dataset(
        dataset_name: str,
        max_conversations: int | None,
        progress_every: int = 10_000,
        sharegpt_file: Path | None = None
)->dict[str,list[int]]:
    if dataset_name == "sharegpt":
        if sharegpt_file is None:
            print(
                "Error: --sharegpt-file is required for the sharegpt dataset.\n"
                "Download it first:\n"
                "  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
                "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json\n"
                "Then pass: --sharegpt-file ShareGPT_V3_unfiltered_cleaned_split.json",
                file=sys.stderr
            )
            sys.exit(1)
        if not sharegpt_file.exists():
            print(f"Error: file not found : {sharegpt_file}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading ShareGPT from local file: {sharegpt_file}")
        conversation_iter: Iterator[tuple[str,str]] = _iter_sharegpt_file(sharegpt_file)
    else:
        config = DATASETS[dataset_name]
        print(f"Loading dataset: {config['name']} (streaming)")
        raw = load_dataset(config["name"],streaming=True, split=config["split"])
        conversation_iter = _iter_conversations(dataset_name,iter(raw))

    results:dict[str,list[int]] = defaultdict(list)
    total = 0
    skipped = 0

    for prompt,response in _iter_conversations(dataset_name,iter(raw)):
        if max_conversations is not None and total >= max_conversations:
            break
        try:
            signals = analyze(prompt)
            classification = classify(signals,raw_prompt=prompt)
            token_count = count_tokens(response)
            results[str(classification.task_type)].append(token_count)
            total+=1
        except Exception:
            skipped +=1
            continue

        if total % progress_every == 0:
            print(f"processed {total:,} Skipped {skipped:,}\n")
    
    print(f"\nDone. Processed: {total:,} Skipped: {skipped:,}\n")
    return dict(results)


#statistics

def compute_stats(results:dict[str,list[int]])->dict[str,dict[str,int]]:
    stats:dict[str,dict[str,int]] = {}
    for task_type, counts in sorted(results.items()):
        if len(counts) < 10:
            #too few samples for reliable quantiles
            continue
        p = quantiles(counts,n=10)
        stats[task_type] = {
            "samples":  len(counts),
            "mean":     round(mean(counts)),
            "p10":      round(p[0]),
            "p50":      round(p[4]),
            "p90":      round(p[8])
        }
    return stats



#output
def print_table(stats:dict[str,dict[str,int]]) -> None:
    col = 18
    header = (
        f"{'task_type':<{col}} {'samples':>8} {'mean':>6} {'p10':>6} {'p50':>6} {'p90':>6}"
    )
    print(header)
    print("-" * len(header))
    for task_type, s in stats.items():
        print(
            f"{task_type:<{col}} {s['samples']:>8,} {s['mean']:>6} "
            f"{s['p10']:6} {s['p50']:>6} {s['p90']:>6}"
        )


def print_yaml_snippet(stats:dict[str,dict[str,int]]) -> None:
    print("\n# --- paste into config.yaml under task_caps: ---")
    print("task_caps")
    for task_type, s in stats.items():
        print(f"{task_type}: {s['p90']}  # p10={s['p10']} samples={s['samples']:,}")
    print("# -----------------------------------------------")


def save_json(stats:dict[str,dict[str,int]],path:str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_tex(json.dumps(stats,indent=2))
    print(f"\nResults saved to: {out}")



#cli

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 dataset analysis - derive base_cap values for config.yaml"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="lmsys",
        help="Dataset to stream {default: lmsys}"
    )
    parser.add_argument(
    "--sharegpt-file",
        type=Path,
        default=None,
        help="Path to local ShareGPT JSON file (required when --dataset sharegpt)"
    )

    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N conversations (defualt: full dataset)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save JSON results to FILE (default: print only)"
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        metavar="N",
        help="Print progress every N conversations (default: 10000)"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = analyze_dataset(
        dataset_name=args.dataset,
        max_conversations=args.max_conversations,
        progress_every=args.progress_every,
        sharegpt_file=args.sharegpt_file
    )

    if not results:
        print("No results - dataset may be empty or inaccessible.")
        sys.exit(1)
    
    stats = compute_stats(results)

    print_table(stats)
    print_yaml_snippet(stats)

    if args.output:
        save_json(stats, args.output)



if __name__ == "__main__":
    main()