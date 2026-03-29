import argparse, asyncio, json, os, sys, yaml
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0,str(Path(__name__).resolve().parent.parent))

from models.request import RequestRecord


#suggestion logic

def _db_url() ->str:
    url = os.getenv("DATABASE_URL","")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://","postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://","postgresql+asyncpg://", 1)
    return url


async def _fetch_misclassified(
    session_factory: async_sessionmaker,
    since: datetime
)-> list[RequestRecord]:
    async with session_factory() as session:
        result = await session.execute(
            select(RequestRecord).where(
                RequestRecord.was_misclassified == True,
                RequestRecord.timestamp >= since
            )
        )
        return list(result.scalars().all())
    

def _find_patterns(
    records: list[RequestRecord],
    min_occurences: int
)->list[dict]:
    """
    Group misclassified records by task_type, count keyword frequency,
    surface keywords that appear >= min_occurrences times.
    """
    #task_type -> keyword -> count
    keyword_counts:dict[str, Counter] = defaultdict(Counter)
    #task_type -> list of avg actual tokens
    token_sums: dict[str, list[int]] = defaultdict(list)

    for record in records:
        try:
            keywords: list[str] = json.loads(record.keywords)
        except(json.JSONDecodeError, TypeError):
            keywords = []

        keyword_counts[record.task_type].update(keywords)
        token_sums[record.task_type].append(record.actual_output_tokens)

    suggestions = []
    for task_type, counter in keyword_counts.items():
        frequent = [
            (kw, count)
            for kw, count in counter.most_common(20)
            if count >= min_occurences
        ]
        if not frequent:
            continue

        avg_tokens = int(sum(token_sums[task_type]) / len(token_sums[task_type]))
        top_keywords = [kw for kw, _ in frequent[:5]]
        total_occurences = sum(count for _, count in frequent[:5])

        suggestions.append({
            "task_type": task_type,
            "keywords":top_keywords,
            "occurences":total_occurences,
            "avg_actual_tokens":avg_tokens,
            "record_count":len(token_sums[task_type])
        })
    
    suggestions.sort(key=lambda s: s["occurences"], reverse=True)
    return suggestions


def _suggest_new_type(suggestions:dict) -> dict:
    """Derive a new task type name and recommended settings from a suggestion."""
    keywords = suggestions["keywords"]
    avg_tokens = suggestions["avg_actual_tokens"]
    base_type = suggestions["task_type"]

    #name the new type after the most frequent keyword + base type
    new_name = f"{keywords[0]}_{base_type}" if keywords else f"custom_{base_type}"

    #tier recommendation based on avg tokens
    if avg_tokens <= 150:
        tier = "small"
    elif avg_tokens <= 600:
        tier = "medium"
    else:
        tier = "large"

    return {
        "name":new_name,
        "keywords": keywords,
        "base_cap":avg_tokens,
        "floor":max(30, avg_tokens // 4),
        "tier":tier
    }



#config.yaml writer

def _write_rule(config_path:Path, new_type: dict)-> Path:
    with config_path.open() as f:
        config = yaml.safe_load(f)

    task_caps:dict = config.setdefault("task_caps","")
    task_caps[new_type["name"]] = {
        "base_cap": new_type["base_cap"],
        "floor":new_type["floor"]
    }

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Rule '{new_type['name']}' written to {config_path}")


#interactive approval loop

def _prompt_approval(question:str)-> bool:
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ("y","yes"):
            return True
        if answer in ("n","no"):
            return False
        print("Please enter y or n")


def _present_suggestion(index:int, suggestion:dict, new_type:dict) -> None:
    keywords_str = ", ".join(f'"{kw}"' for kw in suggestion["keywords"])
    print(f"\n[{index}] Keywords: {keywords_str} ({suggestion['occurences']} occurences)")
    print(f"Classified as '{suggestion['task_type']}' but needed avg {suggestion['avg_actual_tokens']} tokens")
    print(f"Suggestion: new task type '{new_type['name']}' -> {new_type['tier']} model, max_tokens={new_type['base_cap']}")



async def run(
    days:int,
    min_occurences:int,
    config_path: Path
)-> None:
    engine = create_async_engine(_db_url(),echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    since = datetime.now(timezone.utc) - timedelta(days=days)

    print(f"Querying misclassified requests from the last {days} days...", flush=True)
    records = await _fetch_misclassified(session_factory, since)
    await engine.dispose()

    if not records:
        print("No misclassified requests found. Nothing to suggest.")
        return
    
    print(f"Found {len(records)} misclassified requests.\n")
    suggestions = _find_patterns(records,min_occurences=min_occurences)

    if not suggestions:
        print(f"No keyword patterns found with {min_occurences}+ occurences.")
        print("Try lowering --min-occurences or widening --days.")
        return
    
    print(f"Found {len(suggestions)} potential new rule(s) this week:\n")

    approved = 0
    skipped = 0

    for i, suggestion in enumerate(suggestions, start=1):
        new_type = _suggest_new_type(suggestion)
        _present_suggestion(i, suggestion, new_type)

        if _prompt_approval("-> approve?"):
            _write_rule(config_path,new_type)
            approved+=1
        else:
            print("Skipped.")
            skipped+=1

    print(f"\nDone. {approved} rule(s) approved, {skipped} skipped.")
    if approved:
        print("Restart the router (or send SIGHUP if hot-reload is configured) to apply new rules.")


#cli

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 - rule suggestion system")    
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back this many days for misclassified requests (default: 7)"
    )

    parser.add_argument(
        "--min-occurences",
        type=int,
        default=5,
        help="Minimum keyword frequency to surface a suggestion (default: 5)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: config.yaml)"
    )
    args= parser.parse_args()

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(
        days=args.days,
        min_occurences=args.min_occurences,
        config_path=args.config
    ))


if __name__ == "__main__":
    main()
