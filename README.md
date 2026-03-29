# llm-router

A lightweight proxy that sits between your application and multiple LLM providers. Classifies incoming prompts using zero-token rule-based analysis, predicts output size, sets the right token cap automatically, and routes to the most efficient model.

No LLM used for routing — pure Python heuristics, runs in under 1ms.

---

## How it works

```
Client request
    ↓
[ FastAPI proxy ]
    ↓
[ Task Analyzer ]       ← zero tokens, pure Python
  - extract 16 signals from prompt text
  - detect task type, output size, confidence
    ↓
[ Classifier ]
  - signals → task type + model tier + max_tokens
  - 3-tier priority (output overrides → routing overrides → normal)
    ↓
[ Model Selector ]
  - pick provider + model from config
  - apply fallback strategy if needed
    ↓
[ Provider Adapter ]
  - Ollama  (local)
  - OpenAI  (cloud)
  - Groq    (cloud, fast)
  - Anthropic (cloud)
    ↓
[ Result Recorder ]
  - log actual vs predicted tokens
  - detect misclassifications
  - recalibrate estimates over time
    ↓
Response to client
```

---

## Task types and routing

| Task type | Signals | max_tokens | Model tier |
|-----------|---------|------------|------------|
| `factual` | starts with what/who/when/where + short | 80 | small |
| `list` | list/enumerate/give me/examples of | 250 | small |
| `code_small` | write/implement + simple scope | 350 | medium |
| `code_large` | write/implement + class/api/system | 800 | large |
| `explanation` | explain/how does/why does | 500 | medium |
| `reasoning` | long prompt + no clear category | 900 | large |
| `unknown_short` | low confidence + short | 300 | medium |
| `unknown_long` | low confidence + long | 900 | large |

The router sets `max_tokens` automatically. Callers don't need to guess.

---

## Project structure

```
llm-router/
├── core/
│   ├── analyzer.py         # signal extraction from raw prompt
│   ├── classifier.py       # signals → task type + tier + max_tokens
│   ├── config.py           # load and validate config.yaml
│   └── recorder.py         # log results, detect misclassifications
├── providers/
│   ├── base.py             # abstract provider interface
│   ├── ollama.py
│   ├── openai.py
│   ├── groq.py
│   └── anthropic.py
├── router/
│   ├── selector.py         # pick model + provider
│   ├── proxy.py            # execute request, handle retries
│   └── fallback.py         # fallback strategies
├── models/
│   └── request.py          # SQLAlchemy model
├── routes/
│   └── router.py           # /route and /stats endpoints
├── scripts/
│   ├── analyze_dataset.py  # phase 0: derive base_cap values from real data
│   └── suggest_rules.py    # phase 5: surface misclassification patterns
├── tests/
│   ├── test_analyzer.py
│   ├── test_classifier.py
│   ├── test_fallback.py
│   └── test_router.py
├── alembic/                # database migrations
├── main.py
├── config.yaml
├── alembic.ini
├── .env.example
└── pyproject.toml
```

---

## Setup

### Requirements

- Python 3.12+
- PostgreSQL

### Install

```bash
git clone https://github.com/Salman4M/llm-router
cd llm-router
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_router
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Edit `config.yaml` to set your preferred default and fallback providers, model names per tier, and routing thresholds.

### Database

```bash
# create the database
sudo -u postgres psql -c "CREATE DATABASE llm_router;"
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'password';"

# run migrations
alembic upgrade head
```

### Run

```bash
uvicorn main:app --reload
```

---

## API

### `POST /route`

Route a prompt to the best model.

**Request:**
```json
{
    "prompt": "What is Python?",
    "force_model": null,
    "force_provider": null,
    "max_tokens": null
}
```

- `force_model` — `"small"` | `"medium"` | `"large"` — override tier selection
- `force_provider` — provider key from config (e.g. `"groq_cloud"`) — override provider
- `max_tokens` — override the router's token cap decision

**Response:**
```json
{
    "response": "Python is a high-level programming language...",
    "model_used": "llama-3.1-8b-instant",
    "provider_used": "groq_cloud",
    "task_type": "factual",
    "routing_confidence": 0.88,
    "max_tokens_set": 80,
    "actual_input_tokens": 12,
    "actual_output_tokens": 41,
    "routing_time_ms": 0.3,
    "was_fallback": false,
    "fallback_reason": null
}
```

### `GET /stats`

Routing statistics across all requests.

```json
{
    "total_requests": 1240,
    "routing_accuracy": 0.78,
    "model_distribution": { "small": 0.42, "medium": 0.35, "large": 0.23 },
    "provider_distribution": { "groq_cloud": 0.89, "ollama_local": 0.11 },
    "avg_tokens_saved_vs_always_large": 312,
    "fallback_rate": 0.06,
    "upgrade_rate": 0.08,
    "most_misclassified_type": "reasoning",
    "avg_response_time_ms": { "small": 180, "medium": 420, "large": 890 }
}
```

---

## Fallback strategy

Six failure modes, each handled separately:

| Failure | Strategy |
|---------|----------|
| Low confidence + short prompt | medium model, 300 tokens |
| Low confidence + long prompt | large model, 900 tokens |
| Provider unavailable / network error | switch to `fallback_provider` |
| Rate limit (429) | switch to `fallback_provider` for this request |
| Truncated response | retry same provider, `max_tokens × 1.5` |
| Still truncated | retry same provider, upgrade tier |
| All retries exhausted | `503 all_providers_unavailable` |

---

## Configuration

`config.yaml` controls providers, model tiers, and routing thresholds:

```yaml
providers:
  groq_cloud:
    type: "groq"
    base_url: "https://api.groq.com/openai/v1"
    api_key_env: "GROQ_API_KEY"
    models:
      small:  "llama-3.1-8b-instant"
      medium: "llama-3.3-70b-versatile"
      large:  "llama-3.3-70b-versatile"

routing:
  default_provider: "groq_cloud"
  fallback_provider: "ollama_local"
  fallback_model: "medium"

thresholds:
  low_confidence: 0.5
  misclassification_ratio: 2.0
  overprovisioned_ratio: 0.3
```

---

## Tuning

### Option 1 — Dataset analysis (Phase 0)

Run the analysis script to derive real `base_cap` values from the LMSYS-Chat-1M dataset instead of using the placeholder defaults:

```bash
# dry run — 10k conversations
python scripts/analyze_dataset.py --max-conversations 10000 --output results.yaml

# full run
python scripts/analyze_dataset.py --output results.yaml

# ShareGPT validation (download file first)
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python scripts/analyze_dataset.py --dataset sharegpt --sharegpt-file ShareGPT_V3_unfiltered_cleaned_split.json --output sharegpt_results.yaml
```

Paste the output YAML block into `config.yaml` under `task_caps`.

### Option 2 — Live recalibration (Phase 6)

After 100+ requests per task type per provider, the recorder automatically recalibrates token estimates from real traffic. No manual steps needed — just let it run.

### Option 3 — Rule suggestions (Phase 5)

Run the suggestion script weekly to surface misclassification patterns:

```bash
python scripts/suggest_rules.py
python scripts/suggest_rules.py --days 14 --min-occurrences 3
```

Approve or reject each suggestion interactively. Approved rules are written directly to `config.yaml`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Tech stack

- Python 3.12
- FastAPI
- httpx
- SQLAlchemy + asyncpg + PostgreSQL
- PyYAML
- python-dotenv
- tiktoken
- pytest