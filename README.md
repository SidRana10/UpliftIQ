# UpliftIQ

Causal ML and decision intelligence platform built on uplift datasets:

1. **Hillstrom** E-Mail Analytics (64,000 rows) - in use
2. **Corrected Criteo Uplift** (~14M rows) - documented, not yet downloaded

See [docs/datasets.md](docs/datasets.md) for sources, schemas and notes.

## Status

Project scaffolding and Hillstrom data validation only. No models, API or frontend yet.

## Layout

```
src/upliftiq/
  data/        dataset registry, download, validation
  modeling/    (placeholder) uplift models
  evaluation/  (placeholder) Qini/AUUC and policy evaluation
  backend/     (placeholder) API
frontend/      (placeholder) future UI
tests/
data/raw/      downloaded datasets (git-ignored)
docs/
```

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
python -m upliftiq.data.download     # fetch Hillstrom into data/raw/ and verify SHA-256
python -m upliftiq.data.validate     # check schema, row count, integrity
```

## Development

```bash
ruff check .
ruff format --check .
pytest                    # all tests (data tests skip if Hillstrom is not downloaded)
pytest -m "not network"   # unit tests only
```

Raw data, caches and virtual environments are git-ignored.
