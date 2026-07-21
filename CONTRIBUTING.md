# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
```

## Quality requirements

Before opening a pull request, run:

```bash
ruff check src tests scripts examples
pytest -q
python -m compileall -q src scripts examples
```

## Contribution rules

- Preserve compatibility with `AutoModelForCausalLM`.
- Store trainable modules beneath `ASGTransformerForCausalLM`.
- Add tests for serialization and loading changes.
- Keep defensive safety boundaries intact.
- Document configuration or public API changes.
- Never commit secrets, tokens, private datasets, or restricted model artifacts.
