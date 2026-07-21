# Verification Report

## Release

- Project: ASGTransformer
- Version: 7.0.0
- Primary architecture: `ASGTransformerForCausalLM`
- Transformers loader: `AutoModelForCausalLM`
- Checkpoint format: Safetensors

## Completed checks

- Python source compilation completed successfully.
- Test suite completed successfully: 9 passed.
- Configuration validation tested.
- Custom `AutoConfig` and `AutoModelForCausalLM` loading tested.
- Tokenizer loading tested.
- Forward pass tested.
- Standard generation tested.
- Semantic encoder output tested.
- Duration head output tested.
- Scenario head output tested.
- Full state-dictionary save/load equality tested.
- Packaged knowledge prompt construction tested.
- Public `ASGTransformer` alias tested.

## Checkpoint guarantees

The export and verification scripts require these parameter groups:

```text
generator.*
semantic_encoder.*
duration_head.*
scenario_head.*
```

For large checkpoints, Safetensors shards are accepted and validated through
`model.safetensors.index.json`.

## Command used

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src scripts examples
```

## Result

```text
9 passed
```

The optional Ruff command was not executed in the build environment because the
Ruff executable was not installed there. Ruff remains included in the development
dependencies and CI configuration.
