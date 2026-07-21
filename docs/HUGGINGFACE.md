# ASGTransformer Hugging Face Workflow

ASGTransformer uses the standard Transformers custom-model pattern:

- `ASGTransformerConfig` inherits `PretrainedConfig`.
- `ASGTransformerForCausalLM` inherits `PreTrainedModel` and `GenerationMixin`.
- `config.json` declares `auto_map` for `AutoConfig` and `AutoModelForCausalLM`.
- Custom Python files are copied into the exported model repository.
- All model parameters are stored in Safetensors.

Use `scripts/export_transformers_model.py` to create the repository and `scripts/verify_transformers_checkpoint.py` to validate it before upload.
