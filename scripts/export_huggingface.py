from __future__ import annotations

import argparse
from pathlib import Path

from asg_transformer.config import settings
from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import SemanticEncoder
from asg_transformer.models.unified_model import ASGUnifiedModel


def build_model() -> ASGUnifiedModel:
    catalog = KnowledgeCatalog(settings.data_dir)
    encoder = SemanticEncoder(
        settings.model_name,
        str(settings.model_dir),
        settings.device,
        settings.reranker_name if settings.enable_reranker else None,
    )
    return ASGUnifiedModel(catalog=catalog, encoder=encoder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or publish the unified ASG model")
    parser.add_argument("--output-dir", default="dist/asg-unified-model")
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model = build_model()
    output = model.save_pretrained(Path(args.output_dir))
    print(f"Exported unified model to: {output}")
    if args.repo_id:
        url = model.push_to_hub(args.repo_id, private=args.private)
        print(f"Published model: {url}")


if __name__ == "__main__":
    main()
