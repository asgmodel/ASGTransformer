from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from asg_transformer.config import PROJECT_ROOT
from asg_transformer.monitoring import WandbTracker


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_examples(data_dir: Path) -> list[InputExample]:
    catalog_path = data_dir / "techniques.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Training catalog not found: {catalog_path}")

    techniques = json.loads(catalog_path.read_text(encoding="utf-8"))
    examples: list[InputExample] = []
    for item in techniques:
        canonical = f"{item['label']} — {item['description']}"
        examples.append(InputExample(texts=[item["description"], canonical]))
        examples.append(
            InputExample(
                texts=[f"Technique {item['label']} in tactic {item['tactic']}", canonical]
            )
        )
    return examples


def build_evaluator(examples: list[InputExample], seed: int) -> EmbeddingSimilarityEvaluator:
    rng = random.Random(seed)
    anchors = [example.texts[0] for example in examples]
    positives = [example.texts[1] for example in examples]

    sentences1: list[str] = []
    sentences2: list[str] = []
    scores: list[float] = []

    for index, (anchor, positive) in enumerate(zip(anchors, positives, strict=True)):
        sentences1.append(anchor)
        sentences2.append(positive)
        scores.append(1.0)

        negative_index = rng.randrange(len(positives))
        if negative_index == index:
            negative_index = (negative_index + 1) % len(positives)
        sentences1.append(anchor)
        sentences2.append(positives[negative_index])
        scores.append(0.0)

    return EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        name="asg-validation",
        show_progress_bar=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the ASG semantic Transformer")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/asg-encoder"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--evaluation-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    parser.add_argument("--wandb-project", default="asg-transformer")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="semantic-encoder")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    examples = load_examples(data_dir)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model = SentenceTransformer(args.base_model)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    evaluator = build_evaluator(examples, args.seed)
    warmup_steps = max(1, int(len(loader) * args.epochs * args.warmup_ratio))

    tracker_config = {
        "base_model": args.base_model,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "evaluation_steps": args.evaluation_steps,
        "seed": args.seed,
        "training_examples": len(examples),
    }

    with WandbTracker(
        enabled=args.wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=["transformer", "cybersecurity", "sentence-transformers"],
        mode=args.wandb_mode,
        config=tracker_config,
    ) as tracker:

        def evaluation_callback(score: float, epoch: int, steps: int) -> None:
            tracker.log(
                {
                    "evaluation/cosine_spearman": score,
                    "training/epoch": epoch,
                    "training/global_step": steps,
                },
                step=steps,
            )

        tracker.log({"dataset/training_examples": len(examples)})

        model.fit(
            train_objectives=[(loader, train_loss)],
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=args.evaluation_steps,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.learning_rate},
            output_path=str(output_dir),
            callback=evaluation_callback,
            show_progress_bar=True,
            save_best_model=True,
        )

        final_score = evaluator(model, output_path=str(output_dir))
        tracker.log({"evaluation/final_cosine_spearman": final_score})
        tracker.log_model(output_dir, aliases=["latest", "best"])

    print(f"Training completed. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
