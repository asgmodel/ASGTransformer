from __future__ import annotations
import argparse, json
from pathlib import Path
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

def load_examples(data_dir: Path) -> list[InputExample]:
    techniques = json.loads((data_dir / "techniques.json").read_text(encoding="utf-8"))
    examples=[]
    for item in techniques:
        canonical=f"{item['label']} — {item['description']}"
        examples.append(InputExample(texts=[item['description'], canonical]))
        examples.append(InputExample(texts=[f"Technique {item['label']} in tactic {item['tactic']}", canonical]))
    return examples

def main():
    parser=argparse.ArgumentParser(description="Fine-tune the ASG semantic Transformer")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/asg-encoder"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args=parser.parse_args()
    model=SentenceTransformer(args.base_model)
    loader=DataLoader(load_examples(args.data_dir), shuffle=True, batch_size=args.batch_size)
    model.fit(train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))], epochs=args.epochs, warmup_steps=max(1,len(loader)), output_path=str(args.output_dir), show_progress_bar=True)

if __name__ == "__main__": main()
