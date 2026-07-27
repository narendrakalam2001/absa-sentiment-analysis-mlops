"""
train_model.py — Run the full ABSA training pipeline.
Usage:
    python scripts/train_model.py
    python scripts/train_model.py --domain laptops
    python scripts/train_model.py --domain both
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.training_pipeline import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Training Pipeline")
    parser.add_argument(
        "--domain", default="restaurants",
        choices=["restaurants", "laptops", "both"],
        help="SemEval 2014 domain (default: restaurants)"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Starting ABSA Training Pipeline — domain={args.domain}")
    print(f"{'='*60}\n")

    name, model, card = run_training(domain=args.domain)

    print(f"\n✅ Training complete — champion model: {name}")
    print(f"   macro_f1 = {card['metrics'].get('macro_f1', '—')}")
    print(f"   roc_auc  = {card['metrics'].get('roc_auc',  '—')}\n")
