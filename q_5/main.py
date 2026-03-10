"""
IMDB Sentiment Analysis — Main Entry Point
===========================================
Runs 4 experiments (BiLSTM/BiGRU × Learned/GloVe) and generates reports.

Usage:
    python main.py
"""

from pathlib import Path

import pandas as pd
import torch

from src.config import Config, SEED, set_seed, get_device
from src.data import (
    build_vocabulary,
    create_dataloaders,
    encode_reviews,
    load_and_preprocess,
)
from src.train import run_experiment
from src.visualize import (
    plot_comparison,
    plot_training_curves,
    print_results_table,
    save_results_json,
    show_examples,
)


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    base_dir = Path(__file__).parent
    csv_path = base_dir / "IMDB Dataset.csv"
    plots_dir = base_dir / "plots"
    models_dir = base_dir / "models"
    plots_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # ---- Phase 1: Data Loading & Preprocessing ----
    print("\n" + "=" * 60)
    print("PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)

    tokenized, labels = load_and_preprocess(str(csv_path))
    original_reviews = pd.read_csv(str(csv_path))["review"].tolist()

    # ---- Phase 2: Vocabulary ----
    print("\n" + "=" * 60)
    print("PHASE 2: Vocabulary Building")
    print("=" * 60)

    config_base = Config()
    word2idx, _ = build_vocabulary(tokenized, config_base.vocab_size)

    # ---- Phase 3: Encoding ----
    print("\n" + "=" * 60)
    print("PHASE 3: Sequence Encoding & Padding")
    print("=" * 60)

    sequences = encode_reviews(tokenized, word2idx, config_base.max_seq_len)
    print(f"  Encoded shape: {sequences.shape}")

    # ---- Phase 4: DataLoaders ----
    print("\n" + "=" * 60)
    print("PHASE 4: Train/Val/Test Split")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, _, test_idx = create_dataloaders(
        sequences, labels, config_base.batch_size
    )

    # ---- Phase 5: Experiments ----
    print("\n" + "=" * 60)
    print("PHASE 5: Running Experiments")
    print("=" * 60)

    experiments = [
        Config(rnn_type="LSTM", use_glove=False),
        Config(rnn_type="GRU", use_glove=False),
        Config(rnn_type="LSTM", use_glove=True),
        Config(rnn_type="GRU", use_glove=True),
    ]

    results = []
    for cfg in experiments:
        result = run_experiment(
            cfg, train_loader, val_loader, test_loader, word2idx, device, models_dir
        )
        results.append(result)

    # ---- Phase 6: Results & Visualization ----
    print("\n" + "=" * 60)
    print("PHASE 6: Results & Visualization")
    print("=" * 60)

    print_results_table(results)

    best_result = max(results, key=lambda r: r["test_acc"])
    print(
        f"\nBest model: {best_result['config'].name} "
        f"({best_result['test_acc']*100:.2f}% test accuracy)"
    )

    show_examples(
        model=best_result["model"],
        original_reviews=original_reviews,
        tokenized_reviews=tokenized,
        labels=labels,
        test_indices=test_idx,
        word2idx=word2idx,
        max_seq_len=config_base.max_seq_len,
        device=device,
        n=5,
        save_dir=base_dir,
    )

    print("\nGenerating plots...")
    plot_training_curves(results, plots_dir)
    plot_comparison(results, plots_dir)
    save_results_json(results, base_dir)

    print("\n" + "=" * 60)
    print("DONE — All experiments complete!")
    print("=" * 60)
    print(f"  Plots:   {plots_dir}/")
    print(f"  Models:  {models_dir}/")
    print(f"  Results: {base_dir / 'results.json'}")


if __name__ == "__main__":
    main()
