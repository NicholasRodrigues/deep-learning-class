"""Visualization and reporting utilities."""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import PAD_IDX, SEED, UNK_IDX


def plot_training_curves(results: list[dict], save_dir: Path) -> None:
    """Plot training/validation loss and accuracy curves for all experiments."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle(
        "Training Curves — IMDB Sentiment Analysis", fontsize=14, fontweight="bold"
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, res in enumerate(results):
        name = res["config"].name
        h = res["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        c = colors[i % len(colors)]

        axes[0, 0].plot(epochs, h["train_loss"], label=name, color=c, linewidth=1.5)
        axes[0, 1].plot(epochs, h["val_loss"], label=name, color=c, linewidth=1.5)
        axes[1, 0].plot(epochs, h["train_acc"], label=name, color=c, linewidth=1.5)
        axes[1, 1].plot(epochs, h["val_acc"], label=name, color=c, linewidth=1.5)

    titles = [
        "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy"
    ]
    y_labels = ["Loss (BCE)", "Loss (BCE)", "Accuracy", "Accuracy"]
    for ax, title, ylabel in zip(axes.flat, titles, y_labels):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison(results: list[dict], save_dir: Path) -> None:
    """Bar chart comparing test accuracy across all experiments."""
    names = [r["config"].name for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    _, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, accs, color=colors[: len(names)], edgecolor="black", linewidth=0.5
    )

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — Test Accuracy", fontweight="bold")
    ax.set_ylim(bottom=max(0, min(accs) - 5), top=min(100, max(accs) + 3))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = save_dir / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def show_examples(
    model: torch.nn.Module,
    original_reviews: list[str],
    tokenized_reviews: list[list[str]],
    labels: np.ndarray,
    test_indices: np.ndarray,
    word2idx: dict[str, int],
    max_seq_len: int,
    device: torch.device,
    n: int = 5,
    save_dir: Optional[Path] = None,
) -> str:
    """Display n example predictions from the test set."""
    model.eval()

    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(
        len(test_indices), size=min(n * 4, len(test_indices)), replace=False
    )

    examples = []
    for si in sample_idx:
        if len(examples) >= n:
            break
        global_idx = test_indices[si]
        tokens = tokenized_reviews[global_idx][:max_seq_len]
        indices = [word2idx.get(t, UNK_IDX) for t in tokens]
        padded = indices + [PAD_IDX] * (max_seq_len - len(indices))
        tensor = torch.tensor([padded], dtype=torch.long).to(device)

        with torch.no_grad():
            prob = model(tensor).item()

        pred_label = 1 if prob >= 0.5 else 0
        true_label = int(labels[global_idx])
        examples.append(
            {
                "review": original_reviews[global_idx],
                "true": true_label,
                "pred": pred_label,
                "confidence": prob if pred_label == 1 else 1 - prob,
                "correct": true_label == pred_label,
            }
        )

    label_map = {0: "negative", 1: "positive"}
    lines = ["\n" + "=" * 70, "5 TEST EXAMPLES WITH PREDICTIONS", "=" * 70]

    for i, ex in enumerate(examples, 1):
        display = ex["review"][:300].replace("\n", " ")
        if len(ex["review"]) > 300:
            display += "..."
        status = "CORRECT" if ex["correct"] else "WRONG"
        lines.append(f"\nExample {i} [{status}]:")
        lines.append(f'  Review: "{display}"')
        lines.append(f"  True label:      {label_map[ex['true']]}")
        lines.append(f"  Predicted label:  {label_map[ex['pred']]}")
        lines.append(f"  Confidence:       {ex['confidence']:.4f}")

    output = "\n".join(lines)
    print(output)

    if save_dir:
        path = save_dir / "example_predictions.txt"
        with open(path, "w") as f:
            f.write(output)
        print(f"\n  Saved: {path}")

    return output


def print_results_table(results: list[dict]) -> str:
    """Print a formatted comparison table of all experiments."""
    header = (
        f"\n{'='*85}\n"
        f"{'Model':<20} {'Embeddings':<14} {'Test Acc':>10} {'Test Loss':>10} "
        f"{'Best Epoch':>10} {'Params':>12}\n"
        f"{'-'*85}"
    )
    rows = [header]
    for r in results:
        cfg = r["config"]
        direction = "Bi" if cfg.bidirectional else ""
        emb_label = (
            f"GloVe-{cfg.embed_dim}d" if cfg.use_glove else f"Learned-{cfg.embed_dim}d"
        )
        rows.append(
            f"{direction + cfg.rnn_type:<20} "
            f"{emb_label:<14} "
            f"{r['test_acc']*100:>9.2f}% "
            f"{r['test_loss']:>10.4f} "
            f"{r['history']['best_epoch']:>10} "
            f"{r['model'].count_parameters():>12,}"
        )
    rows.append("=" * 85)
    output = "\n".join(rows)
    print(output)
    return output


def save_results_json(results: list[dict], save_dir: Path) -> None:
    """Save experiment results as JSON for reproducibility."""
    data = []
    for r in results:
        cfg = r["config"]
        data.append(
            {
                "model": cfg.name,
                "rnn_type": cfg.rnn_type,
                "use_glove": cfg.use_glove,
                "vocab_size": cfg.vocab_size,
                "embed_dim": cfg.embed_dim,
                "hidden_dim": cfg.hidden_dim,
                "n_layers": cfg.n_layers,
                "bidirectional": cfg.bidirectional,
                "dropout": cfg.dropout,
                "max_seq_len": cfg.max_seq_len,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "test_accuracy": r["test_acc"],
                "test_loss": r["test_loss"],
                "best_epoch": r["history"]["best_epoch"],
                "total_epochs": len(r["history"]["train_loss"]),
                "parameters": r["model"].count_parameters(),
                "history": {
                    k: v for k, v in r["history"].items() if k != "best_epoch"
                },
            }
        )

    path = save_dir / "results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")
