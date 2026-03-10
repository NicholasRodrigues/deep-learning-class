"""Training loop, evaluation, and experiment runner."""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config, GLOVE_DIM, set_seed
from src.data import load_glove_embeddings
from src.model import SentimentRNN


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(seqs).squeeze(1)
        loss = criterion(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5).float() == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a dataset. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        preds = model(seqs).squeeze(1)
        loss = criterion(preds, labels)

        total_loss += loss.item() * len(labels)
        correct += ((preds >= 0.5).float() == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device,
    save_dir: Path,
) -> dict:
    """
    Full training loop with early stopping and LR scheduling.

    Returns a history dict with per-epoch metrics and best_epoch.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    model_path = save_dir / f"best_{config.name}.pt"

    print(f"\n{'='*60}")
    print(f"Training: {config.name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Embed dim: {config.embed_dim}, Hidden dim: {config.hidden_dim}")
    print(f"  Seq len: {config.max_seq_len}, Batch: {config.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:2d}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {time.time() - t0:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"  Best validation loss: {best_val_loss:.4f}")

    history["best_epoch"] = len(history["val_loss"]) - patience_counter  # type: ignore[assignment]
    return history


def run_experiment(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    word2idx: dict[str, int],
    device: torch.device,
    save_dir: Path,
) -> dict:
    """Run a single experiment: build model, train, evaluate on test set."""
    set_seed(config.seed)

    pretrained = None
    if config.use_glove:
        config.embed_dim = GLOVE_DIM
        pretrained = load_glove_embeddings(word2idx, config.embed_dim)

    model = SentimentRNN(
        vocab_size=len(word2idx),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        rnn_type=config.rnn_type,
        pretrained_embeddings=pretrained,
    ).to(device)

    history = train_model(model, train_loader, val_loader, config, device, save_dir)

    criterion = nn.BCELoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"  >>> Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return {
        "config": config,
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
