"""Data loading, preprocessing, vocabulary building, and DataLoader creation."""

import html
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import pandas as pd

from src.config import (
    GLOVE_DIM,
    GLOVE_URL,
    PAD_IDX,
    SEED,
    UNK_IDX,
    UNK_TOKEN,
    PAD_TOKEN,
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_review(text: str) -> list[str]:
    """
    Clean and tokenize a single review.

    Steps:
      1. Unescape HTML entities (&amp; -> &, etc.)
      2. Remove HTML tags (<br />, <p>, etc.)
      3. Lowercase
      4. Remove non-alphabetic characters
      5. Whitespace tokenization

    Stopwords are intentionally kept — negation words ('not', 'no', 'never')
    carry critical sentiment information.
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def load_and_preprocess(csv_path: str) -> tuple[list[list[str]], np.ndarray]:
    """Load the IMDB CSV and return tokenized reviews + binary labels."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total reviews: {len(df)}")
    print(f"  Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}")

    labels = np.array((df["sentiment"] == "positive").astype(int))

    print("  Preprocessing reviews (HTML cleanup, lowercase, tokenization)...")
    tokenized = [preprocess_review(r) for r in df["review"]]

    lengths = [len(t) for t in tokenized]
    print(
        f"  Token lengths — mean: {np.mean(lengths):.0f}, "
        f"median: {np.median(lengths):.0f}, "
        f"max: {np.max(lengths)}, min: {np.min(lengths)}"
    )
    print(
        f"  Percentiles — 75th: {np.percentile(lengths, 75):.0f}, "
        f"90th: {np.percentile(lengths, 90):.0f}, "
        f"95th: {np.percentile(lengths, 95):.0f}"
    )

    return tokenized, labels


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_vocabulary(
    tokenized_reviews: list[list[str]], vocab_size: int
) -> tuple[dict[str, int], dict[int, str]]:
    """Build word-to-index and index-to-word mappings from the most frequent words."""
    counter: Counter[str] = Counter()
    for tokens in tokenized_reviews:
        counter.update(tokens)

    most_common = counter.most_common(vocab_size)
    print(f"  Vocabulary: {vocab_size} words (from {len(counter)} unique tokens)")
    print(f"  Most common: {[w for w, _ in most_common[:10]]}")
    print(f"  Coverage: {sum(c for _, c in most_common) / sum(counter.values()):.1%}")

    word2idx: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for i, (word, _) in enumerate(most_common):
        word2idx[word] = i + 2

    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def encode_reviews(
    tokenized_reviews: list[list[str]],
    word2idx: dict[str, int],
    max_seq_len: int,
) -> np.ndarray:
    """Convert tokenized reviews to padded integer sequences."""
    encoded = np.full((len(tokenized_reviews), max_seq_len), PAD_IDX, dtype=np.int64)
    for i, tokens in enumerate(tokenized_reviews):
        indices = [word2idx.get(t, UNK_IDX) for t in tokens[:max_seq_len]]
        encoded[i, : len(indices)] = indices
    return encoded


# ---------------------------------------------------------------------------
# PyTorch Dataset & DataLoader
# ---------------------------------------------------------------------------


class IMDBDataset(Dataset):
    """Wraps encoded sequences and labels as a PyTorch Dataset."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test (80/10/10) and create DataLoaders.

    Returns (train_loader, val_loader, test_loader, train_idx, val_idx, test_idx).
    """
    indices = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
    )

    print(
        f"  Split sizes — train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    train_loader = DataLoader(
        IMDBDataset(sequences[train_idx], labels[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        IMDBDataset(sequences[val_idx], labels[val_idx]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        IMDBDataset(sequences[test_idx], labels[test_idx]),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# GloVe embeddings
# ---------------------------------------------------------------------------

GLOVE_DIR = Path(__file__).parent.parent / "glove"


def download_glove() -> Path:
    """Download and extract GloVe embeddings if not already present."""
    GLOVE_DIR.mkdir(exist_ok=True)
    glove_file = GLOVE_DIR / f"glove.6B.{GLOVE_DIM}d.txt"

    if glove_file.exists():
        print(f"  GloVe file already exists: {glove_file}")
        return glove_file

    zip_path = GLOVE_DIR / "glove.6B.zip"
    if not zip_path.exists():
        print(f"  Downloading GloVe from {GLOVE_URL}...")
        print("  (This is ~862 MB, may take a few minutes)")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print(f"  Extracting glove.6B.{GLOVE_DIM}d.txt...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(f"glove.6B.{GLOVE_DIM}d.txt", GLOVE_DIR)

    return glove_file


def load_glove_embeddings(
    word2idx: dict[str, int], embed_dim: int
) -> torch.Tensor:
    """
    Build an embedding matrix from GloVe vectors aligned with our vocabulary.

    Words not in GloVe are initialized with N(0, 0.6). The <pad> embedding is zeros.
    """
    glove_path = download_glove()

    print(f"  Loading GloVe-{embed_dim}d vectors from {glove_path}...")
    glove_vectors: dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                glove_vectors[word] = np.array(parts[1:], dtype=np.float32)

    vocab_total = len(word2idx)
    embedding_matrix = torch.randn(vocab_total, embed_dim) * 0.6
    embedding_matrix[PAD_IDX] = torch.zeros(embed_dim)

    found = 0
    for word, idx in word2idx.items():
        if word in glove_vectors:
            embedding_matrix[idx] = torch.from_numpy(glove_vectors[word])
            found += 1

    print(f"  GloVe coverage: {found}/{vocab_total} words ({found / vocab_total:.1%})")
    return embedding_matrix
