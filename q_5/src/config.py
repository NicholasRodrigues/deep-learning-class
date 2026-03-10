"""Experiment configuration and reproducibility utilities."""

import random
from dataclasses import dataclass

import numpy as np
import torch

SEED = 42

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

GLOVE_DIM = 300
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    """Holds all hyperparameters for an experiment."""

    rnn_type: str = "LSTM"
    use_glove: bool = False
    vocab_size: int = 20_000
    embed_dim: int = 100
    hidden_dim: int = 128
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.5
    max_seq_len: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    num_epochs: int = 15
    patience: int = 5
    grad_clip: float = 1.0
    seed: int = SEED

    @property
    def name(self) -> str:
        direction = "Bi" if self.bidirectional else ""
        emb = "GloVe" if self.use_glove else "Learned"
        return f"{direction}{self.rnn_type}-{emb}"
