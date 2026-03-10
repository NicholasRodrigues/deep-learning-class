"""SentimentRNN model with self-attention and pooling."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import PAD_IDX


class SelfAttention(nn.Module):
    """
    Learned self-attention over RNN hidden states.

    Computes a weighted sum of hidden states, letting the model
    focus on the most sentiment-bearing words in a review.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, rnn_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_output: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) boolean — True where input is padding

        Returns:
            context: (batch, hidden_dim) attention-weighted sum
            weights: (batch, seq_len) attention weights
        """
        scores = self.projection(rnn_output).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), rnn_output).squeeze(1)
        return context, weights


class SentimentRNN(nn.Module):
    """
    Bidirectional LSTM/GRU with self-attention for binary sentiment classification.

    Architecture:
      Embedding → BiRNN → [Self-Attention ‖ Mean Pool ‖ Max Pool] → Dropout → FC → Sigmoid

    Three complementary representations are concatenated:
      1. Self-attention weighted sum  (focuses on key sentiment words)
      2. Mean pooling                 (overall sentiment tone)
      3. Max pooling                  (strongest feature activations)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        bidirectional: bool,
        dropout: float,
        rnn_type: str = "LSTM",
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_directions = 2 if bidirectional else 1
        rnn_output_dim = hidden_dim * self.n_directions

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # fine-tune

        self.embed_dropout = nn.Dropout(0.3)

        # RNN
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self._init_rnn_weights()

        # Attention + classifier
        self.attention = SelfAttention(rnn_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_output_dim * 3, 1)

    def _init_rnn_weights(self) -> None:
        """Orthogonal init for recurrent weights (improves gradient flow)."""
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token indices

        Returns:
            (batch, 1) sigmoid probabilities
        """
        pad_mask = x == PAD_IDX  # (batch, seq_len)

        embedded = self.embed_dropout(self.embedding(x))
        rnn_out, _ = self.rnn(embedded)  # (batch, seq_len, rnn_output_dim)

        # 1) Self-attention
        attn_out, _ = self.attention(rnn_out, pad_mask)

        # 2) Mean pooling (exclude padding)
        mask_exp = pad_mask.unsqueeze(-1).expand_as(rnn_out)
        rnn_masked = rnn_out.masked_fill(mask_exp, 0.0)
        lengths = (~pad_mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        mean_pool = rnn_masked.sum(dim=1) / lengths

        # 3) Max pooling (exclude padding)
        max_pool, _ = rnn_out.masked_fill(mask_exp, float("-inf")).max(dim=1)

        combined = torch.cat([attn_out, mean_pool, max_pool], dim=1)
        return torch.sigmoid(self.fc(self.dropout(combined)))

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
