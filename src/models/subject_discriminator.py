import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectDiscriminator(nn.Module):
    """
    Subject discriminator for adversarial training.
    
    Takes mean-pooled embeddings from the foundation model encoder
    and predicts the subject ID. During adversarial training, the
    encoder is trained to fool this discriminator via gradient reversal
    or loss subtraction, encouraging subject-invariant representations.

    Args:
        input_dim: Dimensionality of the input embedding (D)
        hidden_dim: Hidden layer dimensionality
        num_subjects: Number of subjects to discriminate between
        dropout: Dropout probability

    Input:
        x: (B, D) mean-pooled encoder embeddings

    Output:
        logits: (B, num_subjects) unnormalised class scores
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_subjects: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_subjects),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) mean-pooled encoder embeddings

        Returns:
            logits: (B, num_subjects)
        """
        return self.net(x)