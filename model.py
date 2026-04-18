"""
model.py
--------
Core architecture: BiLSTMTCNGAN  (used as FL-GAN / FedTSRGNet baseline).

Architecture follows the paper:
  - BiLSTM (2 layers, hidden=256, bidirectional → 512-dim output)
  - TCN block  (Conv1d + BN + ReLU)
  - Classifier head
  - Generator  (MLP, latent_dim → input_dim)
  - Discriminator (MLP, input_dim → scalar with gradient-penalty support)
"""

import torch
import torch.nn as nn

# ── Hyper-parameters ────────────────────────────────────────────────────────
LATENT_DIM    = 100
LSTM_HIDDEN   = 256
TCN_CHANNELS  = 128
DROPOUT       = 0.3


# ── TCN residual block ───────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    """Single dilated causal-Conv1d block with BatchNorm and residual."""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv   = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn     = nn.BatchNorm1d(out_channels)
        self.relu   = nn.ReLU()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # trim to original length (causal padding adds extra steps)
        out = self.relu(self.bn(self.conv(x)[..., :x.size(-1)]))
        return out + self.residual(x)


# ── Main model ───────────────────────────────────────────────────────────────
class BiLSTMTCNGAN(nn.Module):
    """
    BiLSTM-TCN-GAN.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    classes : int
        Number of output classes for the classifier.
    seq_len : int
        Sequence length. For tabular NetFlow records pass seq_len=1.
    """

    def __init__(self, input_dim: int, classes: int = 2, seq_len: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len   = seq_len

        # ── Encoder ──────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_dim, LSTM_HIDDEN,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=DROPOUT
        )

        self.tcn = TCNBlock(2 * LSTM_HIDDEN, TCN_CHANNELS)

        self.classifier = nn.Sequential(
            nn.Linear(TCN_CHANNELS, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, classes)
        )

        # ── Generator ────────────────────────────────────────────────────────
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        # ── Discriminator (no Sigmoid — use with BCEWithLogitsLoss) ─
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    # ── Forward (classification) ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, input_dim)  or  (B, input_dim) for tabular data.
        """
        if x.dim() == 2:                        # tabular → add seq dimension
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)              # (B, T, 2*H)
        tcn_out = self.tcn(lstm_out.transpose(1, 2))   # (B, C, T)
        pooled  = tcn_out.mean(dim=2)           # (B, C)
        return self.classifier(pooled)

    # ── Generation / discrimination ──────────────────────────────────────────
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, LATENT_DIM)  →  synthetic sample (B, input_dim)"""
        return self.generator(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, input_dim)  →  raw logit (B, 1)"""
        return self.discriminator(x)

    # ── Gradient penalty (GP) ───────────────────────────────────────────
    def gradient_penalty(self, real: torch.Tensor,
                         fake: torch.Tensor,
                         lambda_gp: float = 10.0) -> torch.Tensor:
        B = real.size(0)
        eps = torch.rand(B, 1, device=real.device)
        interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d_interp = self.discriminate(interp)
        grad = torch.autograd.grad(
            d_interp, interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return lambda_gp * gp
