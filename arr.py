"""
arr.py
------
Adversarial Reconstruction Regularizer (ARR).

Implements paper (Section 4: Adversarial Reconstruction Regularizer):

  min_{G} max_{R_φ}  L_GAN(G, D)
                   − λ_arr · E_z[ ℓ_recon(R_φ(G(z)), NN_Φ(D_i, G(z))) ]

Components
----------
1. VIMEEncoder     — lightweight tabular encoder Φ (pre-trained or identity).
2. ProxyReconstructor R_φ — two-hidden-layer TCN mapping Φ(x_f) → x̂.
3. contrastive_margin_loss — ℓ_recon (Eq. 5 in paper).
4. ANNIndex        — FAISS-backed nearest-neighbour search in Φ-space.
5. ARRTrainer      — orchestrates Algorithm 1 (local training loop).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings

# ── Optional FAISS import (falls back to brute-force if unavailable) ─────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn(
        "FAISS not found. Falling back to brute-force nearest-neighbour search. "
        "Install with: pip install faiss-cpu",
        ImportWarning
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Tabular Encoder Φ
# ═══════════════════════════════════════════════════════════════════════════════
class VIMEEncoder(nn.Module):
    """
    Lightweight tabular self-supervised encoder inspired by VIME
    (Yoon et al., 2020).

    In this implementation Φ is a two-layer MLP trained with a
    reconstruction objective on the local dataset before federation starts.
    The encoder is frozen during ARR training — only the generator and
    proxy reconstructor are updated.

    Parameters
    ----------
    input_dim : int
    embed_dim : int
        Output embedding dimension d (default 64).
    """
    def __init__(self, input_dim: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Φ-space embedding."""
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def pretrain(self, X: torch.Tensor,
                 epochs: int = 20,
                 lr: float = 1e-3,
                 mask_ratio: float = 0.3,
                 device: str = 'cpu') -> None:
        """
        Self-supervised pre-training: corrupt features, reconstruct originals.
        Runs once per client before federation (paper-5.3).
        """
        self.to(device)
        X = X.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            mask = (torch.rand_like(X) < mask_ratio).float()
            X_corrupt = X * (1 - mask) + torch.randn_like(X) * mask
            loss = F.mse_loss(self.reconstruct(X_corrupt), X)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        # freeze encoder weights
        for p in self.parameters():
            p.requires_grad_(False)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ANN Index (Φ-space nearest-neighbour search)
# ═══════════════════════════════════════════════════════════════════════════════
class ANNIndex:
    """
    Approximate nearest-neighbour index over the local dataset in Φ-space.

    Uses FAISS IVF index (64 cells) when available, otherwise brute-force.
    Re-built once per FL round after Φ-embeddings are computed (Algorithm 1,
    line 2).
    """
    def __init__(self, embed_dim: int, nlist: int = 64):
        self.embed_dim = embed_dim
        self.nlist     = nlist
        self._index    = None
        self._X_raw    = None   # original (un-embedded) records for retrieval

    def build(self, embeddings: np.ndarray, X_raw: np.ndarray) -> None:
        """
        Parameters
        ----------
        embeddings : (N, d) float32 array in Φ-space.
        X_raw      : (N, input_dim) original feature vectors (for NN retrieval).
        """
        embeddings = embeddings.astype(np.float32)
        self._X_raw = X_raw

        if FAISS_AVAILABLE:
            d = embeddings.shape[1]
            nlist = min(self.nlist, max(1, len(embeddings) // 10))
            quantiser = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist,
                                       faiss.METRIC_L2)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = max(1, nlist // 4)
            self._index = index
        else:
            self._index = embeddings   # store raw for brute-force

    def query(self, query_embeddings: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Return the k nearest raw feature vectors for each query embedding.

        Returns
        -------
        (N, k, input_dim) array of nearest neighbours in original feature space.
        """
        query_embeddings = query_embeddings.astype(np.float32)
        N = query_embeddings.shape[0]

        if FAISS_AVAILABLE and isinstance(self._index, faiss.Index):
            _, I = self._index.search(query_embeddings, k)
        else:
            # brute-force L2
            diffs = (self._index[:, None, :] -
                     query_embeddings[None, :, :]) ** 2
            dists = diffs.sum(-1)           # (M, N)
            I = dists.argmin(0)[:, None]    # (N, 1)

        return self._X_raw[I.reshape(N, k)]     # (N, k, input_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Proxy Reconstructor R_φ
# ═══════════════════════════════════════════════════════════════════════════════
class ProxyReconstructor(nn.Module):
    """
    Two-hidden-layer network mapping Φ(x_f) → x̂  ∈ X.

    Architecture follows paper §4.1:
      Φ(x_f) ∈ R^d  →  256  →  256  →  input_dim
    with ReLU activations.

    Re-initialised at the start of each FL round to prevent cross-round
    information accumulation (paper §4.1).
    """
    def __init__(self, embed_dim: int, input_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )

    def forward(self, phi_xf: torch.Tensor) -> torch.Tensor:
        return self.net(phi_xf)

    def reset(self) -> None:
        """Re-initialise weights (called at start of each round)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Contrastive margin loss  ℓ_recon  (Equation 5 in paper)
# ═══════════════════════════════════════════════════════════════════════════════
def contrastive_margin_loss(
    phi: VIMEEncoder,
    R_output: torch.Tensor,     # R_φ(x_f) decoded record  (B, input_dim)
    x_star: torch.Tensor,       # true NN  (B, input_dim)
    x_tilde: torch.Tensor,      # random negative  (B, input_dim)
    margin: float = 1.0
) -> torch.Tensor:
    """
    ℓ_recon = max(0,  ‖Φ(R_φ(x_f)) − Φ(x*)‖₂
                    − ‖Φ(R_φ(x_f)) − Φ(x̃)‖₂  +  m )

    Generator minimises this → its outputs cannot be matched to x*.
    Proxy reconstructor maximises this → tries to rank x* above x̃.
    """
    with torch.no_grad():
        phi_star  = phi(x_star)
        phi_tilde = phi(x_tilde)

    phi_hat = phi(R_output)     # Φ(R_φ(x_f))

    dist_pos = torch.norm(phi_hat - phi_star,  dim=1)   # (B,)
    dist_neg = torch.norm(phi_hat - phi_tilde, dim=1)   # (B,)

    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ARR Trainer  (Algorithm 1 in paper)
# ═══════════════════════════════════════════════════════════════════════════════
class ARRTrainer:
    """
    ARR-augmented local training for one FL client.

    Implements Algorithm 1:
      For each generator update step k_G:
        For each reconstructor update step k_R:
          1. Sample z, compute x_f = G(z)
          2. Retrieve x* = NN_Φ(D_i, x_f);  sample x̃ ~ U(D_i)
          3. Update R_φ to MAXIMISE ℓ_recon
        Update G to MINIMISE L_GAN − λ_arr · ℓ_recon

    Parameters
    ----------
    generator      : nn.Module  (G_i from BiLSTMTCNGAN)
    discriminator  : nn.Module  (D_i from BiLSTMTCNGAN)
    phi_encoder    : VIMEEncoder  (frozen after pre-training)
    input_dim      : int
    latent_dim     : int
    lambda_arr     : float   regularisation weight (default 0.1, paper §6.3)
    margin         : float   contrastive margin m  (default 1.0)
    k_g            : int     generator update steps per round
    k_r            : int     reconstructor steps per generator step
    lr_g           : float   generator learning rate
    lr_r           : float   reconstructor learning rate
    lambda_gp      : float   WGAN-GP gradient penalty weight
    device         : str
    """

    def __init__(
        self,
        generator:     nn.Module,
        discriminator: nn.Module,
        phi_encoder:   VIMEEncoder,
        input_dim:     int,
        latent_dim:    int  = 100,
        lambda_arr:    float = 0.1,
        margin:        float = 1.0,
        k_g:           int   = 10,
        k_r:           int   = 5,
        lr_g:          float = 1e-4,
        lr_r:          float = 1e-3,
        lambda_gp:     float = 10.0,
        device:        str   = 'cpu'
    ):
        self.G          = generator
        self.D          = discriminator
        self.phi        = phi_encoder
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lambda_arr = lambda_arr
        self.margin     = margin
        self.k_g        = k_g
        self.k_r        = k_r
        self.lr_g       = lr_g
        self.lambda_gp  = lambda_gp
        self.device     = device

        # proxy reconstructor — re-initialised each round
        self.R = ProxyReconstructor(
            embed_dim  = phi_encoder.embed_dim,
            input_dim  = input_dim
        ).to(device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_g,
                                      betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_g,
                                      betas=(0.5, 0.999))
        self.opt_R = torch.optim.Adam(self.R.parameters(), lr=lr_r)

        # ANN index — rebuilt each round
        self.ann = ANNIndex(embed_dim=phi_encoder.embed_dim)

    # ── Build ANN index for the local dataset ────────────────────────────────
    def _build_index(self, X_local: torch.Tensor) -> None:
        """Compute Φ-embeddings of D_i and build ANN index."""
        with torch.no_grad():
            emb = self.phi(X_local.to(self.device)).cpu().numpy()
        self.ann.build(emb, X_local.numpy())

    # ── Sample nearest neighbour and negative for a batch ───────────────────
    def _sample_nn_and_negative(
        self, x_f: torch.Tensor, X_local: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_f     : (B, input_dim) generated samples
        Returns x* (true NN) and x̃ (random negative), both (B, input_dim).
        """
        from typing import Tuple  # local import to avoid circular
        with torch.no_grad():
            emb = self.phi(x_f).cpu().numpy()
        nn_raw  = self.ann.query(emb, k=1)[:, 0, :]   # (B, input_dim)
        x_star  = torch.tensor(nn_raw, dtype=torch.float32, device=self.device)

        B = x_f.size(0)
        neg_idx = torch.randint(len(X_local), (B,))
        x_tilde = X_local[neg_idx].to(self.device)

        return x_star, x_tilde

    # ── One FL round of local training ──────────────────────────────────────
    def train_one_round(
        self,
        dataset: 'NetFlowDataset',          # noqa: F821
        batch_size: int = 256,
        discriminator_steps: int = 5        # D updates per G update
    ) -> dict:
        """
        Run ARR-augmented local training for one federation round.

        Returns dict of mean losses for logging.
        """
        from torch.utils.data import DataLoader

        self.G.train()
        self.D.train()

        # Re-initialise proxy reconstructor (paper §4.1)
        self.R.reset()

        # Collect all local feature vectors
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        X_local, _ = next(iter(loader))     # (N, input_dim)

        # Build Φ-space ANN index
        self._build_index(X_local)

        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

        losses = {'g_total': [], 'g_gan': [], 'arr': [], 'r_recon': []}

        for real_x, _ in train_loader:
            real_x = real_x.to(self.device)
            B = real_x.size(0)

            # ── (A) Update discriminator ──────────────────────────────────
            for _ in range(discriminator_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                with torch.no_grad():
                    fake_x = self.G(z)
                d_real = self.D(real_x)
                d_fake = self.D(fake_x.detach())
                # WGAN-GP loss
                d_loss = (d_fake.mean() - d_real.mean()
                          + self._gradient_penalty(real_x, fake_x.detach()))
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()

            # ── (B) Update proxy reconstructor k_R times ─────────────────
            for _ in range(self.k_r):
                z = torch.randn(B, self.latent_dim, device=self.device)
                with torch.no_grad():
                    x_f = self.G(z)
                x_star, x_tilde = self._sample_nn_and_negative(x_f, X_local)
                phi_xf = self.phi(x_f)
                r_out  = self.R(phi_xf)
                # Reconstructor MAXIMISES ℓ_recon → minimise negative
                r_loss = -contrastive_margin_loss(
                    self.phi, r_out, x_star, x_tilde, self.margin
                )
                self.opt_R.zero_grad()
                r_loss.backward()
                self.opt_R.step()
                losses['r_recon'].append(-r_loss.item())

            # ── (C) Update generator (GAN + ARR penalty) ─────────────────
            for _ in range(self.k_g):
                z = torch.randn(B, self.latent_dim, device=self.device)
                x_f = self.G(z)

                # GAN objective (generator wants D to be fooled)
                g_gan = -self.D(x_f).mean()

                # ARR penalty
                x_star, x_tilde = self._sample_nn_and_negative(
                    x_f.detach(), X_local
                )
                phi_xf = self.phi(x_f)
                r_out  = self.R(phi_xf.detach())
                arr_loss = contrastive_margin_loss(
                    self.phi, r_out, x_star, x_tilde, self.margin
                )

                # Eq. (3): min_G  L_GAN(G, D) − λ_arr · ℓ_recon
                g_total = g_gan - self.lambda_arr * arr_loss

                self.opt_G.zero_grad()
                g_total.backward()
                self.opt_G.step()

                losses['g_total'].append(g_total.item())
                losses['g_gan'].append(g_gan.item())
                losses['arr'].append(arr_loss.item())

        return {k: float(np.mean(v)) for k, v in losses.items() if v}

    # ── WGAN-GP helper ───────────────────────────────────────────────────────
    def _gradient_penalty(self, real: torch.Tensor,
                           fake: torch.Tensor) -> torch.Tensor:
        B = real.size(0)
        eps = torch.rand(B, 1, device=self.device)
        interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d_interp = self.D(interp)
        grad = torch.autograd.grad(
            d_interp, interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_gp * gp

    @property
    def lambda_gp(self) -> float:
        return self.__dict__.get('_lambda_gp', 10.0)

    @lambda_gp.setter
    def lambda_gp(self, v: float) -> None:
        self._lambda_gp = v
