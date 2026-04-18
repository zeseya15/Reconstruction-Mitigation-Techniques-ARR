"""
baselines.py
------------
Baseline privacy mechanisms compared against ARR (paper 5.2):

  1. FL-GAN          — undefended FedTSRGNet (imported from model.py)
  2. FL-GAN + DP-SGD — Opacus-based (ε,δ)-DP training
  3. FL-GAN + Jacobian Reg — output-smoothness regularisation
  4. FL-GAN + privGAN — adversarial membership-privacy mechanism

Each baseline exposes a train_one_round(dataset, ...) interface
consistent with ARRTrainer so that fedavg.py can call them uniformly.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import BiLSTMTCNGAN, LATENT_DIM

# ── Optional Opacus import ────────────────────────────────────────────────────
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn(
        "Opacus not found. DP-SGD baseline will raise at runtime. "
        "Install with: pip install opacus",
        ImportWarning
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared GAN utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _gradient_penalty(D: nn.Module, real: torch.Tensor,
                       fake: torch.Tensor,
                       lambda_gp: float = 10.0,
                       device: str = 'cpu') -> torch.Tensor:
    """WGAN-GP penalty shared by all baselines."""
    B = real.size(0)
    eps = torch.rand(B, 1, device=device)
    interp = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
    d_int = D(interp)
    grad = torch.autograd.grad(
        d_int, interp,
        grad_outputs=torch.ones_like(d_int),
        create_graph=True, retain_graph=True
    )[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


def _wgan_d_loss(D, real, fake, device, lambda_gp=10.0):
    return (D(fake.detach()).mean() - D(real).mean()
            + _gradient_penalty(D, real, fake, lambda_gp, device))


def _wgan_g_loss(D, fake):
    return -D(fake).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FL-GAN (undefended baseline)
# ═══════════════════════════════════════════════════════════════════════════════
class FLGANTrainer:
    """
    Undefended FedTSRGNet trainer.
    Upper bound on utility / lower bound on privacy.
    """
    def __init__(self, model: BiLSTMTCNGAN,
                 latent_dim: int = LATENT_DIM,
                 lr: float = 1e-4,
                 lambda_gp: float = 10.0,
                 device: str = 'cpu'):
        self.G = model.generator
        self.D = model.discriminator
        self.model = model
        self.latent_dim = latent_dim
        self.lambda_gp  = lambda_gp
        self.device     = device

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.opt_cls = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_round(self,
                        dataset,
                        batch_size: int = 256,
                        d_steps: int = 5) -> dict:
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
        g_losses, d_losses, cls_losses = [], [], []
        criterion = nn.CrossEntropyLoss()

        for real_x, labels in loader:
            real_x = real_x.to(self.device)
            labels = labels.to(self.device)
            B = real_x.size(0)

            # discriminator
            for _ in range(d_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_x = self.G(z).detach()
                d_loss = _wgan_d_loss(self.D, real_x, fake_x,
                                      self.device, self.lambda_gp)
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()
                d_losses.append(d_loss.item())

            # generator
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_x = self.G(z)
            g_loss = _wgan_g_loss(self.D, fake_x)
            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()
            g_losses.append(g_loss.item())

            # classifier
            logits = self.model(real_x)
            cls_loss = criterion(logits, labels)
            self.opt_cls.zero_grad()
            cls_loss.backward()
            self.opt_cls.step()
            cls_losses.append(cls_loss.item())

        return {
            'g_loss':   float(np.mean(g_losses)),
            'd_loss':   float(np.mean(d_losses)),
            'cls_loss': float(np.mean(cls_losses))
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FL-GAN + DP-SGD  (Opacus)
# ═══════════════════════════════════════════════════════════════════════════════
class DPSGDTrainer:
    """
    FL-GAN with (ε,δ)-DP-SGD via Opacus.

    Paper setting: ε=1, δ=1e-5, Rényi-DP accounting (paper 5.2).

    Notes
    -----
    DP-SGD is applied to the generator training only, consistent with
    the paper's framing: the privacy threat concerns what the generator
    has memorised.
    """

    def __init__(self, model: BiLSTMTCNGAN,
                 target_epsilon:  float = 1.0,
                 target_delta:    float = 1e-5,
                 max_grad_norm:   float = 1.0,
                 latent_dim:      int   = LATENT_DIM,
                 lr:              float = 1e-4,
                 lambda_gp:       float = 10.0,
                 device:          str   = 'cpu'):
        if not OPACUS_AVAILABLE:
            raise RuntimeError(
                "Opacus is required for DP-SGD. pip install opacus"
            )
        self.model          = model
        self.G              = model.generator
        self.D              = model.discriminator
        self.latent_dim     = latent_dim
        self.lambda_gp      = lambda_gp
        self.device         = device
        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta
        self.max_grad_norm  = max_grad_norm
        self.lr             = lr
        self._privacy_engine: Optional[PrivacyEngine] = None

        self.opt_D   = torch.optim.Adam(self.D.parameters(), lr=lr,
                                        betas=(0.5, 0.999))
        self.opt_cls = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _attach_privacy_engine(self, loader: DataLoader) -> DataLoader:
        """Attach Opacus PrivacyEngine to the generator optimiser."""
        # Validate and fix model layers for Opacus compatibility
        self.G = ModuleValidator.fix(self.G)
        self.G.to(self.device)
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.lr,
                                 betas=(0.5, 0.999))
        engine = PrivacyEngine()
        self.G, opt_G, loader = engine.make_private_with_epsilon(
            module=self.G,
            optimizer=opt_G,
            data_loader=loader,
            epochs=1,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm
        )
        self._privacy_engine = engine
        self.opt_G = opt_G
        return loader

    def train_one_round(self,
                        dataset,
                        batch_size: int = 256,
                        d_steps: int = 5) -> dict:
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)

        # Attach DP engine on first round
        if self._privacy_engine is None:
            loader = self._attach_privacy_engine(loader)

        criterion = nn.CrossEntropyLoss()
        g_losses, d_losses, cls_losses = [], [], []

        for real_x, labels in loader:
            real_x = real_x.to(self.device)
            labels = labels.to(self.device)
            B = real_x.size(0)

            # discriminator (no DP)
            for _ in range(d_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_x = self.G(z).detach()
                d_loss = _wgan_d_loss(self.D, real_x, fake_x,
                                      self.device, self.lambda_gp)
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()
                d_losses.append(d_loss.item())

            # generator with DP-SGD
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_x = self.G(z)
            g_loss = _wgan_g_loss(self.D, fake_x)
            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()
            g_losses.append(g_loss.item())

            # classifier (no DP)
            logits = self.model(real_x)
            cls_loss = criterion(logits, labels)
            self.opt_cls.zero_grad()
            cls_loss.backward()
            self.opt_cls.step()
            cls_losses.append(cls_loss.item())

        epsilon = self._privacy_engine.get_epsilon(self.target_delta)
        return {
            'g_loss':   float(np.mean(g_losses)),
            'd_loss':   float(np.mean(d_losses)),
            'cls_loss': float(np.mean(cls_losses)),
            'epsilon':  epsilon
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FL-GAN + Jacobian Regularisation
# ═══════════════════════════════════════════════════════════════════════════════
class JacobianRegTrainer:
    """
    FL-GAN with Jacobian regularisation on the discriminator output.

    Penalises ‖∂D(x)/∂x‖_F to smooth discriminator landscape,
    reducing sensitivity to individual records (paper 5.2, baseline 3).
    Does NOT directly target reconstructability — confirmed insufficient
    by paper results.
    """

    def __init__(self, model: BiLSTMTCNGAN,
                 lambda_jac: float = 0.01,
                 adaptive:   bool  = False,
                 alpha_jac:  float = 0.5,
                 latent_dim: int   = LATENT_DIM,
                 lr:         float = 1e-4,
                 lambda_gp:  float = 10.0,
                 device:     str   = 'cpu'):
        self.G          = model.generator
        self.D          = model.discriminator
        self.model      = model
        self.lambda_jac = lambda_jac
        self.adaptive   = adaptive
        self.alpha_jac  = alpha_jac
        self.latent_dim = latent_dim
        self.lambda_gp  = lambda_gp
        self.device     = device

        self.opt_G   = torch.optim.Adam(self.G.parameters(), lr=lr,
                                        betas=(0.5, 0.999))
        self.opt_D   = torch.optim.Adam(self.D.parameters(), lr=lr,
                                        betas=(0.5, 0.999))
        self.opt_cls = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _jacobian_penalty(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        y = self.D(x)
        grad = torch.autograd.grad(
            y, x, torch.ones_like(y), create_graph=True
        )[0]
        jac_norm = grad.norm(2)
        lam = (self.lambda_jac *
               (1 + self.alpha_jac * torch.log1p(jac_norm))
               if self.adaptive else self.lambda_jac)
        return lam * jac_norm

    def train_one_round(self,
                        dataset,
                        batch_size: int = 256,
                        d_steps: int = 5) -> dict:
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
        criterion = nn.CrossEntropyLoss()
        g_losses, d_losses, cls_losses = [], [], []

        for real_x, labels in loader:
            real_x = real_x.to(self.device)
            labels = labels.to(self.device)
            B = real_x.size(0)

            # discriminator + Jacobian penalty
            for _ in range(d_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_x = self.G(z).detach()
                d_loss = (_wgan_d_loss(self.D, real_x, fake_x,
                                       self.device, self.lambda_gp)
                          + self._jacobian_penalty(real_x))
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()
                d_losses.append(d_loss.item())

            # generator
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_x = self.G(z)
            g_loss = _wgan_g_loss(self.D, fake_x)
            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()
            g_losses.append(g_loss.item())

            # classifier
            logits = self.model(real_x)
            cls_loss = criterion(logits, labels)
            self.opt_cls.zero_grad()
            cls_loss.backward()
            self.opt_cls.step()
            cls_losses.append(cls_loss.item())

        return {
            'g_loss':   float(np.mean(g_losses)),
            'd_loss':   float(np.mean(d_losses)),
            'cls_loss': float(np.mean(cls_losses))
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FL-GAN + privGAN
# ═══════════════════════════════════════════════════════════════════════════════
class PrivGANTrainer:
    """
    FL-GAN with privGAN auxiliary discriminator (Mukherjee et al., 2021).

    An auxiliary discriminator D_priv tries to distinguish whether a
    generated sample is from the local training set (membership inference).
    The generator is additionally penalised to minimise D_priv's advantage.

    Note: privGAN minimises membership inference risk, NOT reconstructability.
    This leaves the nearest-neighbour reconstruction leakage vector open —
    confirmed by paper results (RSR 0.72 vs 0.19 for ARR).
    """

    def __init__(self, model: BiLSTMTCNGAN,
                 lambda_priv: float = 0.1,
                 latent_dim:  int   = LATENT_DIM,
                 lr:          float = 1e-4,
                 lambda_gp:   float = 10.0,
                 device:      str   = 'cpu'):
        self.G           = model.generator
        self.D           = model.discriminator
        self.model       = model
        self.lambda_priv = lambda_priv
        self.latent_dim  = latent_dim
        self.lambda_gp   = lambda_gp
        self.device      = device

        # Auxiliary membership discriminator D_priv
        input_dim = model.input_dim
        self.D_priv = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        ).to(device)

        self.opt_G      = torch.optim.Adam(self.G.parameters(), lr=lr,
                                           betas=(0.5, 0.999))
        self.opt_D      = torch.optim.Adam(self.D.parameters(), lr=lr,
                                           betas=(0.5, 0.999))
        self.opt_Dpriv  = torch.optim.Adam(self.D_priv.parameters(), lr=lr,
                                           betas=(0.5, 0.999))
        self.opt_cls    = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train_one_round(self,
                        dataset,
                        batch_size: int = 256,
                        d_steps: int = 5) -> dict:
        self.model.train()
        self.D_priv.train()
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
        criterion = nn.CrossEntropyLoss()
        bce = nn.BCEWithLogitsLoss()
        g_losses, d_losses, priv_losses, cls_losses = [], [], [], []

        for real_x, labels in loader:
            real_x = real_x.to(self.device)
            labels = labels.to(self.device)
            B = real_x.size(0)

            # ── D_main ───────────────────────────────────────────────────
            for _ in range(d_steps):
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_x = self.G(z).detach()
                d_loss = _wgan_d_loss(self.D, real_x, fake_x,
                                      self.device, self.lambda_gp)
                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()
                d_losses.append(d_loss.item())

            # ── D_priv: distinguish real (member) vs fake (non-member) ───
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_x = self.G(z).detach()
            real_labels = torch.ones(B, 1, device=self.device)
            fake_labels = torch.zeros(B, 1, device=self.device)
            dp_loss = (bce(self.D_priv(real_x), real_labels)
                       + bce(self.D_priv(fake_x), fake_labels))
            self.opt_Dpriv.zero_grad()
            dp_loss.backward()
            self.opt_Dpriv.step()

            # ── Generator: fool D_main + fool D_priv (confuse membership) ─
            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_x = self.G(z)
            g_gan  = _wgan_g_loss(self.D, fake_x)
            # privGAN: generator wants D_priv to predict 0.5 (confused)
            priv_pred = self.D_priv(fake_x)
            priv_loss = bce(priv_pred,
                            0.5 * torch.ones_like(priv_pred))
            g_total = g_gan + self.lambda_priv * priv_loss
            self.opt_G.zero_grad()
            g_total.backward()
            self.opt_G.step()
            g_losses.append(g_total.item())
            priv_losses.append(priv_loss.item())

            # ── Classifier ───────────────────────────────────────────────
            logits  = self.model(real_x)
            cls_loss = criterion(logits, labels)
            self.opt_cls.zero_grad()
            cls_loss.backward()
            self.opt_cls.step()
            cls_losses.append(cls_loss.item())

        return {
            'g_loss':    float(np.mean(g_losses)),
            'd_loss':    float(np.mean(d_losses)),
            'priv_loss': float(np.mean(priv_losses)),
            'cls_loss':  float(np.mean(cls_losses))
        }
