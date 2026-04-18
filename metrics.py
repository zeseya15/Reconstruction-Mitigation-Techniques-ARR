"""
metrics.py
----------
All privacy and utility evaluation metrics (paper 5.3).

Privacy metrics
---------------
P1  RSR     — Reconstruction Success Rate (exact-match & near-match)
P2  RD      — Reconstruction Distance
P3  MI-AUC  — Membership Inference AUC (shadow-model, LOGAN)
P4  AQE     — Attack Query Efficiency
P5  MD      — Metric Divergence (DCR vs RSR)

Utility metrics
---------------
U1  Macro F1
U2  RAR      — Rare-Attack Recall
U3  FPR      — False Positive Rate

Similarity metrics (for MD computation)
----------------------------------------
DCR   — Distance to Closest Record
NNDR  — Nearest-Neighbour Distance Ratio
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, roc_auc_score,
                              confusion_matrix, recall_score)
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Dict, Tuple, List

# ── Optional FAISS ────────────────────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Similarity metrics  (DCR, NNDR)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dcr(X_real: np.ndarray,
                X_synth: np.ndarray,
                n_samples: int = 5000) -> float:
    """
    Distance to Closest Record.

    DCR = mean over x ∈ X_real of min_{s ∈ X_synth} ‖x − s‖₂ / ‖x‖₂

    Normalised to [0,1] by dividing by the real record's norm.
    """
    idx = np.random.choice(len(X_real),
                           min(n_samples, len(X_real)),
                           replace=False)
    X_r = X_real[idx].astype(np.float32)
    X_s = X_synth[:n_samples].astype(np.float32)

    nn = NearestNeighbors(n_neighbors=1, metric='l2').fit(X_s)
    dists, _ = nn.kneighbors(X_r)
    norms = np.linalg.norm(X_r, axis=1, keepdims=True) + 1e-8
    return float((dists / norms).mean())


def compute_nndr(X_real: np.ndarray,
                 X_synth: np.ndarray,
                 n_samples: int = 5000) -> float:
    """
    Nearest-Neighbour Distance Ratio.

    NNDR = mean over s ∈ X_synth of
             min_{x ∈ X_real} ‖s−x‖₂ / second_min_{x ∈ X_real} ‖s−x‖₂
    """
    idx = np.random.choice(len(X_synth),
                           min(n_samples, len(X_synth)),
                           replace=False)
    X_s = X_synth[idx].astype(np.float32)
    X_r = X_real[:n_samples].astype(np.float32)

    nn = NearestNeighbors(n_neighbors=2, metric='l2').fit(X_r)
    dists, _ = nn.kneighbors(X_s)
    ratio = dists[:, 0] / (dists[:, 1] + 1e-8)
    return float(ratio.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# ReconSyn attack  (Ganev & De Cristofaro, 2023)
# ═══════════════════════════════════════════════════════════════════════════════

def reconsyn_attack(
    generator:    nn.Module,
    X_real:       np.ndarray,
    outlier_mask: np.ndarray,
    latent_dim:   int   = 100,
    n_queries:    int   = 1000,
    eta:          float = 0.01,
    n_synth:      int   = 5000,
    device:       str   = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-box gradient-free reconstruction attack (ReconSyn).

    For each outlier record x*, iteratively refines a candidate x̂ by
    maximising the similarity score s(x̂) = DCR_{G}(x̂) via finite-difference
    gradient approximation (Eq. 1 in paper).

    Parameters
    ----------
    generator    : trained generator G^(T).
    X_real       : (N, D) real training records.
    outlier_mask : (N,) bool — True for outlier records.
    n_queries    : maximum oracle queries per target record.
    eta          : step size for gradient-free optimisation.
    n_synth      : number of synthetic samples drawn per query to compute DCR.

    Returns
    -------
    reconstructed : (M, D) reconstructed records for each outlier.
    targets       : (M, D) original outlier records.
    """
    generator.eval()
    outlier_idx = np.where(outlier_mask)[0]
    M = len(outlier_idx)
    D = X_real.shape[1]

    reconstructed = np.zeros((M, D), dtype=np.float32)
    targets       = X_real[outlier_idx].astype(np.float32)

    # generate a fixed synthetic pool for scoring
    with torch.no_grad():
        z    = torch.randn(n_synth, latent_dim, device=device)
        synth_pool = generator(z).cpu().numpy().astype(np.float32)

    nn_index = NearestNeighbors(n_neighbors=1, metric='l2').fit(synth_pool)

    def _score(x: np.ndarray) -> float:
        """Similarity oracle s(x) = negative DCR (higher = closer to training)."""
        d, _ = nn_index.kneighbors(x.reshape(1, -1))
        return -float(d[0, 0])

    for i, oi in enumerate(outlier_idx):
        x_hat = np.random.randn(D).astype(np.float32)  # random init
        best_score = _score(x_hat)

        for _ in range(n_queries):
            # finite-difference gradient estimate
            eps_vec = np.random.randn(D).astype(np.float32) * 1e-3
            s_pos   = _score(x_hat + eps_vec)
            s_neg   = _score(x_hat - eps_vec)
            grad_est = (s_pos - s_neg) / (2 * 1e-3) * eps_vec / (np.linalg.norm(eps_vec) + 1e-8)
            x_hat = x_hat + eta * grad_est

            s = _score(x_hat)
            if s > best_score:
                best_score = s

        reconstructed[i] = x_hat

    return reconstructed, targets


# ═══════════════════════════════════════════════════════════════════════════════
# P1  RSR — Reconstruction Success Rate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rsr(
    reconstructed:    np.ndarray,
    targets:          np.ndarray,
    exact_tolerance:  float = 1e-3,
    near_tolerance:   float = 0.05,
    phi_encoder:      Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Reconstruction Success Rate.

    exact_tolerance : ε_rec for exact-match RSR (paper: 1e-3, ℓ∞).
    near_tolerance  : ε_rec for near-match RSR  (paper: 0.05, Φ-space ℓ₂).

    Returns dict with keys 'rsr_exact' and 'rsr_near'.
    """
    N = len(targets)
    assert len(reconstructed) == N

    # exact-match: ‖x̂ − x*‖_∞ ≤ exact_tolerance
    exact_matches = np.max(np.abs(reconstructed - targets), axis=1) <= exact_tolerance
    rsr_exact = float(exact_matches.mean())

    # near-match: normalised Φ-space ℓ₂ ≤ near_tolerance
    if phi_encoder is not None:
        with torch.no_grad():
            phi_rec = phi_encoder(torch.tensor(reconstructed)).numpy()
            phi_tar = phi_encoder(torch.tensor(targets)).numpy()
        diffs = np.linalg.norm(phi_rec - phi_tar, axis=1)
        norms = np.linalg.norm(phi_tar, axis=1) + 1e-8
        near_matches = (diffs / norms) <= near_tolerance
    else:
        diffs = np.linalg.norm(reconstructed - targets, axis=1)
        norms = np.linalg.norm(targets, axis=1) + 1e-8
        near_matches = (diffs / norms) <= near_tolerance

    rsr_near = float(near_matches.mean())
    return {'rsr_exact': rsr_exact, 'rsr_near': rsr_near}


# ═══════════════════════════════════════════════════════════════════════════════
# P2  RD — Reconstruction Distance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rd(reconstructed: np.ndarray,
               targets:       np.ndarray) -> float:
    """
    Mean ℓ₂ distance between reconstructed and target records.
    Higher RD → stronger privacy.
    """
    return float(np.linalg.norm(reconstructed - targets, axis=1).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# P3  MI-AUC — Membership Inference AUC (shadow-model, LOGAN)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mi_auc(
    generator:   nn.Module,
    X_train:     np.ndarray,
    X_test:      np.ndarray,
    discriminator: nn.Module,
    latent_dim:  int = 100,
    n_samples:   int = 2000,
    device:      str = 'cpu'
) -> float:
    """
    Membership inference AUC using discriminator statistics (LOGAN).

    Positive class: training records (members).
    Negative class: held-out test records (non-members).
    Feature: discriminator score D(x).

    AUC = 0.5 → random guessing (ideal privacy).
    """
    generator.eval()
    discriminator.eval()

    idx_tr = np.random.choice(len(X_train), min(n_samples, len(X_train)),
                               replace=False)
    idx_te = np.random.choice(len(X_test),  min(n_samples, len(X_test)),
                               replace=False)

    X_m  = torch.tensor(X_train[idx_tr], dtype=torch.float32, device=device)
    X_nm = torch.tensor(X_test[idx_te],  dtype=torch.float32, device=device)

    with torch.no_grad():
        scores_m  = torch.sigmoid(discriminator(X_m)).cpu().numpy().ravel()
        scores_nm = torch.sigmoid(discriminator(X_nm)).cpu().numpy().ravel()

    scores = np.concatenate([scores_m, scores_nm])
    labels = np.concatenate([np.ones(len(scores_m)),
                              np.zeros(len(scores_nm))])
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# P4  AQE — Attack Query Efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aqe(
    generator:    nn.Module,
    X_real:       np.ndarray,
    outlier_mask: np.ndarray,
    latent_dim:   int   = 100,
    rsr_threshold: float = 0.5,
    max_queries:  int   = 2000,
    eta:          float = 0.01,
    n_synth:      int   = 5000,
    device:       str   = 'cpu',
    step_size:    int   = 100
) -> float:
    """
    Number of oracle queries required to reach RSR ≥ rsr_threshold.
    Higher AQE → costlier attack → stronger privacy (paper P4).

    Uses binary search over query budget [step_size, max_queries].
    """
    generator.eval()
    outlier_idx = np.where(outlier_mask)[0]
    if len(outlier_idx) == 0:
        return float(max_queries)

    with torch.no_grad():
        z = torch.randn(n_synth, latent_dim, device=device)
        synth_pool = generator(z).cpu().numpy().astype(np.float32)
    nn_idx = NearestNeighbors(n_neighbors=1, metric='l2').fit(synth_pool)

    for budget in range(step_size, max_queries + 1, step_size):
        _, targets = reconsyn_attack(
            generator, X_real, outlier_mask,
            latent_dim=latent_dim, n_queries=budget,
            eta=eta, n_synth=n_synth, device=device
        )
        reconstructed = _  # unpacked above
        result = compute_rsr(reconstructed, targets)
        if result['rsr_exact'] >= rsr_threshold:
            return float(budget)

    return float(max_queries)


# ═══════════════════════════════════════════════════════════════════════════════
# P5  MD — Metric Divergence
# ═══════════════════════════════════════════════════════════════════════════════

def compute_md(rsr_exact: float, dcr_avg: float) -> float:
    """
    MD = RSR_exact / (1 − DCR_avg).

    MD >> 1 confirms Theorem 1: high RSR coexists with acceptable DCR.
    """
    denominator = max(1.0 - dcr_avg, 1e-8)
    return float(rsr_exact / denominator)


# ═══════════════════════════════════════════════════════════════════════════════
# U1  Macro F1
# ═══════════════════════════════════════════════════════════════════════════════

def compute_macro_f1(model: nn.Module,
                     test_dataset,
                     batch_size: int = 512,
                     device: str = 'cpu') -> float:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return float(f1_score(all_labels, all_preds,
                          average='macro', zero_division=0))


# ═══════════════════════════════════════════════════════════════════════════════
# U2  RAR — Rare-Attack Recall
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rar(model:         nn.Module,
                test_dataset,
                rare_classes:  List[int],
                batch_size:    int = 512,
                device:        str = 'cpu') -> float:
    """
    Per-class recall averaged over the three rarest attack categories
    (paper U2).

    rare_classes : list of class integer labels identified as rarest.
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    recalls = []
    for c in rare_classes:
        mask = all_labels == c
        if mask.sum() == 0:
            continue
        recalls.append(float((all_preds[mask] == c).mean()))

    return float(np.mean(recalls)) if recalls else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# U3  FPR — False Positive Rate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fpr(model:        nn.Module,
                test_dataset,
                benign_class: int   = 0,
                batch_size:   int   = 512,
                device:       str   = 'cpu') -> float:
    """
    Fraction of benign flows misclassified as attacks (paper U3).
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    benign_mask = all_labels == benign_class
    if benign_mask.sum() == 0:
        return 0.0

    fp = ((all_preds[benign_mask] != benign_class)).sum()
    return float(fp / benign_mask.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: compute all metrics in one call
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    model:           nn.Module,
    generator:       nn.Module,
    discriminator:   nn.Module,
    X_train:         np.ndarray,
    X_test_arr:      np.ndarray,
    test_dataset,
    outlier_mask:    np.ndarray,
    rare_classes:    List[int],
    phi_encoder:     Optional[nn.Module] = None,
    latent_dim:      int   = 100,
    n_recon_queries: int   = 500,
    device:          str   = 'cpu'
) -> Dict[str, float]:
    """
    Compute all P1–P5 and U1–U3 metrics and return as a flat dict.
    """
    # ── generate synthetic pool for similarity metrics ────────────────────
    generator.eval()
    with torch.no_grad():
        z = torch.randn(5000, latent_dim, device=device)
        X_synth = generator(z).cpu().numpy()

    # ── run ReconSyn ──────────────────────────────────────────────────────
    reconstructed, targets = reconsyn_attack(
        generator, X_train, outlier_mask,
        latent_dim=latent_dim,
        n_queries=n_recon_queries,
        device=device
    )

    # ── similarity ────────────────────────────────────────────────────────
    dcr  = compute_dcr(X_train, X_synth)
    nndr = compute_nndr(X_train, X_synth)

    # ── privacy ───────────────────────────────────────────────────────────
    rsr  = compute_rsr(reconstructed, targets, phi_encoder=phi_encoder)
    rd   = compute_rd(reconstructed, targets)
    mi   = compute_mi_auc(generator, X_train, X_test_arr,
                          discriminator, latent_dim=latent_dim,
                          device=device)
    md   = compute_md(rsr['rsr_exact'], dcr)

    # ── utility ───────────────────────────────────────────────────────────
    macro_f1 = compute_macro_f1(model, test_dataset, device=device)
    rar      = compute_rar(model, test_dataset, rare_classes, device=device)
    fpr      = compute_fpr(model, test_dataset, device=device)

    return {
        # similarity
        'dcr':       dcr,
        'nndr':      nndr,
        # privacy
        'rsr_exact': rsr['rsr_exact'],
        'rsr_near':  rsr['rsr_near'],
        'rd':        rd,
        'mi_auc':    mi,
        'md':        md,
        # utility
        'macro_f1':  macro_f1,
        'rar':       rar,
        'fpr':       fpr
    }
