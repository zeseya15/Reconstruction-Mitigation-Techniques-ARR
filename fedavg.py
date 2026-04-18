"""
fedavg.py
---------
FedAvg aggregation and federated training orchestration.

Implements:
  - fedavg()              — standard weighted FedAvg (McMahan et al., 2017)
  - federated_train()     — full FL training loop supporting all baselines
                            and ARR (Algorithm 1, paper 4.1)

Client participation fraction ρ = 0.5 (paper 5.3).
T = 100 rounds, E = 5 local steps per round.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BiLSTMTCNGAN, LATENT_DIM


# ═══════════════════════════════════════════════════════════════════════════════
# FedAvg weight aggregation
# ═══════════════════════════════════════════════════════════════════════════════
def fedavg(global_model: nn.Module,
           client_models: List[nn.Module],
           weights: Optional[List[float]] = None) -> nn.Module:
    """
    Weighted FedAvg aggregation.

    Parameters
    ----------
    global_model  : nn.Module  — updated in-place and returned.
    client_models : list of local models after their training round.
    weights       : per-client data fractions n_i/n.
                    If None, uniform weighting is used.
    """
    if not client_models:
        return global_model

    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)

    weights_t = torch.tensor(weights, dtype=torch.float32)
    weights_t = weights_t / weights_t.sum()   # normalise

    global_dict = global_model.state_dict()

    for k in global_dict:
        stacked = torch.stack(
            [m.state_dict()[k].float() for m in client_models], dim=0
        )                                                   # (C, *shape)
        global_dict[k] = (stacked * weights_t.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)

    global_model.load_state_dict(global_dict)
    return global_model


# ═══════════════════════════════════════════════════════════════════════════════
# Federated training loop
# ═══════════════════════════════════════════════════════════════════════════════
def federated_train(
    global_model:   BiLSTMTCNGAN,
    client_datasets: Dict[int, Any],        # client_id → NetFlowDataset
    trainer_class,                           # FLGANTrainer | ARRTrainer | ...
    trainer_kwargs: dict,
    num_rounds:     int   = 100,
    local_epochs:   int   = 5,
    participation:  float = 0.5,
    batch_size:     int   = 256,
    device:         str   = 'cpu',
    seed:           int   = 42,
    verbose:        bool  = True
) -> Tuple[BiLSTMTCNGAN, List[dict]]:
    """
    Federated training loop (Algorithm 1, paper 4.1 / 5.3).

    Parameters
    ----------
    global_model     : shared BiLSTMTCNGAN initialised before federation.
    client_datasets  : dict of per-client datasets.
    trainer_class    : one of FLGANTrainer, DPSGDTrainer,
                       JacobianRegTrainer, PrivGANTrainer, or ARRTrainer.
    trainer_kwargs   : keyword arguments forwarded to trainer_class.__init__
                       (excluding model/generator/discriminator which are
                       set per-client from the global model copy).
    num_rounds       : T   (paper: 100)
    local_epochs     : E   (paper: 5)  — passed to trainer as k_g / epochs.
    participation    : ρ   (paper: 0.5)
    batch_size       : mini-batch size.
    device           : 'cpu' or 'cuda'.
    seed             : random seed for client sampling.

    Returns
    -------
    global_model : final trained model.
    history      : list of per-round loss dicts.
    """
    rng = random.Random(seed)
    client_ids = list(client_datasets.keys())
    N = len(client_ids)
    history: List[dict] = []

    # dataset sizes for weighted aggregation
    dataset_sizes = {cid: len(client_datasets[cid]) for cid in client_ids}
    total_size = sum(dataset_sizes.values())

    global_model.to(device)

    for t in range(1, num_rounds + 1):
        # ── client selection (ρ fraction) ────────────────────────────────
        k = max(1, int(participation * N))
        selected = rng.sample(client_ids, k)

        local_models: List[nn.Module] = []
        round_losses: List[dict]      = []
        round_weights: List[float]    = []

        for cid in selected:
            # ── give client a copy of global weights ──────────────────────
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            # ── instantiate trainer with local model ──────────────────────
            # ARRTrainer takes generator/discriminator/phi_encoder separately
            # all other trainers take model directly
            kw = dict(trainer_kwargs)   # shallow copy
            kw['device'] = device

            from arr import ARRTrainer
            if trainer_class is ARRTrainer:
                trainer = trainer_class(
                    generator     = local_model.generator,
                    discriminator = local_model.discriminator,
                    input_dim     = local_model.input_dim,
                    latent_dim    = LATENT_DIM,
                    **kw
                )
            else:
                trainer = trainer_class(model=local_model, **kw)

            # ── local training for E steps ────────────────────────────────
            for _ in range(local_epochs):
                losses = trainer.train_one_round(
                    client_datasets[cid],
                    batch_size=batch_size
                )

            round_losses.append(losses)
            local_models.append(local_model)
            round_weights.append(dataset_sizes[cid] / total_size)

        # ── aggregate ─────────────────────────────────────────────────────
        selected_weights = [dataset_sizes[cid] / total_size
                            for cid in selected]
        # re-normalise to selected clients
        sw_sum = sum(selected_weights)
        selected_weights = [w / sw_sum for w in selected_weights]

        global_model = fedavg(global_model, local_models, selected_weights)

        # ── logging ───────────────────────────────────────────────────────
        avg_loss = {k: float(np.mean([l.get(k, 0) for l in round_losses]))
                    for k in round_losses[0]}
        avg_loss['round'] = t
        history.append(avg_loss)

        if verbose and (t % 10 == 0 or t == 1):
            loss_str = '  '.join(f"{k}={v:.4f}"
                                 for k, v in avg_loss.items()
                                 if k != 'round')
            print(f"[Round {t:3d}/{num_rounds}]  {loss_str}")

    return global_model, history
