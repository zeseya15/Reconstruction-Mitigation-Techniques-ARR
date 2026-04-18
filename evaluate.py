"""
evaluate.py
-----------
Standalone evaluation script for saved model checkpoints.

Computes all privacy (P1-P5) and utility (U1-U3) metrics
reported in the paper on a saved FL-GAN checkpoint.

Usage
-----
python scripts/evaluate.py \
    --checkpoint results/arr_ton_iot_alpha0.1_N50_seed42.pt \
    --dataset ton_iot \
    --alpha 0.1 \
    --num_clients 50 \
    --method arr
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model          import BiLSTMTCNGAN, LATENT_DIM
from dataloader     import create_federated_datasets, get_dataloader
from arr            import VIMEEncoder
from metrics        import evaluate_all
from results_logger import ResultsLogger


# ── Dataset registry ─────────────────────────────────────────────────────────
DATASET_FILES = {
    'toy': 'toy.csv',
    'ton_iot': 'ton_iot_netflow.csv',
    'cic_ids': 'cic_ids_2018_netflow.csv'
}


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate saved FL-GAN checkpoint')
    p.add_argument('--checkpoint',  type=str, required=True,
                   help='Path to saved .pt checkpoint')
    p.add_argument('--dataset',     type=str, default='toy',
                   choices=['toy', 'ton_iot', 'cic_ids'])
    p.add_argument('--data_path',   type=str, default='data/')
    p.add_argument('--alpha',       type=float, default=0.1)
    p.add_argument('--num_clients', type=int,   default=50)
    p.add_argument('--method',      type=str,   default='arr',
                   choices=['flgan', 'arr', 'dpsgd', 'jacobian', 'privgan'])
    p.add_argument('--embed_dim',   type=int,   default=64)
    p.add_argument('--n_recon_queries', type=int, default=500)
    p.add_argument('--rare_classes', type=int, nargs='+', default=None)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--device',      type=str,   default='cpu')
    p.add_argument('--output_dir',  type=str,   default='results/')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # ── load checkpoint ───────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # infer dimensions from checkpoint
    input_dim   = ckpt.get('input_dim',   10)
    num_classes = ckpt.get('num_classes', 2)

    model = BiLSTMTCNGAN(input_dim=input_dim,
                          classes=num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── load data ─────────────────────────────────────────────────────────
    fname = DATASET_FILES[args.dataset]
    data_path = os.path.join(args.data_path, fname)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    client_datasets, test_dataset, outlier_mask = create_federated_datasets(
        file_path   = data_path,
        num_clients = args.num_clients,
        alpha       = args.alpha,
        seed        = args.seed
    )

    X_train_all = np.vstack([ds.x.numpy() for ds in client_datasets.values()])
    X_test_all  = test_dataset.x.numpy()

    # ── VIME encoder (ARR only) ───────────────────────────────────────────
    phi_encoder = None
    if args.method == 'arr':
        phi_encoder = VIMEEncoder(input_dim=input_dim,
                                   embed_dim=args.embed_dim).to(device)
        if 'phi_state_dict' in ckpt:
            phi_encoder.load_state_dict(ckpt['phi_state_dict'])
            phi_encoder.eval()
            for p in phi_encoder.parameters():
                p.requires_grad_(False)
        else:
            print("  Warning: phi_state_dict not found in checkpoint. "
                  "Re-training VIME encoder.")
            all_X = torch.cat([ds.x for ds in client_datasets.values()])
            phi_encoder.pretrain(all_X, epochs=20, device=device)

    # ── rare classes ──────────────────────────────────────────────────────
    if args.rare_classes is None:
        from collections import Counter
        counts: Counter = Counter()
        for ds in client_datasets.values():
            loader = get_dataloader(ds, batch_size=len(ds), shuffle=False)
            _, labels = next(iter(loader))
            counts.update(labels.numpy().tolist())
        attack_classes = {k: v for k, v in counts.items() if k != 0}
        rare_classes = sorted(attack_classes,
                               key=lambda k: attack_classes[k])[:3]
    else:
        rare_classes = args.rare_classes

    print(f"  Rare attack classes: {rare_classes}")
    print(f"  Outlier records:     {outlier_mask.sum()}")
    print(f"  Running evaluation  ...")

    # ── evaluate ──────────────────────────────────────────────────────────
    results = evaluate_all(
        model            = model,
        generator        = model.generator,
        discriminator    = model.discriminator,
        X_train          = X_train_all,
        X_test_arr       = X_test_all,
        test_dataset     = test_dataset,
        outlier_mask     = outlier_mask,
        rare_classes     = rare_classes,
        phi_encoder      = phi_encoder,
        latent_dim       = LATENT_DIM,
        n_recon_queries  = args.n_recon_queries,
        device           = device
    )

    # ── print ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Evaluation Results")
    print(f"{'─'*50}")
    for k, v in results.items():
        print(f"  {k:<15}: {v:.4f}")
    print(f"{'─'*50}\n")

    # ── save ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── (A) CSV logging ───────────────────────────────────────────────────
    logger = ResultsLogger(results_dir=str(out_dir))
    logger.log(
        config        = vars(args),
        metrics       = results,
        training_time = 0.0        # evaluation only — no training time
    )

    # ── (B) Full per-run record ────────────────────────────────────
    stem     = Path(args.checkpoint).stem
    out_path = out_dir / f"eval_{stem}.json"
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    print(f"  Full evaluation record saved to {out_path}\n")

    # ── print summary ─────────────────────────────────────────────────────
    logger.print_summary(n_rows=5)


if __name__ == '__main__':
    main()
