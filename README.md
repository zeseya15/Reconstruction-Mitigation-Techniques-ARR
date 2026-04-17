Mitigating Reconstruction Attacks: ARR
Overview
This repository provides the full implementation of:

# ARR — Adversarial Reconstruction Regularizer (proposed reconstruction-aware mechanism)
# FL-GAN — Undefended FedTSRGNet baseline
# DP-SGD — Differential Privacy baseline (Opacus)
# Jacobian Regularisation — Output-smoothness baseline
# privGAN — Membership-privacy baseline
# ReconSyn — Black-box reconstruction attack (Ganev & De Cristofaro, 2023)
# All evaluation metrics — RSR, RD, MI-AUC, AQE, MD, Macro-F1, RAR, FPR


Repository Structure
├── src/
│   ├── model.py          # BiLSTMTCNGAN (FedTSRGNet / FL-GAN backbone)
│   ├── arr.py            # ARR: VIMEEncoder, ProxyReconstructor, ARRTrainer
│   ├── baselines.py      # FL-GAN, DP-SGD, Jacobian Reg, privGAN trainers
│   ├── fedavg.py         # FedAvg aggregation + federated training loop
│   ├── dataloader.py     # NetFlow loader + Dirichlet non-IID partitioning
│   ├── metrics.py        # All privacy and utility metrics
│   └── regularizers.py   # Standalone regularizer utilities
├── scripts/
│   ├── train.py          # Main training entry point (all methods)
│   ├── evaluate.py       # Standalone evaluation on saved checkpoints
│   └── run_all.sh        # Reproduce all 24 evaluation configurations
├── data/
│   └── README.md         # Dataset download instructions
├── results/              # Output directory (created at runtime)
├── requirements.txt      # Python dependencies
└── README.md

1. Install dependencies
bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate fgcad-arr

2. Download datasets
See #data/README.mdfor download instructions.

Supported datasets:
TON_IoT — https://research.unsw.edu.au/projects/toniot-datasets
CSE-CIC-IDS2018 — https://www.unb.ca/cic/datasets/ids-2018.html

Place downloaded CSV files in data/:
data/
├── ton_iot_netflow.csv
└── cic_ids_2018_netflow.csv

3. Run experiments
bash
# ARR (proposed) — TON_IoT, α=0.1, N=50
python scripts/train.py --dataset ton_iot --method arr --alpha 0.1 --num_clients 50

# FL-GAN baseline
python scripts/train.py --dataset ton_iot --method flgan --alpha 0.1 --num_clients 50

# DP-SGD baseline (ε=1, δ=1e-5)
python scripts/train.py --dataset ton_iot --method dpsgd --alpha 0.1 --num_clients 50

# Jacobian regularisation
python scripts/train.py --dataset ton_iot --method jacobian --alpha 0.1 --num_clients 50

# privGAN
python scripts/train.py --dataset ton_iot --method privgan --alpha 0.1 --num_clients 50


4. Reproduce all paper results
bash
# bash scripts/run_all.sh

This runs all 24 evaluation configurations (3 α × 4 N × 2 datasets) × 5 seeds for all methods.
