"""
dataloader.py
-------------
NetFlow dataset loader with Dirichlet(α) non-IID federated partitioning.

Supports:
  - IID uniform split
  - Non-IID Dirichlet(α) label-skew partitioning (paper 5.4)
  - Outlier record identification for RSR evaluation (paper 3.2)

Datasets supported: Toy (initial), TON_IoT, CSE_CIC_IDS-2018 (NetFlow format).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import random


# ── Feature columns (NetFlow format, Sarhan et al. 2021) ────────────────────
NETFLOW_FEATURES = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'TCP_FLAGS',
    'L7_PROTO', 'IN_BYTES', 'OUT_BYTES',
    'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS'
]

# Density threshold τ for outlier identification (paper Definition 1)
OUTLIER_DENSITY_THRESHOLD = 0.5   # normalised L2 distance


# ── Dataset ──────────────────────────────────────────────────────────────────
class NetFlowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


# ── Outlier identification ───────────────────────────────────────────────────
def identify_outliers(X: np.ndarray,
                      tau: float = OUTLIER_DENSITY_THRESHOLD) -> np.ndarray:
    """
    Return boolean mask of outlier records.

    A record x* is an outlier if its distance to its nearest neighbour
    in X exceeds τ (paper 3.2, Definition 1).

    Uses approximate batched computation to avoid O(N²) memory.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, metric='l2', algorithm='auto')
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    # dists[:,0] == 0 (self), dists[:,1] == nearest other record
    return dists[:, 1] > tau


# ── Dirichlet partitioning ───────────────────────────────────────────────────
def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition (X, y) into num_clients shards using Dir(α) label skew.

    Parameters
    ----------
    alpha : float
        Dirichlet concentration. Smaller → more heterogeneous.
        Paper evaluates α ∈ {0.1, 0.5, 1.0}.
    min_samples : int
        Minimum samples per client (re-draws if violated).

    Returns
    -------
    List of (X_client, y_client) arrays, length == num_clients.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    C = len(classes)

    # class-indexed lists of sample indices
    class_indices: Dict[int, np.ndarray] = {
        c: np.where(y == c)[0] for c in classes
    }

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        idx = class_indices[c].copy()
        rng.shuffle(idx)
        # Dir(α) proportions for this class across clients
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # ensure no client gets 0 samples from a class if possible
        proportions = np.maximum(proportions, 1e-6)
        proportions /= proportions.sum()

        splits = (proportions * len(idx)).astype(int)
        # fix rounding so sum == len(idx)
        splits[-1] = len(idx) - splits[:-1].sum()

        ptr = 0
        for client_id, n in enumerate(splits):
            client_indices[client_id].extend(idx[ptr: ptr + n].tolist())
            ptr += n

    # build arrays, enforce min_samples
    partitions = []
    for client_id in range(num_clients):
        ci = np.array(client_indices[client_id])
        if len(ci) < min_samples:
            # pad with random samples from the full training set
            extra = rng.choice(len(X), min_samples - len(ci), replace=False)
            ci = np.concatenate([ci, extra])
        partitions.append((X[ci], y[ci]))

    return partitions


# ── Main loader ──────────────────────────────────────────────────────────────
def create_federated_datasets(
    file_path: str,
    num_clients: int = 50,
    alpha: Optional[float] = None,         # None → IID
    iid: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
    feature_cols: Optional[List[str]] = None,
    label_col: str = 'Label',
    outlier_tau: float = OUTLIER_DENSITY_THRESHOLD
) -> Tuple[Dict[int, NetFlowDataset], NetFlowDataset, np.ndarray]:
    """
    Load a NetFlow CSV and produce federated client datasets.
    Currently used "Toy" dataset, replace it with the original datasets (TON_IoT, CSE_CIC_IDS-2018)

    Returns
    -------
    clients      : dict mapping client_id → NetFlowDataset
    test_dataset : NetFlowDataset (global test set)
    outlier_mask : bool array over training set — True == outlier record
                   (used by RSR evaluation)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(file_path)

    # ── feature selection ────────────────────────────────────────────────────
    if feature_cols is None:
        available = [c for c in NETFLOW_FEATURES if c in df.columns]
        feature_cols = available if available else [
            c for c in df.columns if c != label_col
        ]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    # encode string labels
    if y.dtype.kind not in ('i', 'u', 'f'):
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    y = y.astype(np.int64)

    # ── train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # ── normalisation ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── outlier mask (computed on normalised training set) ───────────────────
    outlier_mask = identify_outliers(X_train, tau=outlier_tau)

    # ── federated partitioning ───────────────────────────────────────────────
    clients: Dict[int, NetFlowDataset] = {}

    if iid or alpha is None:
        # IID: uniform random split
        idx = np.random.permutation(len(X_train))
        samples = len(X_train) // num_clients
        for c in range(num_clients):
            sel = idx[c * samples: (c + 1) * samples]
            clients[c] = NetFlowDataset(X_train[sel], y_train[sel])
    else:
        # Non-IID Dirichlet(α) partitioning (paper 5.4)
        partitions = dirichlet_partition(
            X_train, y_train,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed
        )
        for c, (Xc, yc) in enumerate(partitions):
            clients[c] = NetFlowDataset(Xc, yc)

    test_dataset = NetFlowDataset(X_test, y_test)
    return clients, test_dataset, outlier_mask


# ── Convenience DataLoader factory ──────────────────────────────────────────
def get_dataloader(dataset: NetFlowDataset,
                   batch_size: int = 256,
                   shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)
