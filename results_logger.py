""""
results_logger.py
-----------------
Centralised CSV results logger for all experiments.

Saves all privacy (P1-P5), similarity (DCR, NNDR), and utility (U1-U3)
metrics to structured CSV files under the results/ folder.

Output files
------------
results/
    - all_results.csv        -> one row per run (all methods, all configs)
    - privacy_results.csv    -> P1-P5 metrics only
    - utility_results.csv    -> U1-U3 metrics only
    - similarity_results.csv -> DCR, NNDR metrics only
    - training_history.csv   -> per-round training loss history

Each CSV row includes full experiment configuration as columns so results
are fully self-describing and reproducible.

Usage
-----
This module is NOT run directly. It is imported by train.py and evaluate.py:

    from results_logger import ResultsLogger
    logger = ResultsLogger(results_dir='results/')
    logger.log(config=vars(args), metrics=results, training_time=elapsed)
    logger.log_history(config=vars(args), history=history)
    logger.print_summary()
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Column definitions
# ============================================================

# Configuration columns (experiment identity)
CONFIG_COLS = [
    'timestamp',
    'method',
    'dataset',
    'alpha',
    'num_clients',
    'num_rounds',
    'local_epochs',
    'participation',
    'batch_size',
    'seed',
    # method-specific
    'lambda_arr',
    'margin',
    'k_g',
    'k_r',
    'embed_dim',
    'epsilon',
    'delta',
    'lambda_jac',
    'lambda_priv',
    'training_time_s',
]

# Privacy metric columns (P1-P5)
PRIVACY_COLS = [
    'rsr_exact',     # P1 exact-match RSR          ↓
    'rsr_near',      # P1 near-match RSR            ↓
    'rd',            # P2 Reconstruction Distance   ↑
    'mi_auc',        # P3 Membership Inference AUC  ↓
    'aqe',           # P4 Attack Query Efficiency   ↑
    'md',            # P5 Metric Divergence         ↓
]

# Similarity metric columns
SIMILARITY_COLS = [
    'dcr',           # Distance to Closest Record        ↑
    'nndr',          # Nearest-Neighbour Distance Ratio  ↑
]

# Utility metric columns (U1-U3)
UTILITY_COLS = [
    'macro_f1',      # U1 Macro F1           ↑
    'rar',           # U2 Rare-Attack Recall  ↑
    'fpr',           # U3 False Positive Rate ↓
]

ALL_METRIC_COLS = PRIVACY_COLS + SIMILARITY_COLS + UTILITY_COLS

# Training history columns
HISTORY_COLS = [
    'timestamp', 'method', 'dataset', 'alpha', 'num_clients', 'seed',
    'round', 'g_loss', 'd_loss', 'cls_loss', 'arr_loss', 'r_recon_loss',
    'epsilon',   # DP-SGD only
]


# ============================================================
# CSV helpers
# ============================================================

def _ensure_csv(path: Path, fieldnames: List[str]) -> None:
    """Create CSV with header row if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _append_row(path: Path, fieldnames: List[str], row: dict) -> None:
    """Append a single row to a CSV file."""
    _ensure_csv(path, fieldnames)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writerow(row)


# ============================================================
# Main logger class
# ============================================================

class ResultsLogger:

    def __init__(self, results_dir: str = 'results/'):
        self.root = Path(results_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        # file paths
        self.all_results_path  = self.root / 'all_results.csv'
        self.privacy_path      = self.root / 'privacy_results.csv'
        self.utility_path      = self.root / 'utility_results.csv'
        self.similarity_path   = self.root / 'similarity_results.csv'
        self.history_path      = self.root / 'training_history.csv'

        # column schemas
        self._all_cols        = CONFIG_COLS + ALL_METRIC_COLS
        self._privacy_cols    = CONFIG_COLS + PRIVACY_COLS
        self._utility_cols    = CONFIG_COLS + UTILITY_COLS
        self._similarity_cols = CONFIG_COLS + SIMILARITY_COLS

    # ----------------------------------------------------------
    # Log one experiment run
    # ----------------------------------------------------------
    def log(
        self,
        config:        Dict[str, Any],
        metrics:       Dict[str, float],
        training_time: float = 0.0
    ) -> None:
        """
        Save results of one experiment run to all CSV files.

        Called automatically by train.py and evaluate.py after each run.

        Parameters
        ----------
        config        : dict of experiment arguments — pass vars(args).
        metrics       : dict returned by evaluate_all().
        training_time : wall-clock training time in seconds.
        """
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {
            'timestamp':       ts,
            'method':          config.get('method',        ''),
            'dataset':         config.get('dataset',       ''),
            'alpha':           config.get('alpha',         ''),
            'num_clients':     config.get('num_clients',   ''),
            'num_rounds':      config.get('num_rounds',    ''),
            'local_epochs':    config.get('local_epochs',  ''),
            'participation':   config.get('participation', ''),
            'batch_size':      config.get('batch_size',    ''),
            'seed':            config.get('seed',          ''),
            'lambda_arr':      config.get('lambda_arr',    ''),
            'margin':          config.get('margin',        ''),
            'k_g':             config.get('k_g',           ''),
            'k_r':             config.get('k_r',           ''),
            'embed_dim':       config.get('embed_dim',     ''),
            'epsilon':         config.get('epsilon',       ''),
            'delta':           config.get('delta',         ''),
            'lambda_jac':      config.get('lambda_jac',    ''),
            'lambda_priv':     config.get('lambda_priv',   ''),
            'training_time_s': round(training_time, 2),
        }

        # add metric values (rounded to 6 decimal places)
        for col in ALL_METRIC_COLS:
            row[col] = round(float(metrics.get(col, float('nan'))), 6)

        # write to all four CSV files
        _append_row(self.all_results_path,  self._all_cols,        row)
        _append_row(self.privacy_path,      self._privacy_cols,    row)
        _append_row(self.utility_path,      self._utility_cols,    row)
        _append_row(self.similarity_path,   self._similarity_cols, row)

        print(f"  Results appended to:")
        print(f"    {self.all_results_path}")
        print(f"    {self.privacy_path}")
        print(f"    {self.utility_path}")
        print(f"    {self.similarity_path}")

    # ----------------------------------------------------------
    # Log per-round training history
    # ----------------------------------------------------------
    def log_history(
        self,
        config:  Dict[str, Any],
        history: List[Dict[str, float]]
    ) -> None:
        """
        Save per-round training loss history to training_history.csv.

        Called automatically by train.py after federated_train() completes.

        Parameters
        ----------
        config  : dict of experiment arguments — pass vars(args).
        history : list of per-round loss dicts from federated_train().
        """
        _ensure_csv(self.history_path, HISTORY_COLS)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        base = {
            'timestamp':   ts,
            'method':      config.get('method',      ''),
            'dataset':     config.get('dataset',     ''),
            'alpha':       config.get('alpha',       ''),
            'num_clients': config.get('num_clients', ''),
            'seed':        config.get('seed',        ''),
        }

        with open(self.history_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_COLS,
                                    extrasaction='ignore')
            for entry in history:
                writer.writerow({**base, **entry})

        print(f"  Training history appended to: {self.history_path}")

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    def print_summary(self, n_rows: int = 10) -> None:
        """Print the last n_rows of all_results.csv as a summary table."""
        if not self.all_results_path.exists():
            print("  No results file found.")
            return

        with open(self.all_results_path, 'r') as f:
            rows = list(csv.DictReader(f))

        if not rows:
            print("  Results file is empty.")
            return

        recent = rows[-n_rows:]

        header = (
            f"{'method':<12} {'dataset':<10} {'alpha':<6} {'N':<6} "
            f"{'seed':<6} {'RSR↓':<8} {'MI-AUC↓':<10} "
            f"{'F1↑':<8} {'RAR↑':<8}"
        )
        print(f"\n{'─' * 70}")
        print(f"  Recent Results (last {len(recent)} rows)")
        print(f"{'─' * 70}")
        print(f"  {header}")
        print(f"  {'─' * 68}")

        for r in recent:
            line = (
                f"  {r.get('method',      ''):<12}"
                f"{r.get('dataset',     ''):<10}"
                f"{r.get('alpha',       ''):<6}"
                f"{r.get('num_clients', ''):<6}"
                f"{r.get('seed',        ''):<6}"
                f"{r.get('rsr_exact',   ''):<8}"
                f"{r.get('mi_auc',      ''):<10}"
                f"{r.get('macro_f1',    ''):<8}"
                f"{r.get('rar',         ''):<8}"
            )
            print(line)

        print(f"{'─' * 70}\n")