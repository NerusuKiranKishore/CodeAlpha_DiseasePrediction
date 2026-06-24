"""
helpers.py
----------
Logging setup, directory creation, and other utilities.
"""

import os
import logging
import sys
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure root logger for both console and optional file output."""
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


def ensure_dirs(*dirs: str):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_summary_table(all_disease_results: list):
    """Print a final summary table across all diseases."""
    header = f"\n{'='*75}"
    header += f"\n{'FINAL SUMMARY — ALL DISEASES':^75}"
    header += f"\n{'='*75}"
    header += f"\n{'Disease':<18} {'Best Model':<22} {'ROC-AUC':>8} {'F1':>8} {'Accuracy':>10}"
    header += f"\n{'-'*75}"
    print(header)

    for dr in all_disease_results:
        best = next(r for r in dr.results if r.name == dr.best_model_name)
        print(f"{dr.disease.replace('_',' ').title():<18} {best.name:<22} "
              f"{best.roc_auc:>8.4f} {best.f1:>8.4f} {best.accuracy:>10.4f}")

    print("=" * 75)
    print(f"Outputs saved to: outputs/")
    print(f"Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
