"""
plots.py
--------
All visualization functions for the Disease Prediction project.
Generates: ROC curves, confusion matrices, feature importance,
           model comparison bar chart, class distribution.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

DISEASE_LABELS = {
    "heart": ("No Disease", "Disease"),
    "diabetes": ("No Diabetes", "Diabetes"),
    "breast_cancer": ("Malignant", "Benign"),
}


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved: %s", path)


# ── Class Distribution ────────────────────────────────────────────────────────

def plot_class_distribution(y, disease: str, output_dir: str):
    labels = DISEASE_LABELS.get(disease, ("Class 0", "Class 1"))
    counts = [int((y == 0).sum()), int((y == 1).sum())]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=PALETTE[:2], edgecolor="white", linewidth=1.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(cnt), ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Class Distribution — {disease.replace('_', ' ').title()}")
    ax.set_ylabel("Count")
    path = os.path.join(output_dir, "plots", f"{disease}_class_distribution.png")
    _savefig(fig, path)


# ── ROC Curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(results, disease: str, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r.y_test, r.y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{r.name} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {disease.replace('_', ' ').title()}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    path = os.path.join(output_dir, "plots", f"{disease}_roc_curves.png")
    _savefig(fig, path)


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(result, disease: str, output_dir: str):
    labels = DISEASE_LABELS.get(disease, ("Class 0", "Class 1"))
    cm = result.confusion
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", cbar=False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.4, str(cm[i, j]),
                    ha="center", va="center", fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
            ax.text(j + 0.5, i + 0.65, f"({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "gray")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {result.name}\n({disease.replace('_', ' ').title()})")

    path = os.path.join(output_dir, "plots", f"{disease}_confusion_{result.name.replace(' ', '_')}.png")
    _savefig(fig, path)


# ── Model Comparison ──────────────────────────────────────────────────────────

def plot_model_comparison(results, disease: str, output_dir: str):
    names = [r.name for r in results]
    metrics = {
        "Accuracy": [r.accuracy for r in results],
        "Precision": [r.precision for r in results],
        "Recall": [r.recall for r in results],
        "F1-Score": [r.f1 for r in results],
        "ROC-AUC": [r.roc_auc for r in results],
    }

    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, vals) in enumerate(metrics.items()):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=metric,
                      color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {disease.replace('_', ' ').title()}")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    path = os.path.join(output_dir, "plots", f"{disease}_model_comparison.png")
    _savefig(fig, path)


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(result, feature_names: list, disease: str, output_dir: str):
    """Works for Random Forest and XGBoost; skips others gracefully."""
    clf = result.pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        logger.debug("Skipping feature importance for %s (no feature_importances_)", result.name)
        return

    importances = clf.feature_importances_

    # Preprocessor may change feature count; handle gracefully
    n_features = len(importances)
    if len(feature_names) >= n_features:
        names = feature_names[:n_features]
    else:
        names = feature_names + [f"feat_{i}" for i in range(n_features - len(feature_names))]

    indices = np.argsort(importances)[::-1][:20]  # Top 20

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top Feature Importances — {result.name}\n({disease.replace('_', ' ').title()})")

    path = os.path.join(output_dir, "plots",
                        f"{disease}_feature_importance_{result.name.replace(' ', '_')}.png")
    _savefig(fig, path)


# ── CV Score Distribution ─────────────────────────────────────────────────────

def plot_cv_scores(results, disease: str, output_dir: str):
    names = [r.name for r in results]
    means = [r.cv_mean for r in results]
    stds = [r.cv_std for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim([max(0, min(means) - 0.1), 1.05])
    ax.set_ylabel("ROC-AUC (5-Fold CV)")
    ax.set_title(f"Cross-Validation ROC-AUC — {disease.replace('_', ' ').title()}")
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    path = os.path.join(output_dir, "plots", f"{disease}_cv_scores.png")
    _savefig(fig, path)


# ── Master Function ───────────────────────────────────────────────────────────

def generate_all_plots(disease_results, y_full, output_dir: str = "outputs"):
    """
    Generate all plots for a disease.
    Call after training.
    """
    disease = disease_results.disease
    results = disease_results.results
    feature_names = disease_results.feature_names

    logger.info("Generating plots for %s ...", disease)

    plot_class_distribution(y_full, disease, output_dir)
    plot_roc_curves(results, disease, output_dir)
    plot_model_comparison(results, disease, output_dir)
    plot_cv_scores(results, disease, output_dir)

    # Best model confusion matrix
    best = next(r for r in results if r.name == disease_results.best_model_name)
    plot_confusion_matrix(best, disease, output_dir)

    # Feature importance for tree-based models
    for r in results:
        plot_feature_importance(r, feature_names, disease, output_dir)

    logger.info("All plots generated for %s.", disease)
