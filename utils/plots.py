import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

def moving_std(data, window=3):
    """Compute moving std deviation with edge padding."""
    data = np.array(data)
    pad = window // 2
    padded = np.pad(data, (pad, pad), mode='edge')
    stds = [np.std(padded[i:i+window]) for i in range(len(data))]
    return np.array(stds)

def plot_evaluation_metrics(
    results,
    plot_every=100,
    save_plots=False,
    save_dir="plots/pre_FT",
    window=3
):
    """
    Plot evaluation metrics over time with moving std bands.
    Save each plot individually (HD), show all plots in 3x2 grid.
    Grid background is removed for cleaner plots.

    Args:
        results: dict with evaluation metrics.
        plot_every: int, samples between metric points.
        save_plots: bool, save plots if True.
        save_dir: str, folder to save plots.
        window: int, window size for moving std deviation.
    """

    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("accuracy_list", "Accuracy", "tab:blue"),
        ("f1_list", "F1 Score", "tab:orange"),
        ("precision_list", "Precision", "tab:green"),
        ("recall_list", "Recall", "tab:red"),
        ("auc_roc_list", "AUC-ROC", "tab:cyan")
    ]

    x_vals = np.arange(plot_every, len(results['accuracy_list']) * plot_every + 1, plot_every)

    # Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(results['true_labels_list'], results['probabilities_list'])
    pr_auc = auc(recall_curve, precision_curve)

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    for i, (key, title, color) in enumerate(metrics):
        ax = axes[i]
        vals = np.array(results[key])
        std_vals = moving_std(vals, window)

        ax.plot(x_vals, vals, marker='o', linestyle='-', color=color, label=title)
        ax.fill_between(x_vals, vals - std_vals, vals + std_vals, color=color, alpha=0.3)
        ax.set_xlabel("Processed Samples")
        ax.set_ylabel(title)
        ax.set_title(f"{title} Over Time")
        ax.set_ylim(0, 1)
        ax.grid(False)  # No grid background
        ax.legend()

        if save_plots:
            save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_pre_FT.png")
            fig_single, ax_single = plt.subplots(figsize=(8, 6))
            ax_single.plot(x_vals, vals, marker='o', linestyle='-', color=color, label=title)
            ax_single.fill_between(x_vals, vals - std_vals, vals + std_vals, color=color, alpha=0.3)
            ax_single.set_xlabel("Processed Samples")
            ax_single.set_ylabel(title)
            ax_single.set_title(f"{title} Over Time")
            ax_single.set_ylim(0, 1)
            ax_single.grid(False)
            ax_single.legend()
            fig_single.tight_layout()
            fig_single.savefig(save_path, dpi=300)
            plt.close(fig_single)

    # Precision-Recall Curve plot (last subplot)
    ax = axes[-1]
    ax.plot(recall_curve, precision_curve, color='tab:purple', linewidth=2, label=f'PR Curve (AUC={pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend()

    if save_plots:
        save_path = os.path.join(save_dir, "precision_recall_curve_pre_FT.png")
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        ax_single.plot(recall_curve, precision_curve, color='tab:purple', linewidth=2, label=f'PR Curve (AUC={pr_auc:.4f})')
        ax_single.set_xlabel('Recall')
        ax_single.set_ylabel('Precision')
        ax_single.set_title('Precision-Recall Curve')
        ax_single.set_xlim(0, 1)
        ax_single.set_ylim(0, 1)
        ax_single.grid(False)
        ax_single.legend()
        fig_single.tight_layout()
        fig_single.savefig(save_path, dpi=300)
        plt.close(fig_single)

    plt.tight_layout()
    plt.show()



# ────────────────────────────────────────────────────────────────────────────────
# PLOTTING FUNCTION (can be called repeatedly without recomputation)
# ────────────────────────────────────────────────────────────────────────────────
def plot_lime_words(
    positive_words,
    negative_words,
    *,
    top_x=20,
    palette_hate="Reds_r",
    palette_nothate="Greens_r",
    title_size=16,
):
    """
    Create side-by-side horizontal bar-plots for the two word lists.
    Shows only top_x words per class.

    Args:
        positive_words: list of tuples (word, normalized weight) for 'hate'
        negative_words: list of tuples (word, normalized weight) for 'nothate'
        top_x: int, number of top words to show per class
    """
    sns.set(style="whitegrid", context="talk", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Select top_x words
    pos_top = positive_words[:top_x]
    neg_top = negative_words[:top_x]

    # Plot for 'hate'
    if pos_top:
        w_pos, v_pos = zip(*pos_top)
        sns.barplot(
            x=list(v_pos),
            y=list(w_pos),
            hue=list(w_pos),          # avoids seaborn 0.14 palette warning
            dodge=False,
            palette=sns.color_palette(palette_hate, len(w_pos)),
            ax=axes[0],
            legend=False,
        )
        axes[0].set_title("Top words for class 'hate'", weight="bold", fontsize=title_size)
        axes[0].set_xlabel("Normalised aggregated weight")
        axes[0].set_ylabel("Words")
        axes[0].grid(False)
    else:
        axes[0].text(0.5, 0.5, "No positive words", ha="center", va="center")
        axes[0].axis("off")

    # Plot for 'nothate'
    if neg_top:
        w_neg, v_neg = zip(*neg_top)
        sns.barplot(
            x=list(v_neg),
            y=list(w_neg),
            hue=list(w_neg),
            dodge=False,
            palette=sns.color_palette(palette_nothate, len(w_neg)),
            ax=axes[1],
            legend=False,
        )
        axes[1].set_title("Top words for class 'nothate'", weight="bold", fontsize=title_size)
        axes[1].set_xlabel("Normalised aggregated weight")
        axes[1].set_ylabel("Words")
        axes[1].grid(False)
    else:
        axes[1].text(0.5, 0.5, "No negative words", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

def plot_evaluation_metrics_comparison(
    results_pre,
    results_post,
    plot_every=100,
    save_plots=False,
    save_dir="plots/comparison"
):
    """
    Plot and compare evaluation metrics over time for pre- and post-fine-tuning results.
    Saves each metric plot individually and shows all six plots in a 3x2 grid.

    Args:
        results_pre: dict of metrics before fine-tuning.
        results_post: dict of metrics after fine-tuning.
        plot_every: int, sample interval between metrics points.
        save_plots: bool, whether to save plots.
        save_dir: str, folder path to save individual plots.
    """

    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("accuracy_list", "Accuracy", "tab:blue"),
        ("f1_list", "F1 Score", "tab:orange"),
        ("precision_list", "Precision", "tab:green"),
        ("recall_list", "Recall", "tab:red"),
        ("auc_roc_list", "AUC-ROC", "tab:cyan")
    ]

    x_vals_pre = range(plot_every, len(results_pre['accuracy_list']) * plot_every + 1, plot_every)
    x_vals_post = range(plot_every, len(results_post['accuracy_list']) * plot_every + 1, plot_every)

    # Precision-Recall curve special case
    pr_curve_pre = precision_recall_curve(results_pre['true_labels_list'], results_pre['probabilities_list'])
    pr_auc_pre = auc(pr_curve_pre[1], pr_curve_pre[0])
    pr_curve_post = precision_recall_curve(results_post['true_labels_list'], results_post['probabilities_list'])
    pr_auc_post = auc(pr_curve_post[1], pr_curve_post[0])

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    # Plot main metrics
    for i, (key, title, color) in enumerate(metrics):
        ax = axes[i]
        ax.plot(x_vals_pre, results_pre[key], marker='o', linestyle='-', color=color, label='Pre Fine-tuning')
        ax.plot(x_vals_post, results_post[key], marker='o', linestyle='--', color='black', label='Post Fine-tuning')
        ax.set_xlabel("Processed Samples")
        ax.set_ylabel(title)
        ax.set_title(f"{title} Over Time")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        if save_plots:
            save_path = os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_comparison.png")
            fig_single, ax_single = plt.subplots(figsize=(8,6))
            ax_single.plot(x_vals_pre, results_pre[key], marker='o', linestyle='-', color=color, label='Pre Fine-tuning')
            ax_single.plot(x_vals_post, results_post[key], marker='o', linestyle='--', color='black', label='Post Fine-tuning')
            ax_single.set_xlabel("Processed Samples")
            ax_single.set_ylabel(title)
            ax_single.set_title(f"{title} Over Time")
            ax_single.set_ylim(0,1)
            ax_single.grid(True, linestyle='--', alpha=0.7)
            ax_single.legend()
            fig_single.tight_layout()
            fig_single.savefig(save_path)
            plt.close(fig_single)

    # Plot Precision-Recall Curve comparison
    ax = axes[-1]
    ax.plot(pr_curve_pre[1], pr_curve_pre[0], color='tab:purple', label=f'Pre FT PR Curve (AUC={pr_auc_pre:.4f})')
    ax.plot(pr_curve_post[1], pr_curve_post[0], color='black', linestyle='--', label=f'Post FT PR Curve (AUC={pr_auc_post:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    if save_plots:
        save_path = os.path.join(save_dir, "precision_recall_curve_comparison.png")
        fig_single, ax_single = plt.subplots(figsize=(8,6))
        ax_single.plot(pr_curve_pre[1], pr_curve_pre[0], color='tab:purple', label=f'Pre FT PR Curve (AUC={pr_auc_pre:.4f})')
        ax_single.plot(pr_curve_post[1], pr_curve_post[0], color='black', linestyle='--', label=f'Post FT PR Curve (AUC={pr_auc_post:.4f})')
        ax_single.set_xlabel('Recall')
        ax_single.set_ylabel('Precision')
        ax_single.set_title('Precision-Recall Curve')
        ax_single.set_xlim(0, 1)
        ax_single.set_ylim(0, 1)
        ax_single.grid(True, linestyle='--', alpha=0.7)
        ax_single.legend()
        fig_single.tight_layout()
        fig_single.savefig(save_path)
        plt.close(fig_single)

    plt.tight_layout()
    plt.show()
