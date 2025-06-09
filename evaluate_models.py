#!/usr/bin/env python3
"""
evaluate_models.py

This script evaluates a hate speech classification model before and after fine-tuning
on either the DynaHate or HateCheck dataset. It logs final performance metrics
and generates comparison plots of pre- vs post-fine-tuning metrics only.
"""

import os
import random
import logging
import warnings
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.datasets import create_dynahate_dataset, create_hatecheck_dataset

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BASE_MODEL_CHECKPOINT = "PY007/TinyLlama-1.1B-step-50K-105b"
POST_CHECKPOINTS = {
    "dynahate": "./checkpoints/TinyLlama/dynahate/best_checkpoint_42",
    "hatecheck": "./checkpoints/TinyLlama/hatecheck/best_checkpoint_33",
}
CLASS_NAMES = ["nothate", "hate"]
RESULTS_ROOT = "./results"
PLOT_EVERY = 100

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def get_device() -> torch.device:
    """Return the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        logger.info("Using Apple MPS")
        return torch.device("mps")
    logger.info("Using CPU")
    return torch.device("cpu")


def prepare_model_and_tokenizer(checkpoint: str, device: torch.device):
    """Load a sequence classification model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=os.path.isdir(checkpoint))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(CLASS_NAMES),
        local_files_only=os.path.isdir(checkpoint)
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()
    logger.info("Loaded model from %s", checkpoint)
    return model, tokenizer


def set_seed(seed: int):
    """Configure random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def evaluate_model_inference(
    model,
    tokenizer,
    texts,
    labels,
    device: torch.device,
    plot_every: int = PLOT_EVERY
):
    """
    Run inference and collect metrics over time.
    Returns a dict of lists: accuracy_list, f1_list, precision_list,
    recall_list, auc_roc_list, true_labels_list, probabilities_list.
    """
    model.to(device)
    model.eval()

    preds, trues, probs = [], [], []
    accs, f1s, precs, recs, aucs = [], [], [], [], []

    for i, (text, true) in enumerate(tqdm(zip(texts, labels), total=len(texts), desc="Evaluating")):
        enc = tokenizer(text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512).to(device)
        with torch.no_grad():
            logits = model(**enc).logits

        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = int(torch.argmax(logits, dim=-1).item())

        preds.append(pred)
        trues.append(true)
        probs.append(prob)

        if (i + 1) % plot_every == 0:
            accs.append(accuracy_score(trues, preds))
            f1s.append(f1_score(trues, preds))
            precs.append(precision_score(trues, preds))
            recs.append(recall_score(trues, preds))
            try:
                aucs.append(roc_auc_score(trues, probs))
            except ValueError:
                aucs.append(np.nan)

    # final point
    accs.append(accuracy_score(trues, preds))
    f1s.append(f1_score(trues, preds))
    precs.append(precision_score(trues, preds))
    recs.append(recall_score(trues, preds))
    aucs.append(roc_auc_score(trues, probs))

    logger.info("Final Accuracy:   %.4f", accs[-1])
    logger.info("Final F1 Score:   %.4f", f1s[-1])
    logger.info("Final Precision:  %.4f", precs[-1])
    logger.info("Final Recall:     %.4f", recs[-1])
    logger.info("Final AUC-ROC:    %.4f", aucs[-1])

    return {
        "accuracy_list": accs,
        "f1_list": f1s,
        "precision_list": precs,
        "recall_list": recs,
        "auc_roc_list": aucs,
        "true_labels_list": trues,
        "probabilities_list": probs
    }


def plot_evaluation_metrics_comparison(
    results_pre,
    results_post,
    plot_every: int = PLOT_EVERY,
    save_plots: bool = True,
    save_dir: str = "plots/comparison"
):
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        ("accuracy_list", "Accuracy"),
        ("f1_list", "F1 Score"),
        ("precision_list", "Precision"),
        ("recall_list", "Recall"),
        ("auc_roc_list", "AUC-ROC"),
    ]

    x_pre = np.arange(plot_every, len(results_pre["accuracy_list"]) * plot_every + 1, plot_every)
    x_post = np.arange(plot_every, len(results_post["accuracy_list"]) * plot_every + 1, plot_every)

    # precision-recall
    pr_pre = precision_recall_curve(results_pre["true_labels_list"], results_pre["probabilities_list"])
    pr_post = precision_recall_curve(results_post["true_labels_list"], results_post["probabilities_list"])
    auc_pre = auc(pr_pre[1], pr_pre[0])
    auc_post = auc(pr_post[1], pr_post[0])

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(x_pre, results_pre[key], marker="o", linestyle="-", label="Pre FT")
        ax.plot(x_post, results_post[key], marker="o", linestyle="--", label="Post FT")
        ax.set_title(title)
        ax.set_xlabel("Samples processed")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(False)
        if save_plots:
            fig_single, ax_single = plt.subplots(figsize=(6, 4))
            ax_single.plot(x_pre, results_pre[key], marker="o", linestyle="-", label="Pre FT")
            ax_single.plot(x_post, results_post[key], marker="o", linestyle="--", label="Post FT")
            ax_single.set_title(title)
            ax_single.set_xlabel("Samples processed")
            ax_single.set_ylabel(title)
            ax_single.set_ylim(0, 1)
            ax_single.legend()
            fig_single.tight_layout()
            fig_single.savefig(os.path.join(save_dir, f"{title.lower().replace(' ', '_')}_comparison.png"))
            plt.close(fig_single)

    # PR curve
    ax = axes[-1]
    ax.plot(pr_pre[1], pr_pre[0], label=f"Pre FT (AUC={auc_pre:.4f})")
    ax.plot(pr_post[1], pr_post[0], linestyle="--", label=f"Post FT (AUC={auc_post:.4f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    if save_plots:
        fig_single, ax_single = plt.subplots(figsize=(6, 4))
        ax_single.plot(pr_pre[1], pr_pre[0], label=f"Pre FT (AUC={auc_pre:.4f})")
        ax_single.plot(pr_post[1], pr_post[0], linestyle="--", label=f"Post FT (AUC={auc_post:.4f})")
        ax_single.set_title("Precision-Recall Curve")
        ax_single.set_xlabel("Recall")
        ax_single.set_ylabel("Precision")
        ax_single.set_xlim(0, 1)
        ax_single.set_ylim(0, 1)
        ax_single.legend()
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(save_dir, "precision_recall_curve_comparison.png"))
        plt.close(fig_single)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre/post fine-tuning models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dynahate", action="store_true", help="Evaluate on DynaHate")
    group.add_argument("--hatecheck", action="store_true", help="Evaluate on HateCheck")
    args = parser.parse_args()

    if args.dynahate:
        name, seed = "dynahate", 42
        dataset = create_dynahate_dataset()
        text_key = "text"
    else:
        name, seed = "hatecheck", 33
        dataset = create_hatecheck_dataset()
        text_key = "test_case"

    set_seed(seed)
    device = get_device()

    out_dir = os.path.join(RESULTS_ROOT, name)
    comp_dir = os.path.join(out_dir, "comparison_plots")
    os.makedirs(comp_dir, exist_ok=True)

    df = dataset["test"].to_pandas()
    texts = df[text_key].tolist()
    labels = df["label"].astype(int).tolist()

    # Pre-FT evaluation
    model_pre, tok_pre = prepare_model_and_tokenizer(BASE_MODEL_CHECKPOINT, device)
    results_pre = evaluate_model_inference(model_pre, tok_pre, texts, labels, device)

    # Post-FT evaluation
    post_ckpt = POST_CHECKPOINTS[name]
    model_post, tok_post = prepare_model_and_tokenizer(post_ckpt, device)
    results_post = evaluate_model_inference(model_post, tok_post, texts, labels, device)

    # Comparison plots
    logger.info("Generating comparison plots")
    plot_evaluation_metrics_comparison(
        results_pre,
        results_post,
        save_plots=True,
        save_dir=comp_dir
    )


if __name__ == "__main__":
    main()
