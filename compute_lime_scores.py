#!/usr/bin/env python3
"""
compute_lime_scores.py

This script computes signed LIME word weights for hate speech classification
before and after fine-tuning, for either the DynaHate or HateCheck dataset.
Results are saved as pickle files. Supports CUDA, MPS (Mac), and CPU devices.
"""

import os
import random
import pickle
import logging
import warnings
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.datasets import create_hatecheck_dataset, create_dynahate_dataset

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BASE_MODEL_CHECKPOINT = "PY007/TinyLlama-1.1B-step-50K-105b"
POST_CHECKPOINTS = {
    "hatecheck": "./checkpoints/TinyLlama/hatecheck/best_checkpoint_33",
    "dynahate":   "./checkpoints/TinyLlama/dynahate/best_checkpoint_42",
}
CLASS_NAMES = ["nothate", "hate"]
RESULTS_DIR = "./results"

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Return the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        logger.info("CUDA available. Using GPU.")
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        logger.info("MPS available. Using Apple GPU.")
        return torch.device("mps")
    logger.info("Using CPU.")
    return torch.device("cpu")

def prepare_model_and_tokenizer(checkpoint: str, device: torch.device):
    """Load sequence classification model and tokenizer from checkpoint."""
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
    logger.info("Loaded model from %s onto %s", checkpoint, device)
    return model, tokenizer

def predict_proba(texts, model, tokenizer, device):
    """Return class probabilities for a list of texts."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    return F.softmax(logits, dim=-1).cpu().numpy()

def compute_lime_weights_signed(
    texts,
    tokenizer,
    explainer: LimeTextExplainer,
    model,
    device,
    n_samples: int = 500,
    num_features: int = 10,
    num_samples_lime: int = 500,
):
    """
    Aggregate signed LIME weights over a random subset of texts.
    Returns two lists of (word, normalized_weight): positive and negative.
    """
    n = min(n_samples, len(texts))
    logger.info("Computing LIME weights on %d samples", n)
    weights = defaultdict(float)
    indices = random.sample(range(len(texts)), n)

    for idx in tqdm(indices, desc="LIME weights"):
        text = texts.iloc[idx]
        probs = predict_proba([text], model, tokenizer, device)
        pred_label = int(np.argmax(probs, axis=1)[0])
        explanation = explainer.explain_instance(
            text,
            lambda xs: predict_proba(xs, model, tokenizer, device),
            num_features=num_features,
            labels=[pred_label],
            num_samples=num_samples_lime,
        )
        for word, weight in explanation.as_list(label=pred_label):
            weights[word] += weight

    # Normalize by vocabulary size
    vocab_size = len(tokenizer)
    for w in weights:
        weights[w] /= vocab_size

    pos = [(w, wt) for w, wt in weights.items() if wt > 0]
    neg = [(w, wt) for w, wt in weights.items() if wt < 0]
    pos.sort(key=lambda x: abs(x[1]), reverse=True)
    neg.sort(key=lambda x: abs(x[1]), reverse=True)

    return pos, neg

def main():
    parser = argparse.ArgumentParser(description="Compute LIME scores pre/post fine-tuning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hatecheck", action="store_true", help="Use the HateCheck dataset")
    group.add_argument("--dynahate",   action="store_true", help="Use the DynaHate dataset")
    args = parser.parse_args()

    # Select dataset, seed, post-checkpoint, column name, and sample count
    if args.hatecheck:
        dataset_name = "hatecheck"
        seed = 33
        dataset = create_hatecheck_dataset()
        text_col = "test_case"
        n_samples = len(dataset["test"])
    else:
        dataset_name = "dynahate"
        seed = 42
        dataset = create_dynahate_dataset()
        text_col = "text"
        n_samples = 1000

    set_seed(seed)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Prepare test texts
    df_test = dataset["test"].to_pandas()
    X_test = df_test[text_col]

    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    # Pre-fine-tuning
    logger.info("Computing pre-FT LIME weights for %s", dataset_name)
    model_pre, tok_pre = prepare_model_and_tokenizer(BASE_MODEL_CHECKPOINT, device)
    pos_pre, neg_pre = compute_lime_weights_signed(
        X_test, tok_pre, explainer, model_pre, device,
        n_samples=n_samples, num_features=10, num_samples_lime=500
    )
    with open(os.path.join(RESULTS_DIR, f"positive_pre_FT_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(pos_pre, f)
    with open(os.path.join(RESULTS_DIR, f"negative_pre_FT_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(neg_pre, f)
    logger.info("Saved pre-FT LIME weights for %s", dataset_name)

    # Post-fine-tuning
    post_checkpoint = POST_CHECKPOINTS[dataset_name]
    logger.info("Computing post-FT LIME weights for %s", dataset_name)
    model_post, tok_post = prepare_model_and_tokenizer(post_checkpoint, device)
    pos_post, neg_post = compute_lime_weights_signed(
        X_test, tok_post, explainer, model_post, device,
        n_samples=n_samples, num_features=10, num_samples_lime=500
    )
    with open(os.path.join(RESULTS_DIR, f"positive_post_FT_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(pos_post, f)
    with open(os.path.join(RESULTS_DIR, f"negative_post_FT_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(neg_post, f)
    logger.info("Saved post-FT LIME weights for %s", dataset_name)

if __name__ == "__main__":
    main()
