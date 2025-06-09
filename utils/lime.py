import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

def predict_proba_gpu(texts, model, tokenizer, device):
    """
    Predict class probabilities for a list of texts using the model on GPU.

    Args:
        texts: List of raw text strings.
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        device: torch.device.

    Returns:
        Numpy array of shape (len(texts), num_classes) with predicted probabilities.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs


def lime_explain_single_instance(
    text,
    model,
    tokenizer,
    device,
    explainer,
    class_names,
    num_features=10,
    num_samples=500,
    show_in_notebook=False,
):
    """
    Explain a single text prediction with LIME.

    Args:
        text: string, raw input text to explain.
        model: HF model.
        tokenizer: HF tokenizer.
        device: torch.device.
        explainer: LimeTextExplainer instance.
        class_names: list of class names (str).
        num_features: number of words to show in explanation.
        num_samples: number of perturbed samples for LIME.
        show_in_notebook: bool, whether to show LIME inline plot (Jupyter).

    Returns:
        exp: LIME explanation object.
        pred_label: predicted class index (int).
        probs: predicted class probabilities (numpy array).
    """
    probs = predict_proba_gpu([text], model, tokenizer, device)
    pred_label = int(probs.argmax(axis=1)[0])

    exp = explainer.explain_instance(
        text,
        lambda x: predict_proba_gpu(x, model, tokenizer, device),
        num_features=num_features,
        labels=[pred_label],
        num_samples=num_samples
    )

    if show_in_notebook:
        exp.show_in_notebook(text=True)

    return exp, pred_label, probs


from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _min_max(values: np.ndarray) -> np.ndarray:
    """Min–max normalise an array to the [0, 1] interval."""
    v_min, v_max = values.min(), values.max()
    return np.ones_like(values) if v_max == v_min else (values - v_min) / (v_max - v_min)


# ────────────────────────────────────────────────────────────────────────────────
# DATA-GENERATION FUNCTION (no plotting)
# ────────────────────────────────────────────────────────────────────────────────
def compute_lime_weights_signed(
    X_test,
    tokenizer,
    explainer,
    predict_proba_gpu,
    *,
    n_samples: int = 500,
    num_features: int = 10,
    num_samples_lime: int = 500,
    top_x: int = 20,
):
    """
    Aggregate signed LIME word-weights over *n_samples* random test texts.

    Returns
    -------
    positive_words : list[tuple[str, float]]
        (word, weight) pushing towards 'hate'  – normalised weight ∈ [0, 1].
    negative_words : list[tuple[str, float]]
        (word, weight) pushing towards 'nothate' – normalised weight ∈ [0, 1].
    """
    weights = defaultdict(float)
    indices = random.sample(range(len(X_test)), n_samples)

    for idx in tqdm(indices, desc="Computing LIME weights"):
        text = X_test.iloc[idx]
        pred_label = int(predict_proba_gpu([text]).argmax(axis=1)[0])

        exp = explainer.explain_instance(
            text,
            predict_proba_gpu,
            num_features=num_features,
            labels=[pred_label],
            num_samples=num_samples_lime,
        )

        for word, weight in exp.as_list(label=pred_label):
            weights[word] += weight  # accumulate (signed)

    # Normalise by vocabulary size
    vocab_len = len(tokenizer)
    for w in weights:
        weights[w] /= vocab_len

    # Split, order, min-max normalise and truncate
    pos_raw = [(w, wt) for w, wt in weights.items() if wt > 0]
    neg_raw = [(w, wt) for w, wt in weights.items() if wt < 0]

    pos_raw.sort(key=lambda x: x[1], reverse=True)
    neg_raw.sort(key=lambda x: abs(x[1]), reverse=True)

    pos_top = pos_raw[:top_x]
    neg_top = neg_raw[:top_x]

    if pos_top:
        pos_vals = np.array([wt for _, wt in pos_top])
        pos_norm = _min_max(pos_vals)
        positive_words = [(w, v) for (w, _), v in zip(pos_top, pos_norm)]
    else:
        positive_words = []

    if neg_top:
        neg_vals = np.array([abs(wt) for _, wt in neg_top])
        neg_norm = _min_max(neg_vals)
        negative_words = [(w, v) for (w, _), v in zip(neg_top, neg_norm)]
    else:
        negative_words = []

    return positive_words, negative_words
