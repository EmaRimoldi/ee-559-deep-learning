# ---------------------------------------------------------------------
# Plot class balance (0 = nothate, 1 = hate)
# ---------------------------------------------------------------------
# src/utils/visualisations.py
from __future__ import annotations          # ← keep if you like, harmless
from typing import Optional                 # ← add this import
import matplotlib.pyplot as plt
from datasets import DatasetDict
from collections import OrderedDict

__all__ = ["plot_class_balance"]

def plot_class_balance(
    ds_dict: DatasetDict,
    name: Optional[str] = None              # ← Optional[...] instead of |
) -> None:
    """
    Bar-plot the number of examples for label 0 and label 1
    in each split (*train*, *validation*, *test*).

    Parameters
    ----------
    ds_dict : DatasetDict
        Hugging Face DatasetDict with at least one of the three canonical splits.
    name : str, optional
        Dataset name – shown in the figure title (e.g. "HateEval", "DynaHate").
    """
    # Decide which splits are actually present (keep canonical order)
    split_order = OrderedDict.fromkeys(("train", "validation", "test"))
    splits = [s for s in split_order if s in ds_dict]

    if not splits:
        raise ValueError("DatasetDict contains none of the standard splits.")

    # Count label occurrences
    def _count(cls: int, split: str) -> int:
        return ds_dict[split].to_pandas()["label"].value_counts().get(cls, 0)

    counts_0 = [_count(0, s) for s in splits]
    counts_1 = [_count(1, s) for s in splits]

    # Plot
    width = 0.35
    x = range(len(splits))

    plt.figure()
    plt.bar([i - width / 2 for i in x], counts_0, width,
            label="nothate (0)", color="#9999ff")
    plt.bar([i + width / 2 for i in x], counts_1, width,
            label="hate (1)",    color="#ff9999")

    plt.xticks(list(x), splits)
    plt.ylabel("Number of examples")
    title = "Class Balance Across Splits"
    if name:
        title += f" – {name}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
