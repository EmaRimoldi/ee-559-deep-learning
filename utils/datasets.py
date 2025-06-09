# src/utils/datasets.py

# This must be at the very beginning of the file
from __future__ import annotations

# Now you can import other modules
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final
import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
import pandas as pd

try:
    from datasets import Dataset, DatasetDict
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The `datasets` library is required. Install it with:\n"
        "    pip install datasets"
    ) from _e


# -----------------------------------------------------------------------------
# DynaHate ─ download
# -----------------------------------------------------------------------------
_REPO_DYNAHATE: Final = (
    "https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset.git"
)
_DYNA_FILE: Final = "Dynamically Generated Hate Dataset v0.2.3.csv"


def download_dynahate(base_dir: str | Path = "data") -> Path:
    """Download DynaHate v0.2.3 into `<base_dir>/DynaHate/`."""
    dataset_dir = Path(base_dir).resolve() / "DynaHate"
    clone_root = dataset_dir / "_clone_sparse"
    if clone_root.exists():
        shutil.rmtree(clone_root)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            _REPO_DYNAHATE,
            str(clone_root),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_root), "sparse-checkout", "set", "."],
        check=True,
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    src = clone_root / _DYNA_FILE
    dst = dataset_dir / "dynahate_v0.2.3.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)
    print(f"✓ copied {src} → {dst}")

    shutil.rmtree(clone_root, ignore_errors=True)
    print("DynaHate file saved in:", dataset_dir)
    return dataset_dir


# -------------------------------------------------------------------------
# DynaHate ─ prepare DatasetDict (with on-the-fly binarisation)
# -------------------------------------------------------------------------
def _split_and_binarise_dynahate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Return three DataFrames (train/validation/test) with binary label column.

    • The original CSV contains columns: text, label (str), split (train/dev/test).
    • We map   'hate' -> 1   and everything else -> 0.
    """
    df = df[["text", "label", "split"]].copy()

    # Binarise textual labels
    df["label"] = (df["label"].str.lower() == "hate").astype(int)

    # Rename 'dev' split → 'validation'
    df["split"] = df["split"].map(
        {"train": "train", "dev": "validation", "test": "test"}
    )

    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        sub = (
            df[df["split"] == split]
            .drop(columns="split")
            .reset_index(drop=True)
        )
        out[split] = sub
    return out


def create_dynahate_dataset(base_dir: str | Path = "data") -> DatasetDict:
    """
    Build a Hugging Face `DatasetDict` from the DynaHate CSV in
    ``<base_dir>/DynaHate/`` **with binary labels (0 = nothate, 1 = hate)**.

    Returns
    -------
    DatasetDict
        Keys: 'train', 'validation', 'test'.
        Columns per split: 'text' (str) and 'label' (int 0/1).
    """
    csv_path = Path(base_dir).resolve() / "DynaHate" / "dynahate_v0.2.3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    splits = _split_and_binarise_dynahate(df)

    ds_dict = {
        split: Dataset.from_pandas(sub_df) for split, sub_df in splits.items()
    }
    return DatasetDict(ds_dict)


# -------------------------------------------------------------------------
# Pretty console summary for a DatasetDict
# -------------------------------------------------------------------------
from textwrap import indent
from datasets import DatasetDict

__all__ = []  # Define __all__ before appending to it
__all__.append("describe_dataset")


def _fmt_row(split: str, ds) -> str:
    n_examples = len(ds)
    columns    = ", ".join(ds.column_names)
    sample     = indent(str(ds[0]), " " * 8) if n_examples else "        (empty)"
    return (
        f"• {split:<10}  {n_examples:>6,} rows   "
        f"[columns: {columns}]\n"
        f"{sample}\n"
    )


def describe_dataset(ds_dict: DatasetDict, name: str | None = None) -> None:
    """
    Print a concise, nicely-formatted overview of a DatasetDict.

    Shows:
      – dataset name (optional)
      – rows / columns for each split
      – the first record of every split
    """
    title = f"Dataset structure – {name}" if name else "Dataset structure"
    bar   = "═" * len(title)
    print(f"{bar}\n{title}\n{bar}")

    for split in ("train", "validation", "test"):
        if split in ds_dict:
            print(_fmt_row(split, ds_dict[split]))
        else:
            print(f"• {split:<10}  (missing split)\n")


# import pandas as pd
# from datasets import Dataset, DatasetDict
# from sklearn.model_selection import train_test_split
# from pathlib import Path

# def _split_and_binarise_hatecheck(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
#     """
#     This function splits the input DataFrame into train, validation, and test sets,
#     and binarizes the 'label_gold' column into a 'label' column (0 for non-hateful, 1 for hateful).

#     The 'split' column is added to indicate the data split (train, validation, or test).

#     • The input DataFrame contains columns: 'test_case' (tweet text) and 'label_gold' (label indicating whether the tweet is hateful or not).
#     • 'label_gold' values are mapped: 'hateful' -> 1, 'non-hateful' -> 0.
#     • The data is split into 70% for training, 10% for validation, and 20% for testing.
#     """
    
#     # Binarize the 'label_gold' column: convert 'hateful' to 1 and 'non-hateful' to 0
#     df["label"] = (df["label_gold"].str.lower() == "hateful").astype(int)
    
#     # Drop the 'label_gold' column as it's no longer needed
#     df = df.drop(columns=["label_gold"])

#     # First split: 80% train, 20% test
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

#     # Second split: 90% train (from original), 10% validation (from the train set)
#     train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])

#     # Assign the appropriate split labels
#     train_df["split"] = "train"
#     val_df["split"] = "validation"
#     test_df["split"] = "test"

#     # Return a dictionary with the split dataframes: 'train', 'validation', and 'test'
#     out: dict[str, pd.DataFrame] = {
#         "train": train_df.reset_index(drop=True),
#         "validation": val_df.reset_index(drop=True),
#         "test": test_df.reset_index(drop=True),
#     }
#     return out


# def create_hatecheck_dataset(base_dir: str | Path = "data") -> DatasetDict:
#     """
#     This function loads the HateCheck CSV file, processes the text and labels into binary format,
#     splits the data into train, validation, and test sets (70%, 10%, 20%), and returns a Hugging Face `DatasetDict` object.

#     If a CSV file with the split data exists, it will be used directly to avoid recomputing splits.

#     The CSV file must be located at <base_dir>/hatecheck/test_suite_cases.csv and contains two columns:
#     - 'test_case' (text of the tweet)
#     - 'label_gold' (either 'hateful' or 'non-hateful')

#     Returns
#     -------
#     DatasetDict
#         A dictionary containing three splits: 'train', 'validation', and 'test'.
#         Each split contains two columns: 'text' (tweet) and 'label' (binary, 0 or 1).
#     """
#     # Define the path to the CSV file
#     csv_path = Path(base_dir).resolve() / "hatecheck" / "test_suite_cases.csv"
    
#     # Check if the CSV file exists
#     if not csv_path.exists():
#         raise FileNotFoundError(csv_path)

#     # Define the path for the CSV with splits
#     split_csv_path = Path("hatecheck_split.csv")

#     # If the split CSV file exists, load it
#     if split_csv_path.exists():
#         df = pd.read_csv(split_csv_path)
#         splits = {
#             "train": df[df["split"] == "train"],
#             "validation": df[df["split"] == "validation"],
#             "test": df[df["split"] == "test"]
#         }
#     else:
#         # Read the original CSV file into a DataFrame
#         df = pd.read_csv(csv_path)

#         # Apply the function to split and binarize the DataFrame
#         splits = _split_and_binarise_hatecheck(df)

#         # Save the DataFrame with splits to CSV for future use
#         df_with_split = pd.concat([splits["train"], splits["validation"], splits["test"]]).reset_index(drop=True)
#         df_with_split.to_csv(split_csv_path, index=False)

#     # Create a Hugging Face DatasetDict from the DataFrames for each split
#     ds_dict = {
#         split: Dataset.from_pandas(sub_df[["test_case", "label"]]) for split, sub_df in splits.items()
#     }

#     # Return the dataset dictionary
#     return DatasetDict(ds_dict)


def create_hatecheck_dataset(base_dir: str | Path = "data") -> DatasetDict:
    """
    This function loads the HateCheck CSV file, processes the text and labels into binary format,
    splits the data into train, validation, and test sets (70%, 10%, 20%), and returns a Hugging Face `DatasetDict` object.

    If a CSV file with the split data exists, it will be used directly to avoid recomputing splits.

    The CSV file must be located at <base_dir>/hatecheck/test_suite_cases.csv and contains two columns:
    - 'test_case' (text of the tweet)
    - 'label_gold' (either 'hateful' or 'non-hateful')

    Returns
    -------
    DatasetDict
        A dictionary containing three splits: 'train', 'validation', and 'test'.
        Each split contains two columns: 'text' (tweet) and 'label' (binary, 0 or 1).
    """

    # Define the path for the CSV with splits
    split_csv_path = Path("./data/hatecheck/hatecheck_split.csv")

    # If the split CSV file exists, load it
    if split_csv_path.exists():
        df = pd.read_csv(split_csv_path)
        # Split the data based on the 'split' column
        splits = {
            "train": df[df["split"] == "train"],
            "validation": df[df["split"] == "validation"],
            "test": df[df["split"] == "test"]
        }
    else:
        raise FileNotFoundError("Split CSV not found. Please generate the splits first.")

    # Create a Hugging Face DatasetDict from the DataFrames for each split
    ds_dict = {
        split: Dataset.from_pandas(sub_df[["test_case", "label"]]) for split, sub_df in splits.items()
    }

    return DatasetDict(ds_dict)


def create_gab_dataset(base_dir: str | Path = "data") -> DatasetDict:
    """
    This function loads the preprocessed Gab CSV file, processes the text and labels into binary format,
    and splits the data into train, validation, and test sets (70%, 10%, 20%) based on the 'split' column.

    The CSV file must be located at <base_dir>/gab/processed_gab_final.csv and contains the following columns:
    - 'text' (the tweet)
    - 'hate_speech_idx' (binary label: 0 for non-hate speech, 1 for hate speech)
    - 'split' (which indicates the split: 'train', 'validation', 'test')

    Returns
    -------
    DatasetDict
        A dictionary containing three splits: 'train', 'validation', and 'test'.
        Each split contains two columns: 'text' (tweet) and 'hate_speech_idx' (binary, 0 or 1).
    """

    # Define the path for the preprocessed CSV file
    csv_path = Path(base_dir) / "gab" / "processed_gab_final.csv"

    # Check if the CSV file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Please make sure the file is available.")

    # Load the preprocessed CSV file
    df = pd.read_csv(csv_path)

    # Ensure that the 'split' column exists in the dataframe
    if 'split' not in df.columns:
        raise ValueError("The 'split' column is missing in the input file.")

    # Split the data based on the 'split' column into 'train', 'validation', and 'test'
    splits = {
        "train": df[df["split"] == "train"],
        "validation": df[df["split"] == "evaluation"],
        "test": df[df["split"] == "test"]
    }

    # Create a Hugging Face DatasetDict from the DataFrames for each split
    ds_dict = {
        split: Dataset.from_pandas(sub_df[["text", "hate_speech_idx"]]) for split, sub_df in splits.items()
    }

    
    return DatasetDict(ds_dict)

