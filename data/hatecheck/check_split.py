#!/usr/bin/env python3
"""
print_splits.py

This script reads the CSV file 'hatecheck_split.csv', computes the
number of unique values in the 'split' column, prints the count of
unique splits, the list of split names, and the number of elements
in each split.
"""

import pandas as pd

# --- Configuration ---
CSV_FILE = "hatecheck_split.csv"

def main():
    # Load the CSV file
    df = pd.read_csv(CSV_FILE)

    # Ensure the 'split' column exists
    if 'split' not in df.columns:
        raise KeyError(f"The file '{CSV_FILE}' does not contain a 'split' column.")

    # Compute unique split values and counts
    counts = df['split'].value_counts()
    unique_values = counts.index.tolist()
    num_unique = len(unique_values)

    # Print summary
    print(f"Number of unique splits: {num_unique}")
    print("Unique split names:")
    for val in unique_values:
        print(f" - {val}")

    # Print counts per split
    print("\nNumber of elements per split:")
    for split_name, count in counts.items():
        print(f" - {split_name}: {count}")

if __name__ == "__main__":
    main()
