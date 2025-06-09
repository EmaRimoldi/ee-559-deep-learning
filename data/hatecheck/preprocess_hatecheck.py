import pandas as pd
from pathlib import Path

def count_samples_per_split(csv_path: str | Path = "hatecheck_split.csv"):
    """
    This function loads the CSV file with splits and returns the number of samples per split 
    (train, validation, and test) by looking at the 'split' column.
    
    Args:
    - csv_path: The path to the CSV file with the splits.
    
    Returns:
    - A dictionary with the number of samples per split.
    """
    # Convert the path to a Path object
    csv_path = Path(csv_path)
    
    # Check if the CSV file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Debug: print the first few rows of the DataFrame
    print("First few rows of the CSV data:")
    print(df.head())

    # Ensure the 'split' column exists in the dataframe
    if 'split' not in df.columns:
        raise ValueError("The 'split' column is missing in the CSV file.")

    # Count the number of samples per split using the 'split' column
    samples_per_split = df["split"].value_counts()

    # Return the counts as a dictionary
    return samples_per_split.to_dict()

# Example usage:
csv_path = "hatecheck_split.csv"  # Specify the path to your CSV file
samples = count_samples_per_split(csv_path)
print("Number of samples per split:", samples)
