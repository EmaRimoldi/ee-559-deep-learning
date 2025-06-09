from datasets import DatasetDict

def downsample(dataset_dict, fraction=0.3):
    """
    Downsample each split of the DatasetDict to a fraction of its original size.

    Args:
        dataset_dict (DatasetDict): The DatasetDict containing 'train', 'validation', and 'test' splits.
        fraction (float): The fraction to downsample the dataset (default is 0.3 for 30%).

    Returns:
        DatasetDict: A downsampled DatasetDict with the same splits but reduced size.
    """
    # Create a downsampled version of each split
    downsampled_dict = DatasetDict({
        split: ds.select(range(int(len(ds) * fraction)))  # Select the first `fraction` of the data
        for split, ds in dataset_dict.items()
    })

    return downsampled_dict