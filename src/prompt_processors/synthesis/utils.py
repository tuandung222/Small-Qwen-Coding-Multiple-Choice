import os
import random
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import Dataset, load_from_disk


def create_special_validation_dataset(
    data_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_val",
    num_samples: int = 100,
    random_seed: int = 42,
    output_dir: str = "./datasets",
    save_dataset: bool = True,
    verbose: bool = True,
) -> Optional[Dataset]:
    """
    Create a special validation dataset with randomly sampled examples

    Args:
        data_path: Path to the validation dataset
        num_samples: Number of examples to sample
        random_seed: Random seed for reproducibility
        output_dir: Directory to save the dataset info
        save_dataset: Whether to save the dataset information
        verbose: Whether to print information

    Returns:
        The sampled dataset
    """
    if verbose:
        print(f"Loading validation data from {data_path}...")

    try:
        split_val = load_from_disk(data_path)

        if verbose:
            print(f"Validation dataset loaded with {len(split_val)} examples")
            print(f"Dataset features: {split_val.features}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Set random seed
    random.seed(random_seed)

    # Get total examples
    total_val_examples = len(split_val)

    # Sample examples
    num_samples = min(num_samples, total_val_examples)
    random_indices = random.sample(range(total_val_examples), num_samples)
    special_val_dataset = split_val.select(random_indices)

    if verbose:
        print(f"Created special validation dataset with {len(special_val_dataset)} examples")
        print("\nFirst few examples:")
        for i in range(min(5, len(special_val_dataset))):
            print(f"Example {i}: {special_val_dataset[i]['task_id']}")

    # Save dataset information
    if save_dataset:
        os.makedirs(output_dir, exist_ok=True)

        # Save task IDs
        task_ids = [example["task_id"] for example in special_val_dataset]
        task_ids_df = pd.DataFrame({"task_id": task_ids})
        task_ids_path = os.path.join(output_dir, "special_val_task_ids.csv")
        task_ids_df.to_csv(task_ids_path, index=False)

        if verbose:
            print(f"\nSaved task IDs to {task_ids_path}")

        # Create task ID prefix distribution
        if all(isinstance(id, str) for id in task_ids):
            prefixes = [id[:1] if len(id) > 0 else "unknown" for id in task_ids]
            prefix_counts = pd.Series(prefixes).value_counts().reset_index()
            prefix_counts.columns = ["prefix", "count"]

            # Save distribution
            prefix_path = os.path.join(output_dir, "task_prefix_distribution.csv")
            prefix_counts.to_csv(prefix_path, index=False)

            if verbose:
                print(f"Saved task prefix distribution to {prefix_path}")

            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x="prefix", y="count", data=prefix_counts)
            plt.title("Distribution of Task ID Prefixes")
            plt.xlabel("Prefix")
            plt.ylabel("Count")
            plt.tight_layout()

            viz_path = os.path.join(output_dir, "task_prefix_distribution.png")
            plt.savefig(viz_path)
            plt.close()

            if verbose:
                print(f"Saved visualization to {viz_path}")

    # Print statistics
    if verbose and "num_choices" in special_val_dataset.features:
        choice_counts = (
            pd.Series([ex["num_choices"] for ex in special_val_dataset]).value_counts().sort_index()
        )
        print("\nDistribution of number of choices:")
        for choices, count in choice_counts.items():
            print(f"  {choices} choices: {count} examples ({count/len(special_val_dataset):.1%})")

    return special_val_dataset


def create_special_training_dataset(
    data_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_train",
    num_samples: int = 100,
    random_seed: int = 42,
    output_dir: str = "./datasets",
    save_dataset: bool = True,
    verbose: bool = True,
) -> Optional[Dataset]:
    """
    Create a special training dataset with randomly sampled examples

    Args:
        data_path: Path to the training dataset
        num_samples: Number of examples to sample
        random_seed: Random seed for reproducibility
        output_dir: Directory to save the dataset info
        save_dataset: Whether to save the dataset information
        verbose: Whether to print information

    Returns:
        The sampled dataset
    """
    if verbose:
        print(f"Loading training data from {data_path}...")

    try:
        split_train = load_from_disk(data_path)

        if verbose:
            print(f"Training dataset loaded with {len(split_train)} examples")
            print(f"Dataset features: {split_train.features}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Set random seed
    random.seed(random_seed)

    # Get total examples
    total_train_examples = len(split_train)

    # Sample examples
    num_samples = min(num_samples, total_train_examples)
    random_indices = random.sample(range(total_train_examples), num_samples)
    special_train_dataset = split_train.select(random_indices)

    if verbose:
        print(f"Created special training dataset with {len(special_train_dataset)} examples")
        print("\nFirst few examples:")
        for i in range(min(5, len(special_train_dataset))):
            print(f"Example {i}: {special_train_dataset[i]['task_id']}")

    # Save dataset information
    if save_dataset:
        os.makedirs(output_dir, exist_ok=True)

        # Save task IDs
        task_ids = [example["task_id"] for example in special_train_dataset]
        task_ids_df = pd.DataFrame({"task_id": task_ids})
        task_ids_path = os.path.join(output_dir, "special_train_task_ids.csv")
        task_ids_df.to_csv(task_ids_path, index=False)

        if verbose:
            print(f"\nSaved task IDs to {task_ids_path}")

        # Create task ID prefix distribution
        if all(isinstance(id, str) for id in task_ids):
            prefixes = [id[:1] if len(id) > 0 else "unknown" for id in task_ids]
            prefix_counts = pd.Series(prefixes).value_counts().reset_index()
            prefix_counts.columns = ["prefix", "count"]

            # Save distribution
            prefix_path = os.path.join(output_dir, "train_task_prefix_distribution.csv")
            prefix_counts.to_csv(prefix_path, index=False)

            if verbose:
                print(f"Saved task prefix distribution to {prefix_path}")

            # Create visualization
            plt.figure(figsize=(10, 6))
            sns.barplot(x="prefix", y="count", data=prefix_counts)
            plt.title("Distribution of Training Task ID Prefixes")
            plt.xlabel("Prefix")
            plt.ylabel("Count")
            plt.tight_layout()

            viz_path = os.path.join(output_dir, "train_task_prefix_distribution.png")
            plt.savefig(viz_path)
            plt.close()

            if verbose:
                print(f"Saved visualization to {viz_path}")

    # Print statistics
    if verbose and "num_choices" in special_train_dataset.features:
        choice_counts = (
            pd.Series([ex["num_choices"] for ex in special_train_dataset])
            .value_counts()
            .sort_index()
        )
        print("\nDistribution of number of choices:")
        for choices, count in choice_counts.items():
            print(f"  {choices} choices: {count} examples ({count/len(special_train_dataset):.1%})")

    return special_train_dataset


def create_filtered_dataset(dataset: Dataset, prefix: str = "k", max_samples: int = 50) -> Dataset:
    """
    Create a dataset filtered by task_id prefix

    Args:
        dataset: Input dataset
        prefix: Task ID prefix to filter by
        max_samples: Maximum number of samples to include

    Returns:
        Filtered dataset
    """
    filtered_examples = [i for i in range(len(dataset)) if dataset[i]["task_id"].startswith(prefix)]

    # Sample if we have more than requested
    if len(filtered_examples) > max_samples:
        filtered_examples = random.sample(filtered_examples, max_samples)

    return dataset.select(filtered_examples)


def create_balanced_dataset(dataset: Dataset, samples_per_choice_count: int = 25) -> Dataset:
    """
    Create a dataset with balanced number of choices

    Args:
        dataset: Input dataset
        samples_per_choice_count: Number of samples to include per choice count

    Returns:
        Balanced dataset
    """
    # Group by number of choices
    choice_groups = {}
    for i in range(len(dataset)):
        num_choices = dataset[i]["num_choices"]
        if num_choices not in choice_groups:
            choice_groups[num_choices] = []
        choice_groups[num_choices].append(i)

    # Sample from each group
    balanced_indices = []
    for num_choices, indices in choice_groups.items():
        sample_size = min(samples_per_choice_count, len(indices))
        balanced_indices.extend(random.sample(indices, sample_size))

    return dataset.select(balanced_indices)
