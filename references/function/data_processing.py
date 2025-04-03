import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    """
    Data collator for training and evaluation
    """
    tokenizer: PreTrainedTokenizer
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Dict[str, torch.Tensor]: Collated batch
        """
        try:
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            
            return batch
            
        except Exception as e:
            logger.error(f"Error collating batch: {str(e)}")
            raise


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    text_column: str = "text",
    label_column: Optional[str] = None,
) -> Dataset:
    """
    Preprocess a dataset for training or evaluation
    
    Args:
        dataset: Dataset to preprocess
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Dataset: Preprocessed dataset
    """
    try:
        def tokenize_function(examples):
            texts = examples[text_column]
            result = tokenizer(
                texts,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=None,
            )
            
            if label_column is not None:
                result["labels"] = examples[label_column]
                
            return result
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise


def create_data_loaders(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    text_column: str = "text",
    label_column: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create data loaders for training and evaluation
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer to use
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        max_length: Maximum sequence length
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences
        text_column: Name of the text column
        label_column: Name of the label column
        
    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: Training and evaluation data loaders
    """
    try:
        if tokenizer is not None:
            # Preprocess datasets
            train_dataset = preprocess_dataset(
                train_dataset,
                tokenizer,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                text_column=text_column,
                label_column=label_column,
            )
            
            if eval_dataset is not None:
                eval_dataset = preprocess_dataset(
                    eval_dataset,
                    tokenizer,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    text_column=text_column,
                    label_column=label_column,
                )
                
        # Create data collator
        data_collator = DataCollator(
            tokenizer=tokenizer,
            padding=padding,
            max_length=max_length,
        )
        
        # Create data loaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            
        return train_dataloader, eval_dataloader
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        raise


def augment_dataset(
    dataset: Dataset,
    augmentation_fn: Callable[[str], str],
    text_column: str = "text",
    num_augmentations: int = 1,
) -> Dataset:
    """
    Augment a dataset using a custom augmentation function
    
    Args:
        dataset: Dataset to augment
        augmentation_fn: Function to apply for augmentation
        text_column: Name of the text column
        num_augmentations: Number of augmentations per example
        
    Returns:
        Dataset: Augmented dataset
    """
    try:
        def augment_example(example):
            augmented_texts = []
            for _ in range(num_augmentations):
                augmented_text = augmentation_fn(example[text_column])
                augmented_texts.append(augmented_text)
            return {text_column: augmented_texts}
            
        augmented_dataset = dataset.map(
            augment_example,
            remove_columns=[text_column],
            batched=False,
        )
        
        # Flatten the augmented texts
        augmented_dataset = augmented_dataset.flatten()
        
        return augmented_dataset
        
    except Exception as e:
        logger.error(f"Error augmenting dataset: {str(e)}")
        raise


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[Dataset, Dataset, Dataset]: Train, validation, and test datasets
    """
    try:
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        # Shuffle dataset
        shuffled_dataset = dataset.shuffle(seed=seed)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        
        # Split dataset
        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error splitting dataset: {str(e)}")
        raise 