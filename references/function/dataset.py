import logging
import os
import time
from typing import Optional, Dict, Any, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_datasets(
    hf_token: str,
    dataset_id: str,
    test_mode: bool = False,
    test_training_mode: bool = False,
    batch_size: int = 32,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load datasets from HuggingFace Hub
    
    Args:
        hf_token: HuggingFace token for authentication
        dataset_id: ID of the dataset on HuggingFace Hub
        test_mode: If True, use only 2 dataset instances for quick testing
        test_training_mode: If True, use only enough examples to fill one batch
        batch_size: Batch size for training (used when test_training_mode is True)
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple[Dataset, Optional[Dataset]]: Training dataset and validation dataset (if val_split > 0)
    """
    try:
        logger.info(f"Loading dataset {dataset_id} from HuggingFace Hub...")
        dataset = load_dataset(dataset_id, token=hf_token, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")

        # Apply test mode if enabled
        if test_mode:
            logger.info("TEST MODE ENABLED: Using only 2 dataset instances")
            dataset = dataset.select(range(2))
            logger.info(f"Dataset reduced to {len(dataset)} examples")
        elif test_training_mode:
            # Use one full batch + a few extra examples for validation if needed
            num_examples = batch_size + max(2, int(batch_size * 0.2))  # batch_size + 20% for validation
            logger.info(
                f"TEST TRAINING MODE ENABLED: Using only {num_examples} dataset instances (one batch + validation)"
            )
            dataset = dataset.select(range(min(num_examples, len(dataset))))
            logger.info(f"Dataset reduced to {len(dataset)} examples")

        # Split dataset if val_split > 0
        val_dataset = None
        if val_split > 0:
            logger.info(f"Splitting dataset with val_split={val_split}")
            split_datasets = dataset.train_test_split(test_size=val_split, seed=random_seed)
            dataset = split_datasets["train"]
            val_dataset = split_datasets["test"]
            logger.info(
                f"Split dataset into {len(dataset)} train and {len(val_dataset)} validation examples"
            )

        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"Features: {list(dataset.features.keys())}")
        logger.info(f"Example:\n{dataset[0]}")

        return dataset, val_dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def create_output_dirs(experiment_name: str) -> str:
    """
    Create output directories for training
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        str: Path to the output directory
    """
    # Ensure output directory is inside the 'outputs' folder
    outputs_root = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outputs_root, exist_ok=True)

    # Create full output path with experiment name
    output_dir = os.path.join(outputs_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Training outputs will be saved to: {output_dir}")

    # Create a symlink to the latest output
    latest_link = os.path.join(outputs_root, "latest")
    if os.path.exists(latest_link) and os.path.islink(latest_link):
        os.remove(latest_link)

    try:
        os.symlink(output_dir, latest_link, target_is_directory=True)
        logger.info(f"Created symlink: {latest_link} -> {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to create symlink: {e}")

    return output_dir


def setup_hub_configs(
    hf_token: str,
    source_model_id: Optional[str] = None,
    destination_repo_id: Optional[str] = None,
    private: bool = False,
    save_method: str = "lora",
) -> Tuple[Any, Any]:
    """
    Setup source and destination hub configurations
    
    Args:
        hf_token: HuggingFace token for authentication
        source_model_id: ID of the source model
        destination_repo_id: ID for the destination repo
        private: Whether the destination repo should be private
        save_method: Method to use for saving the model
        
    Returns:
        Tuple[Any, Any]: Source and destination hub configurations
    """
    from huggingface_hub import HfApi, create_repo
    from src.model.qwen_handler import HubConfig

    # Set default source model if not provided
    if not source_model_id:
        source_model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"

    source_hub = HubConfig(model_id=source_model_id, token=hf_token)

    # Set default destination repo if not provided
    if not destination_repo_id:
        # Use the default repository name
        destination_repo_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v3"
        logger.info(f"Using default destination repository: {destination_repo_id}")

    # Check if the repository exists
    api = HfApi(token=hf_token)
    try:
        # Try to get the repo info to check if it exists
        api.repo_info(repo_id=destination_repo_id, repo_type="model")
        logger.info(f"Repository {destination_repo_id} already exists")
    except Exception as e:
        # If the repo doesn't exist, create it
        logger.info(f"Repository {destination_repo_id} not found, creating it...")
        try:
            create_repo(
                repo_id=destination_repo_id,
                token=hf_token,
                private=private,
                repo_type="model",
            )
            logger.info(f"Repository {destination_repo_id} created successfully")
            # Give HF a moment to register the new repo
            time.sleep(2)
        except Exception as create_error:
            logger.error(f"Failed to create repository: {str(create_error)}")
            raise

    destination_hub = HubConfig(
        model_id=destination_repo_id,
        token=hf_token,
        private=private,
        save_method=save_method,
    )

    logger.info(f"Source model: {source_hub.model_id}")
    logger.info(f"Destination model: {destination_hub.model_id}")

    return source_hub, destination_hub 