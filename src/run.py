#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompt_creator import PromptCreator
from model.qwen_handler import HubConfig, ModelSource, QwenModelHandler
from training.trainer import QwenTrainer
from utils.auth import setup_authentication

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup training environment and configurations"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow!")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Setup authentication and get HuggingFace token
    try:
        setup_authentication()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set!")
    except Exception as e:
        logger.error(f"Authentication setup failed: {str(e)}")
        raise

    return hf_token


def setup_hub_configs(
    hf_token, source_model_id=None, destination_repo_id=None, private=True, save_method="lora"
):
    """
    Setup source and destination hub configurations

    Args:
        hf_token: Hugging Face token for authentication
        source_model_id: ID of the source model (defaults to Qwen2.5-Coder-1.5B-Instruct)
        destination_repo_id: ID for the destination repo (username/repo-name)
        private: Whether the destination repo should be private
        save_method: Method to use for saving the model

    Returns:
        Tuple of source and destination hub configurations
    """
    # Set default source model if not provided
    if not source_model_id:
        source_model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"

    source_hub = HubConfig(model_id=source_model_id, token=hf_token)

    # Set default destination repo if not provided
    if not destination_repo_id:
        # Get username from HF API
        api = HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            username = user_info.get("name", "user")
            # Create a default repo name based on source model
            model_name = source_model_id.split("/")[-1]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            destination_repo_id = f"{username}/{model_name}_finetuned_{timestamp}"
        except Exception as e:
            logger.warning(f"Could not get username from HF API: {str(e)}")
            destination_repo_id = f"user/qwen_finetuned_{time.strftime('%Y%m%d_%H%M%S')}"

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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a Qwen model for multiple choice questions")

    # Model configuration
    parser.add_argument(
        "--source-model",
        type=str,
        default="unsloth/Qwen2.5-Coder-1.5B-Instruct",
        help="Source model ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--destination-repo", type=str, help="Destination repository ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length for the model"
    )

    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per device batch size for training"
    )
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--output-dir", type=str, default="./model_output", help="Directory to save model outputs"
    )

    # Repository configuration
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the destination repository private",
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        help="Method to use for saving the model",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="tuandunghcmut/coding-mcq-reasoning",
        help="Dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction of data to use for validation"
    )

    return parser.parse_args()


def setup_model_and_trainer(source_hub, destination_hub, max_seq_length=2048):
    """Initialize model handler and trainer"""
    try:
        # Initialize model handler
        logger.info("Initializing model handler...")
        model_handler = QwenModelHandler(
            model_name=source_hub.model_id,
            max_seq_length=max_seq_length,
            quantization="4bit",
            model_source=ModelSource.UNSLOTH,
            device_map="auto",
            source_hub_config=source_hub,
        )

        # Configure LoRA
        logger.info("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Attention modules
                "gate_proj",
                "up_proj",
                "down_proj",  # FFN modules
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = QwenTrainer(
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            prompt_creator=PromptCreator(PromptCreator.YAML_REASONING),
            lora_config=lora_config,
            destination_hub_config=destination_hub,
            debug_samples=3,  # Log 3 samples per epoch for debugging
        )

        return trainer

    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise


def load_datasets(hf_token, dataset_id):
    """
    Load datasets from HuggingFace Hub

    Args:
        hf_token: HuggingFace token for authentication
        dataset_id: ID of the dataset on HuggingFace Hub

    Returns:
        Dataset: Training dataset
    """
    try:
        logger.info(f"Loading dataset {dataset_id} from HuggingFace Hub...")
        dataset = load_dataset(dataset_id, token=hf_token, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")

        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"Features: {list(dataset.features.keys())}")
        logger.info(f"Example:\n{dataset[0]}")

        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def main():
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup environment
        hf_token = setup_environment()

        # Setup hub configurations
        source_hub, destination_hub = setup_hub_configs(
            hf_token=hf_token,
            source_model_id=args.source_model,
            destination_repo_id=args.destination_repo,
            private=args.private,
            save_method=args.save_method,
        )

        # Initialize model and trainer
        trainer = setup_model_and_trainer(
            source_hub, destination_hub, max_seq_length=args.max_seq_length
        )

        # Load dataset from HuggingFace Hub
        train_dataset = load_datasets(hf_token, args.dataset)

        # Training configuration
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Training outputs will be saved to: {output_dir}")

        # Start training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataset=train_dataset,
            val_split=args.val_split,
            output_dir=output_dir,
            # Training parameters
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            # Validation and checkpointing
            save_strategy="steps",
            save_steps=500,
            logging_steps=100,
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Hub integration
            push_to_hub_strategy="best",
            # Other settings
            save_total_limit=3,
            random_seed=42,
        )

        # Log results
        logger.info("Training completed!")
        if isinstance(results, dict):
            logger.info("Training metrics:")
            for key, value in results.items():
                logger.info(f"{key}: {value}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1
    finally:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
