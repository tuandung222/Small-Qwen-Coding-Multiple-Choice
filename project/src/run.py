#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import torch
from data.prompt_creator import PromptCreator
from datasets import load_dataset
from model.qwen_handler import HubConfig, ModelSource, QwenModelHandler
from peft import LoraConfig
from training.trainer import QwenTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

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

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set!")

    return hf_token


def setup_hub_configs(hf_token):
    """Setup source and destination hub configurations"""
    source_hub = HubConfig(model_id="unsloth/Qwen2.5-Coder-1.5B-Instruct", token=hf_token)

    destination_hub = HubConfig(
        model_id="tuandunghcmut/Qwen25_Coder_MultipleChoice_v2",
        token=hf_token,
        private=True,
        save_method="lora",
    )

    logger.info(f"Source model: {source_hub.model_id}")
    logger.info(f"Destination model: {destination_hub.model_id}")

    return source_hub, destination_hub


def setup_model_and_trainer(source_hub, destination_hub):
    """Initialize model handler and trainer"""
    try:
        # Initialize model handler
        logger.info("Initializing model handler...")
        model_handler = QwenModelHandler(
            model_name=source_hub.model_id,
            max_seq_length=2048,
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


def load_datasets(hf_token: str):
    """
    Load datasets from HuggingFace Hub

    Args:
        hf_token: HuggingFace token for authentication

    Returns:
        Dataset: Training dataset
    """
    try:
        logger.info("Loading dataset from HuggingFace Hub...")
        dataset = load_dataset("tuandunghcmut/coding-mcq-reasoning", token=hf_token, split="train")
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
        # Setup environment
        hf_token = setup_environment()

        # Setup hub configurations
        source_hub, destination_hub = setup_hub_configs(hf_token)

        # Initialize model and trainer
        trainer = setup_model_and_trainer(source_hub, destination_hub)

        # Load dataset from HuggingFace Hub
        train_dataset = load_datasets(hf_token)

        # Training configuration
        output_dir = "./model_output"
        os.makedirs(output_dir, exist_ok=True)

        # Start training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataset=train_dataset,
            val_split=0.1,  # 10% of data for validation
            output_dir=output_dir,
            # Training parameters
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            # Validation and checkpointing
            save_strategy="steps",
            save_steps=500,
            logging_steps=100,
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Learning rate schedule
            lr_scheduler_type="cosine",  # Cosine decay with warmup
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
