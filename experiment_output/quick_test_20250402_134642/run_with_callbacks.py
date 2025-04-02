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
from training.callbacks import EarlyStoppingCallback, ValidationCallback
from training.trainer import QwenTrainer
from utils.auth import setup_authentication
from utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./experiment_output/quick_test_20250402_134642/training.log")
    ],
)
logger = logging.getLogger(__name__)

def main():
    # Setup environment
    try:
        setup_authentication()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set!")

        # Log GPU information
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("CUDA is not available. Training will be slow!")
    except Exception as e:
        logger.error(f"Authentication setup failed: {str(e)}")
        raise

    # Setup hub configurations
    model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
    repo_id = ""

    source_hub = HubConfig(model_id=model_id, token=hf_token)

    # Handle destination repo
    if repo_id:
        destination_repo_id = repo_id
    else:
        # Get username from HF API
        api = HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            username = user_info.get("name", "user")
            model_name = model_id.split("/")[-1]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            destination_repo_id = f"{username}/{model_name}_finetuned_{timestamp}"
        except Exception as e:
            logger.warning(f"Could not get username from HF API: {str(e)}")
            destination_repo_id = f"user/qwen_finetuned_{time.strftime('%Y%m%d_%H%M%S')}"

    # Check if the repository exists
    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=destination_repo_id, repo_type="model")
        logger.info(f"Repository {destination_repo_id} already exists")
    except Exception as e:
        # If the repo doesn't exist, create it
        logger.info(f"Repository {destination_repo_id} not found, creating it...")
        try:
            create_repo(
                repo_id=destination_repo_id,
                token=hf_token,
                private=True,
                repo_type="model",
            )
            logger.info(f"Repository {destination_repo_id} created successfully")
            time.sleep(2)
        except Exception as create_error:
            logger.error(f"Failed to create repository: {str(create_error)}")
            raise

    destination_hub = HubConfig(
        model_id=destination_repo_id,
        token=hf_token,
        private=True,
        save_method="lora",
    )

    logger.info(f"Source model: {source_hub.model_id}")
    logger.info(f"Destination model: {destination_hub.model_id}")

    # Initialize model and trainer
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
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
                "gate_proj", "up_proj", "down_proj",     # FFN modules
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
            debug_samples=5,  # Log 5 samples per epoch for debugging
        )
    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise

    # Load dataset
    try:
        logger.info(f"Loading dataset {\"tuandunghcmut/coding-mcq-reasoning\"} from HuggingFace Hub...")
        dataset = load_dataset("tuandunghcmut/coding-mcq-reasoning", token=hf_token, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")

        # Apply test mode if enabled
        test_mode = true
        if test_mode:
            logger.info("TEST MODE ENABLED: Using only 2 dataset instances")
            dataset = dataset.select(range(min(2, len(dataset))))
            logger.info(f"Reduced dataset to {len(dataset)} instances")

        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"Features: {list(dataset.features.keys())}")
        logger.info(f"Example:\n{dataset[0]}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

    # Setup callbacks
    callbacks = []

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.01)
    callbacks.append(early_stopping)
    logger.info(f"Added early stopping callback with patience=3")

    # Validation callback
    validation_callback = ValidationCallback(trainer_instance=trainer)
    callbacks.append(validation_callback)
    logger.info("Added validation callback for model monitoring")

    # Setup WandB logging
    try:
        # Create WandB configuration
        model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct".split("/")[-1]
        project_name = f"{model_name}-Coding-MCQ-Training"
        exp_name = "quick_test"
        run_name = f"{exp_name}_batch{16}_lr{2e-4}_e{2}_{int(time.time())}"

        if test_mode:
            run_name = f"TEST_{run_name}"

        wandb_config = WandBConfig(
            project_name=project_name,
            run_name=run_name,
            tags=["qwen", "coding", "lora", "multiple-choice", "callbacks"] + (["test_mode"] if test_mode else []),
            notes=f"{'TEST MODE: ' if test_mode else ''}Comprehensive training with all callbacks enabled: EarlyStop, Validation, WandB",
            log_memory=True,
            log_gradients=True,
        )

        # Initialize WandB logger
        wandb_logger = WandBLogger(config=wandb_config)
        wandb_logger.setup()

        # Add WandB callback
        wandb_callback = WandBCallback(logger=wandb_logger)
        callbacks.append(wandb_callback)

        logger.info(f"WandB logging enabled with project: {project_name}, run: {run_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize WandB logging: {e}")
        logger.warning("Continuing without WandB callbacks")

    # Training configuration
    output_dir = "./experiment_output/quick_test_20250402_134642"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training outputs will be saved to: {output_dir}")

    # Start training
    logger.info("Starting training with all callbacks enabled...")
    try:
        # In test mode, we use higher frequency logging and saving
        logging_steps = 10 if test_mode else 50
        save_steps = 20 if test_mode else 200

        results = trainer.train(
            train_dataset=dataset,
            val_split=0.15,
            output_dir=output_dir,
            # Training parameters
            num_train_epochs=2,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            # Validation and checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Hub integration
            push_to_hub_strategy="best",
            # Other settings
            save_total_limit=3,
            random_seed=42,
            # Pass the callbacks
            callbacks=callbacks,
        )

        # Log results
        logger.info("Training completed!")
        if isinstance(results, dict):
            logger.info("Training metrics:")
            for key, value in results.items():
                logger.info(f"{key}: {value}")

        # Save metrics to file
        with open(os.path.join(output_dir, "training_metrics.txt"), "w") as f:
            f.write(f"Experiment: quick_test\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test mode: {str(test_mode).lower()}\n\n")
            f.write("Training Metrics:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

        # Create a success marker file
        with open(os.path.join(output_dir, "TRAINING_COMPLETE"), "w") as f:
            f.write("Training completed successfully at " + time.strftime("%Y-%m-%d %H:%M:%S"))

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        with open(os.path.join(output_dir, "TRAINING_INTERRUPTED"), "w") as f:
            f.write("Training was interrupted at " + time.strftime("%Y-%m-%d %H:%M:%S"))
        return 1
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        with open(os.path.join(output_dir, "TRAINING_FAILED"), "w") as f:
            f.write(f"Training failed at {time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}")
        return 1
    finally:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
