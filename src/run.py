#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from src.model.qwen_handler import HubConfig, ModelSource, QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.training.callbacks import EarlyStoppingCallback, ValidationCallback
from src.training.trainer import QwenTrainer
from src.utils.auth import setup_authentication
from src.utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

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
    hf_token, source_model_id=None, destination_repo_id=None, private=False, save_method="lora"
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
        # Use the default repository name
        destination_repo_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v2"
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
        "--destination-repo",
        type=str,
        default="tuandunghcmut/Qwen25_Coder_MultipleChoice_v2",
        help="Destination repository ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization level for the model",
    )
    parser.add_argument(
        "--model-loader",
        type=str,
        default="transformers",
        choices=["transformers", "unsloth"],
        help="Library to use for loading the model (transformers or unsloth)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "flash_attention_2", "sdpa", "xformers"],
        help="Attention implementation to use (default: flash_attention_2 for best performance with Unsloth)",
    )
    parser.add_argument(
        "--force-attn-implementation",
        action="store_true",
        help="Force the specified attention implementation even if not optimal",
    )

    # Distributed training configuration
    parser.add_argument(
        "--distributed-mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Distributed training mode (none, ddp, or fsdp)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend (nccl for GPU, gloo for CPU)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--fsdp-min-num-params",
        type=int,
        default=1e6,
        help="Minimum number of parameters to use FSDP",
    )
    parser.add_argument(
        "--fsdp-state-dict-type",
        type=str,
        default="FULL_STATE_DICT",
        choices=["FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"],
        help="FSDP state dict type",
    )
    parser.add_argument(
        "--fsdp-offload-params",
        action="store_true",
        help="Enable CPU offloading for FSDP",
    )
    parser.add_argument(
        "--fsdp-auto-wrap-policy",
        type=str,
        default=None,
        choices=["transformer", "size", None],
        help="FSDP auto wrap policy",
    )

    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Per device batch size for training"
    )
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="Proportion of steps for warmup"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_output",
        help="Directory to save model outputs (will be created inside the 'outputs' folder)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3, help="Patience for early stopping"
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.01,
        help="Minimum change to qualify as improvement for early stopping",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="yaml_reasoning",
        choices=[
            "yaml_reasoning",
            "basic",
            "teacher_reasoned",
            "options",
            "socratic",
            "scientist",
            "lawyer",
            "debugger",
            "philosopher",
            "expert_novice",
            "pros_cons",
            "code_review",
            "math_proof",
        ],
        help="Prompt template to use for formatting",
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Use only 2 dataset instances for quick testing"
    )
    parser.add_argument(
        "--test-training-mode",
        action="store_true",
        help="Use only enough examples to fill one batch (batch_size) for minimal training testing",
    )
    parser.add_argument(
        "--train-on-responses-only",
        action="store_true",
        help="Enable response-only training",
    )
    parser.add_argument(
        "--instruction-token",
        type=str,
        default="<|im_start|>user\n",
        help="Token/prefix indicating start of instruction",
    )
    parser.add_argument(
        "--response-token",
        type=str,
        default="<|im_start|>assistant\n",
        help="Token/prefix indicating start of response",
    )
    parser.add_argument(
        "--instruction-token-id",
        type=int,
        default=None,
        help="Token ID for instruction start (optional)",
    )
    parser.add_argument(
        "--response-token-id",
        type=int,
        default=None,
        help="Token ID for response start (optional)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Name for this experiment (used for WandB and checkpoint naming)",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Number of samples to log for debugging (0 to disable)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Number of steps between logging updates (non-test mode)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Number of steps between model checkpoints (non-test mode)",
    )
    parser.add_argument(
        "--test-logging-steps",
        type=int,
        default=10,
        help="Number of steps between logging updates (test modes)",
    )
    parser.add_argument(
        "--test-save-steps",
        type=int,
        default=20,
        help="Number of steps between model checkpoints (test modes)",
    )

    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")

    # Repository configuration
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the destination repository private",
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        help="Method to use for saving the model",
    )
    parser.add_argument(
        "--push-strategy",
        type=str,
        default="best",
        choices=["best", "end", "all", "no"],
        help="When to push to hub: 'best'=best checkpoint, 'end'=end of training, 'all'=each save, 'no'=don't push",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
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
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_hf", "adam8bit", "pagedadam", "lion", "adafactor"],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Beta1 for Adam-based optimizers"
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999, help="Beta2 for Adam-based optimizers"
    )
    parser.add_argument(
        "--adam-epsilon", type=float, default=1e-8, help="Epsilon for Adam-based optimizers"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--optim-bits",
        type=int,
        default=8,
        choices=[8, 32],
        help="Quantization bits for 8-bit optimizers",
    )

    return parser.parse_args()


def setup_model_and_trainer(source_hub, destination_hub, args):
    """Setup model and trainer with the specified configurations"""
    try:
        # Initialize model handler
        model_handler = QwenModelHandler(
            model_name=args.source_model,
            max_seq_length=args.max_seq_length,
            quantization=args.quantization,
            model_source=ModelSource.UNSLOTH
            if args.model_loader == "unsloth"
            else ModelSource.HUGGINGFACE,
            device_map="auto",
            source_hub_config=source_hub,
            destination_hub_config=destination_hub,
            attn_implementation=args.attn_implementation,
            force_attn_implementation=args.force_attn_implementation,
        )

        # Load dataset
        dataset = load_dataset(
            "json",
            data_files="data/multiple_choice_questions.json",
            split="train",
        )

        # Split dataset into train and validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Create distributed training config
        from src.training.distributed_trainer import DistributedTrainingConfig

        dist_config = DistributedTrainingConfig(
            backend=args.backend,
            mixed_precision=args.mixed_precision,
            gradient_checkpointing=args.gradient_checkpointing,
            use_fsdp=args.distributed_mode == "fsdp",
            fsdp_min_num_params=args.fsdp_min_num_params,
            fsdp_state_dict_type=args.fsdp_state_dict_type,
            fsdp_offload_params=args.fsdp_offload_params,
            fsdp_auto_wrap_policy=args.fsdp_auto_wrap_policy,
        )

        # Initialize appropriate trainer based on distributed mode
        if args.distributed_mode == "ddp":
            from src.training.distributed_trainer import DDPTrainer

            trainer = DDPTrainer(
                model=model_handler.model,
                tokenizer=model_handler.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=dist_config,
            )
        elif args.distributed_mode == "fsdp":
            from src.training.distributed_trainer import FSDPTrainer

            trainer = FSDPTrainer(
                model=model_handler.model,
                tokenizer=model_handler.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=dist_config,
            )
        else:
            trainer = QwenTrainer(
                model=model_handler.model,
                tokenizer=model_handler.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

        return model_handler, trainer

    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise


def get_prompt_template(template_name):
    """Get prompt template constant from name"""
    templates = {
        "yaml_reasoning": PromptCreator.YAML_REASONING,
        "basic": PromptCreator.BASIC,
        "teacher_reasoned": PromptCreator.TEACHER_REASONED,
        "options": PromptCreator.OPTIONS,
    }
    return templates.get(template_name, PromptCreator.YAML_REASONING)


def load_datasets(hf_token, dataset_id, test_mode=False, test_training_mode=False, batch_size=4):
    """
    Load datasets from HuggingFace Hub

    Args:
        hf_token: HuggingFace token for authentication
        dataset_id: ID of the dataset on HuggingFace Hub
        test_mode: If True, use only 2 dataset instances for quick testing
        test_training_mode: If True, use only enough examples to fill one batch
        batch_size: Batch size for training (used when test_training_mode is True)

    Returns:
        Dataset: Training dataset
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
            num_examples = batch_size + max(
                2, int(batch_size * 0.2)
            )  # batch_size + 20% for validation
            logger.info(
                f"TEST TRAINING MODE ENABLED: Using only {num_examples} dataset instances (one batch + validation)"
            )
            dataset = dataset.select(range(min(num_examples, len(dataset))))
            logger.info(f"Dataset reduced to {len(dataset)} examples")

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

        # Generate experiment name if not provided
        if not args.experiment_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if args.test_mode:
                args.experiment_name = f"test_{timestamp}"
            elif args.test_training_mode:
                args.experiment_name = f"test_train_{timestamp}"
            else:
                args.experiment_name = f"experiment_{timestamp}"

        logger.info(f"Using experiment name: {args.experiment_name}")

        # Setup hub configurations
        source_hub, destination_hub = setup_hub_configs(
            hf_token=hf_token,
            source_model_id=args.source_model,
            destination_repo_id=args.destination_repo,
            private=args.private,
            save_method=args.save_method,
        )

        # Initialize model and trainer
        model_handler, trainer = setup_model_and_trainer(source_hub, destination_hub, args)

        # Load dataset from HuggingFace Hub
        train_dataset = load_datasets(
            hf_token,
            args.dataset,
            test_mode=args.test_mode,
            test_training_mode=args.test_training_mode,
            batch_size=args.batch_size,
        )

        # Ensure output directory is inside the 'outputs' folder
        outputs_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_root, exist_ok=True)

        # Create full output path with experiment name
        output_dir = os.path.join(outputs_root, args.experiment_name)
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

        # Setup callbacks
        callbacks = []

        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            patience=args.early_stopping_patience, min_delta=args.early_stopping_delta
        )
        callbacks.append(early_stopping)
        logger.info(
            f"Added early stopping callback with patience={args.early_stopping_patience}, min_delta={args.early_stopping_delta}"
        )

        # Validation callback
        validation_callback = ValidationCallback(trainer_instance=trainer)
        callbacks.append(validation_callback)
        logger.info("Added validation callback for model monitoring")

        # Setup WandB logging if available
        try:
            # Create WandB configuration
            model_name = args.source_model.split("/")[-1]
            project_name = f"{model_name}-Coding-MCQ-Training"

            # Add test mode indicator to run name if enabled
            run_prefix = ""
            if args.test_mode:
                run_prefix = "TEST_"
            elif args.test_training_mode:
                run_prefix = "TEST-TRAIN_"

            run_name = f"{run_prefix}{args.experiment_name}_b{args.batch_size}_lr{args.learning_rate}_e{args.epochs}"

            # Add test mode tag if enabled
            tags = ["qwen", "coding", "lora", "multiple-choice", "callbacks"]
            if args.test_mode:
                tags.append("test_mode")
            elif args.test_training_mode:
                tags.append("test_training_mode")

            notes_prefix = ""
            if args.test_mode:
                notes_prefix = "TEST MODE: "
            elif args.test_training_mode:
                notes_prefix = "TEST TRAINING MODE: "

            notes = f"{notes_prefix}Training with all callbacks enabled: validation, early stopping (patience={args.early_stopping_patience})"

            wandb_config = WandBConfig(
                project_name=project_name,
                run_name=run_name,
                tags=tags,
                notes=notes,
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

        # Set logging and save frequency based on test mode
        if args.test_mode or args.test_training_mode:
            logging_steps = args.test_logging_steps
            save_steps = args.test_save_steps
        else:
            logging_steps = args.logging_steps
            save_steps = args.save_steps

        logger.info(f"Using logging_steps={logging_steps}, save_steps={save_steps}")

        # Map push strategy to trainer terms
        push_strategy_map = {
            "best": "checkpoint",
            "end": "end",
            "all": "every_save",
            "no": "no",
        }
        # Use the push strategy key (not the mapped value) for trainer.train()
        push_to_hub_strategy = args.push_strategy

        # Configure optimizer parameters
        optimizer_config = {
            "optimizer_type": args.optimizer,
            "weight_decay": args.weight_decay,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon,
            "max_grad_norm": args.max_grad_norm,
            "optim_bits": args.optim_bits,
        }

        # Start training
        logger.info("Starting training with all callbacks enabled...")
        logger.info(f"Using optimizer: {args.optimizer}")
        results = trainer.train(
            train_dataset=train_dataset,
            val_split=args.val_split,
            output_dir=output_dir,
            # Training parameters
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            # Validation and checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Hub integration
            push_to_hub_strategy=push_to_hub_strategy,
            # Other settings
            save_total_limit=args.save_total_limit,
            random_seed=args.random_seed,
            # Pass the callbacks
            callbacks=callbacks,
            # Optimizer configuration
            optimizer_config=optimizer_config,
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
