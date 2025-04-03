import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from src.model.qwen_handler import HubConfig, QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser
from src.utils.auth import setup_authentication

from .callbacks import (
    EarlyStoppingCallback,
    LRMonitorCallback,
    ModelLoadingAlertCallback,
    PromptMonitorCallback,
    ValidationCallback,
)

# define logger
logger = logging.getLogger(__name__)


def is_bf16_supported():
    """Check if BF16 is supported on the current device"""
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except:
        return False


class QwenTrainer:
    """
    Training handler for Qwen models with support for both Hugging Face and Unsloth optimizations.

    This class provides a unified interface for:
    1. Fine-tuning Qwen models with LoRA (Low-Rank Adaptation)
    2. Supporting both standard training and teacher-guided training
    3. Handling model checkpointing and validation
    4. Integration with Weights & Biases for experiment tracking
    5. Pushing models to Hugging Face Hub

    Key Features:
    - Parameter-efficient fine-tuning using LoRA
    - Support for teacher-guided training with YAML-formatted responses
    - Automatic mixed precision training (FP16/BF16)
    - Integration with Unsloth for optimized training
    - Flexible prompt formatting through PromptCreator
    - Comprehensive validation and checkpointing

    Example usage:
    ```python
    trainer = QwenTrainer(
        model=model,
        tokenizer=tokenizer,
        prompt_creator=PromptCreator(PromptCreator.TEACHER_REASONED),
        lora_config=lora_config,
        destination_hub_config=destination_hub_config,
    )

    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="./model_output",
        num_train_epochs=3,
        per_device_train_batch_size=16
    )
    ```
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_creator: PromptCreator,
        lora_config: Optional[LoraConfig] = None,
        destination_hub_config: Optional[HubConfig] = None,
        debug_samples: int = 3,  # Number of samples to log for debugging
        responses_only_config: Optional[Dict[str, Any]] = None,
        attention_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the QwenTrainer with model, tokenizer, and configuration.

        Args:
            model: The base model to fine-tune (Qwen model from HF or Unsloth)
            tokenizer: The tokenizer associated with the model
            prompt_creator: PromptCreator instance for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient training
            destination_hub_config: Optional configuration for pushing to HuggingFace Hub
            debug_samples: Number of random samples to log during training for debugging.
                         Set to 0 to disable debug logging. Default: 3
            responses_only_config: Optional configuration for response-only training
            attention_config: Optional configuration for attention implementation

        The trainer will:
        1. Set up authentication if needed
        2. Configure the model for training
        3. Initialize tracking metrics
        4. Set up sequence length constraints
        """
        # Setup authentication for HF Hub access
        setup_authentication()

        self.model = model
        self.tokenizer = tokenizer
        self.prompt_creator = prompt_creator
        self.lora_config = lora_config
        self.destination_hub_config = destination_hub_config
        self.debug_samples = debug_samples
        self.responses_only_config = responses_only_config
        self.attention_config = attention_config

        # Initialize training state
        self.peft_model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.best_val_metric = float("inf")
        self.best_checkpoint_path = None
        self.training_stats = {}
        self.validation_stats = {}

        # Set maximum sequence length based on model config
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_seq_length = min(2048, self.model.config.max_position_embeddings)
        else:
            self.max_seq_length = 2048  # Default fallback

        self.debug_examples = []  # Store debug examples

    def validate(
        self,
        val_dataset,
        quality_val_dataset=None,
        prompt_type=None,
        batch_size=64,
        temperature=0.0,
    ):
        """Validate the model on validation dataset"""
        if prompt_type:
            self.prompt_creator.set_prompt_type(prompt_type)

        validation_callback = ValidationCallback(
            trainer_instance=self,
            val_dataset=val_dataset,
            quality_val_dataset=quality_val_dataset,
            validation_strategy="steps",
            validation_steps=100,
            save_best_checkpoint=True,
            early_stopper=EarlyStoppingCallback(patience=3),
        )

        return validation_callback

    def _create_temp_model_handler(self):
        """Create a temporary model handler for validation"""
        temp_handler = QwenModelHandler(
            model_name=self.model.config.name_or_path,
            device_map="auto",
        )
        temp_handler.model = self.model
        temp_handler.tokenizer = self.tokenizer
        return temp_handler

    def save_checkpoint(self, output_dir, val_metric=None, is_best=False):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)

        if is_best:
            output_dir = os.path.join(output_dir, "best_model")
        else:
            output_dir = os.path.join(output_dir, "checkpoint")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if val_metric is not None:
            with open(os.path.join(output_dir, "val_metric.txt"), "w") as f:
                f.write(str(val_metric))

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        self.model = self.model.from_pretrained(checkpoint_path)
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)

    def setup(self):
        """Setup trainer with LoRA if configured"""
        if self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config)

    def prepare_model_for_training(self) -> Any:
        """
        Prepare model for training with LoRA configuration.

        This method:
        1. Applies LoRA configuration if provided
        2. Uses Unsloth's optimizations when possible
        3. Configures gradient checkpointing
        4. Sets up mixed precision training

        Returns:
            The prepared model ready for training

        Raises:
            RuntimeError: If model preparation fails
            Exception: If Unsloth optimization fails and fallback is needed
        """
        use_unsloth = False
        if self.lora_config:
            try:
                # Try using Unsloth's optimized LoRA implementation
                # Extract parameters from the PEFT LoraConfig
                r = self.lora_config.r
                lora_alpha = self.lora_config.lora_alpha
                lora_dropout = self.lora_config.lora_dropout
                target_modules = self.lora_config.target_modules

                print("\033[92mAttempting to use Unsloth's LoRA implementation...\033[0m")
                use_unsloth = True

                # Use Unsloth's LoRA implementation
                self.peft_model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )

                # Print parameters info
                total_params = sum(p.numel() for p in self.peft_model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.peft_model.parameters() if p.requires_grad
                )
                print(
                    f"\033[92mtrainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}\033[0m"
                )

                # Mark model as Unsloth model
                self.peft_model.is_unsloth_model = True
                return self.peft_model

            except Exception as e:
                print(f"\033[91mFailed to use Unsloth's LoRA implementation: {e}\033[0m")
                print("\033[91mFalling back to standard PEFT LoRA\033[0m")

                try:
                    # Fallback to standard PEFT LoRA
                    self.peft_model = get_peft_model(self.model, self.lora_config)
                    self.peft_model.print_trainable_parameters()
                    # Mark model as not using Unsloth
                    self.peft_model.is_unsloth_model = False
                    return self.peft_model
                except Exception as e:
                    print(f"\033[91mFailed to apply standard PEFT LoRA: {e}\033[0m")
                    return None

        # If no LoRA config, prepare base model
        try:
            self.model.train()
            # Mark model as not using Unsloth
            self.model.is_unsloth_model = False
            return self.model
        except Exception as e:
            print(f"\033[91mFailed to prepare base model: {e}\033[0m")
            return None

    def _log_debug_examples(self, dataset: Dataset, epoch: int = 0):
        """
        Log debug examples to wandb for monitoring training data.

        Args:
            dataset: The dataset to sample from
            epoch: Current training epoch

        The function will:
        1. Sample random examples from the dataset
        2. Format them with the chat template
        3. Log them to wandb with epoch information
        4. Include both raw input and tokenized format
        """
        if self.debug_samples <= 0:
            return

        try:
            import wandb

            if not wandb.run:
                return

            # Sample random indices
            import random

            indices = random.sample(range(len(dataset)), min(self.debug_samples, len(dataset)))

            debug_table = []
            for idx in indices:
                example = dataset[idx]

                # Get raw text
                raw_text = example["text"]

                # Get tokenized form
                tokens = self.tokenizer.encode(raw_text)
                decoded = self.tokenizer.decode(tokens)

                # Calculate token statistics
                n_tokens = len(tokens)

                debug_table.append(
                    {
                        "epoch": epoch,
                        "example_idx": idx,
                        "raw_text": raw_text,
                        "tokenized_text": decoded,
                        "n_tokens": n_tokens,
                    }
                )

            # Log to wandb
            wandb.log(
                {
                    "debug/training_examples": wandb.Table(data=debug_table),
                    "debug/epoch": epoch,
                    "debug/max_tokens": max(ex["n_tokens"] for ex in debug_table),
                    "debug/min_tokens": min(ex["n_tokens"] for ex in debug_table),
                    "debug/avg_tokens": sum(ex["n_tokens"] for ex in debug_table)
                    / len(debug_table),
                }
            )

            # Store examples for later reference
            self.debug_examples = debug_table

            print(f"\nLogged {len(debug_table)} debug examples for epoch {epoch}")
            print("Example preview:")
            for ex in debug_table[:1]:  # Show first example
                print(f"\nExample {ex['example_idx']} ({ex['n_tokens']} tokens):")
                print(f"Raw text preview: {ex['raw_text'][:200]}...")

        except Exception as e:
            print(f"Warning: Failed to log debug examples: {e}")

    def _prepare_dataset_for_training(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for training by formatting it with the prompt creator.

        Args:
            dataset: The raw dataset to format

        Returns:
            Formatted dataset ready for training
        """
        logger.info("Preparing dataset for training...")

        def format_example(example):
            # Use the prompt creator to format the prompt
            prompt = self.prompt_creator.create_training_prompt(
                question=example["question"], choices=example["choices"]
            )

            # Get the target completion from the example
            if "teacher_reasoning" in example and example["teacher_reasoning"]:
                # Use teacher reasoning if available
                completion = example["teacher_reasoning"]
            else:
                # Default to just the answer if no reasoning is available
                completion = f"The answer is {example['answer']}."

            # Format as a conversation
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]

            # Apply the tokenizer's chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Tokenize the text
            tokenized = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Return the input_ids and attention_mask
            result = {
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "labels": tokenized.input_ids[
                    0
                ].clone(),  # For causal LM, labels are the same as input_ids
            }

            return result

        # Apply the formatting to each example
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=dataset.column_names,  # Remove all original columns
            load_from_cache_file=False,
        )

        logger.info(f"Dataset prepared: {len(formatted_dataset)} examples")

        # Log a few examples for debugging
        if self.debug_samples > 0:
            logger.info("Sample formatted examples:")
            for i in range(min(self.debug_samples, len(formatted_dataset))):
                sample = formatted_dataset[i]
                decoded_text = self.tokenizer.decode(sample["input_ids"])
                logger.info(f"Example {i}:\n{decoded_text[:200]}... (truncated)")

        return formatted_dataset

    def _generate_wandb_run_name(
        self,
        num_train_epochs: int,
        learning_rate: float,
        batch_size: int,
        model_name: str,
        scheduler_type: str = "cosine",
        warmup_ratio: float = 0.1,
    ) -> str:
        """
        Generate a professional and informative wandb run name.

        Format: {model_variant}_{training_type}_{batch_size}b_{lr}lr_{epochs}e_{scheduler}_{warmup_ratio}wu_{timestamp}
        Example: qwen1.5-7b_lora-ft_32b_2.0e-4lr_3e_cosine_0.1wu_20240220
        """
        # Extract model variant (e.g., "qwen1.5-7b" from "Qwen/Qwen1.5-7B")
        model_variant = model_name.split("/")[-1].lower()

        # Determine training type
        training_type = "lora-ft" if self.lora_config else "full-ft"

        # Format learning rate (e.g., 2e-4)
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e-")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d")

        # Add scheduler and warmup info
        scheduler_info = f"{scheduler_type}_{warmup_ratio:.1f}wu"

        # Construct run name with scheduler info
        run_name = f"{model_variant}_{training_type}_{batch_size}b_{lr_str}lr_{num_train_epochs}e_{scheduler_info}_{timestamp}"

        return run_name

    def _setup_wandb_config(
        self,
        num_train_epochs: int,
        learning_rate: float,
        batch_size: int,
        gradient_accumulation_steps: int,
        max_steps: Optional[int],
        warmup_steps: int,
        warmup_ratio: float,
        user_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Setup a comprehensive wandb configuration."""
        model_name = self.model.config.name_or_path

        # Generate run name
        run_name = self._generate_wandb_run_name(
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_name=model_name,
            warmup_ratio=warmup_ratio,
        )

        # Calculate effective batch size
        effective_batch_size = batch_size * gradient_accumulation_steps

        # Setup default configuration
        default_config = {
            "project": "Qwen2.5-Coder-1.5B-Instruct-Coding-Multiple-Choice",
            "name": run_name,
            "tags": [
                "qwen",
                "multiple-choice",
                "coding",
                "lora" if self.lora_config else "full-finetune",
                f"{model_name}",
            ],
            "notes": f"Training {model_name} for coding multiple choice tasks",
            "config": {
                # Model Configuration
                "model": {
                    "name": model_name,
                    "type": "decoder-only",
                    "parameters": sum(p.numel() for p in self.model.parameters()),
                },
                # Training Configuration
                "training": {
                    "epochs": num_train_epochs,
                    "learning_rate": learning_rate,
                    "batch_size_per_device": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_batch_size": effective_batch_size,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "sequence_length": self.max_seq_length,
                },
                # LoRA Configuration
                "lora": self.lora_config.__dict__ if self.lora_config else None,
                # Hardware Configuration
                "hardware": {
                    "precision": "bf16" if is_bf16_supported() else "fp16",
                    "device": str(self.model.device),
                },
                # Dataset Information
                "dataset": {
                    "type": "multiple-choice",
                    "domain": "coding",
                },
            },
            "group": f"{model_name}_experiments",
        }

        # Update with user-provided config if any
        if user_config:
            # Deep update the configuration
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

        default_config["config"]["training"].update(
            {
                "scheduler": {
                    "type": "cosine",
                    "warmup_ratio": warmup_ratio,
                    "warmup_steps": warmup_steps,
                }
            }
        )

        return default_config

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        val_split: float = 0.1,
        output_dir: str = "./model_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        max_steps: Optional[int] = None,
        logging_steps: int = 10,
        save_steps: int = 100,
        save_strategy: str = "epoch",
        save_total_limit: int = 1,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        callbacks: Optional[List[Any]] = None,
        random_seed: int = 42,
        push_to_hub_strategy: str = "end",
        wandb_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        responses_only_config: Optional[Dict[str, Any]] = None,
        attention_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with comprehensive configuration and monitoring.

        Learning Rate Schedule:
        - Configurable LR schedulers: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt
        - Warmup phase: Linear increase from 0 to learning_rate for warmup_steps
        - Decay phase: Follows the selected scheduler pattern
        - warmup_ratio determines warmup_steps as a fraction of total training steps

        Data Processing:
        - Uses DataCollatorForLanguageModeling for efficient batching
        - Handles padding and attention masks automatically
        - Ensures inputs are properly formatted for the model

        Response-Only Training (Unsloth feature):
        - Option to train only on model responses using Unsloth's train_on_responses_only
        - Identifies instruction and response segments using configurable tokens
        - Enables more focused training on generated responses rather than instructions

        Validation Strategy:
        - If val_dataset is None, splits train_dataset using val_split ratio
        - Evaluates model according to save_strategy
        - Uses metric_for_best_model to track best checkpoint

        Checkpointing:
        - Saves checkpoints according to save_strategy
        - Keeps save_total_limit number of checkpoints
        - Optionally loads best model at end of training

        Hub Integration:
        - Can push to hub based on push_to_hub_strategy:
          * "end": Push only at end of training
          * "best": Push when new best model is found
          * "all": Push after each save
          * "no": Don't push to hub

        Optimizer Configuration:
        - Configurable optimizer type: adamw_torch, adamw_hf, adam8bit, pagedadam, lion, adafactor
        - Customizable optimizer parameters (beta1, beta2, epsilon, etc.)
        - Support for 8-bit optimizers for memory efficiency

        Args:
            train_dataset: Dataset for training
            val_dataset: Optional dataset for validation
            val_split: Fraction of training data to use for validation if val_dataset is None
            output_dir: Directory to save model checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per GPU/CPU
            gradient_accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate for training
            warmup_ratio: Ratio of total steps to use for warmup (default: 0.1)
            max_steps: Maximum number of training steps (overrides num_train_epochs)
            logging_steps: Number of steps between logging updates
            save_steps: Number of steps between model saves
            save_strategy: When to save checkpoints ("steps", "epoch", or "no")
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Whether to load the best model after training
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            callbacks: Optional list of training callbacks
            random_seed: Random seed for dataset splitting and shuffling
            push_to_hub_strategy: When to push to hub ("end", "best", "all", "no")
            wandb_config: Optional configuration for Weights & Biases logging
            optimizer_config: Optional configuration for optimizer selection and parameters
            lr_scheduler_config: Optional configuration for learning rate scheduler selection and parameters
            responses_only_config: Optional configuration for training only on responses with Unsloth
            attention_config: Optional configuration for attention implementation

        Returns:
            Dict containing training metrics and results
        """
        # Use stored configs if not explicitly provided
        if responses_only_config is None:
            responses_only_config = self.responses_only_config

        if attention_config is None:
            attention_config = self.attention_config

        # Initialize wandb if needed
        try:
            import wandb

            if wandb.run is None:
                # Calculate total steps for warmup
                if max_steps is not None and max_steps > 0:
                    total_steps = max_steps
                else:
                    total_steps = (
                        len(train_dataset)
                        // (per_device_train_batch_size * gradient_accumulation_steps)
                        * num_train_epochs
                    )
                warmup_steps = int(total_steps * warmup_ratio)
                print(
                    f"Using {warmup_steps} warmup steps ({warmup_ratio:.1%} of {total_steps} total steps)"
                )

                # Setup wandb configuration
                wandb_config = self._setup_wandb_config(
                    num_train_epochs=num_train_epochs,
                    learning_rate=learning_rate,
                    batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    warmup_ratio=warmup_ratio,
                    user_config=wandb_config,
                )

                # Initialize wandb
                wandb.init(**wandb_config)
                print(f"Initialized wandb run: {wandb_config['name']}")
                print(f"Project: {wandb_config['project']}")
                print(f"Tags: {', '.join(wandb_config['tags'])}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")

        # Prepare model
        model_to_train = self.prepare_model_for_training()
        if model_to_train is None:
            raise RuntimeError("Model preparation failed")

        # Format datasets for training
        formatted_train_dataset = self._prepare_dataset_for_training(train_dataset)
        formatted_val_dataset = None
        if val_dataset is not None:
            formatted_val_dataset = self._prepare_dataset_for_training(val_dataset)
        elif val_split > 0:
            # Split train dataset if no validation dataset is provided
            logger.info(f"Splitting train dataset with val_split={val_split}")
            split_datasets = formatted_train_dataset.train_test_split(
                test_size=val_split, seed=random_seed
            )
            formatted_train_dataset = split_datasets["train"]
            formatted_val_dataset = split_datasets["test"]
            logger.info(
                f"Split dataset into {len(formatted_train_dataset)} train and {len(formatted_val_dataset)} validation examples"
            )

        # Map push_to_hub_strategy
        hub_strategy_mapping = {
            "end": "end",
            "best": "checkpoint",
            "all": "every_save",
            "no": "end",
        }

        if push_to_hub_strategy not in hub_strategy_mapping:
            raise ValueError(
                f"Invalid push_to_hub_strategy: {push_to_hub_strategy}. "
                "Valid values are: 'end', 'best', 'all', 'no'"
            )

        # Default optimizer config if not provided
        if optimizer_config is None:
            optimizer_config = {
                "optimizer_type": "adamw_torch",
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "max_grad_norm": 1.0,
                "optim_bits": 8,
            }

        # Default LR scheduler config if not provided
        if lr_scheduler_config is None:
            lr_scheduler_config = {
                "lr_scheduler_type": "cosine",
                "num_cycles": 1,
                "power": 1.0,
                "last_epoch": -1,
            }

        logger.info(f"Using optimizer: {optimizer_config['optimizer_type']}")
        logger.info(f"Using LR scheduler: {lr_scheduler_config['lr_scheduler_type']}")

        # Calculate warmup steps
        if max_steps is not None and max_steps > 0:
            total_steps = max_steps
        else:
            total_steps = (
                len(formatted_train_dataset)
                // (per_device_train_batch_size * gradient_accumulation_steps)
                * num_train_epochs
            )
        warmup_steps = int(total_steps * warmup_ratio)

        # Setup training arguments
        training_args_dict = {
            # Basic training configuration
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,  # Use calculated warmup steps
            # Learning rate schedule
            "lr_scheduler_type": lr_scheduler_config["lr_scheduler_type"],
            # Logging and saving configuration
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_strategy": save_strategy,
            "save_total_limit": save_total_limit,
            # Model selection configuration
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            # Mixed precision training
            "fp16": not is_bf16_supported(),
            "bf16": is_bf16_supported(),
            # Optimizer configuration
            "optim": optimizer_config["optimizer_type"],
            "weight_decay": optimizer_config["weight_decay"],
            "adam_beta1": optimizer_config["beta1"],
            "adam_beta2": optimizer_config["beta2"],
            "adam_epsilon": optimizer_config["epsilon"],
            "max_grad_norm": optimizer_config["max_grad_norm"],
            # Integration configuration
            "report_to": "wandb",
            "push_to_hub": bool(self.destination_hub_config),
            "hub_model_id": self.destination_hub_config.model_id
            if self.destination_hub_config
            else None,
            "hub_token": self.destination_hub_config.token if self.destination_hub_config else None,
            "hub_strategy": hub_strategy_mapping[push_to_hub_strategy],
            # Dataset configuration
            "remove_unused_columns": False,  # Already formatted the dataset
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            # Set random seed
            "seed": random_seed,
        }

        # Handle max_steps
        if max_steps is not None and max_steps > 0:
            training_args_dict["max_steps"] = max_steps
            training_args_dict.pop("num_train_epochs", None)

        # Set evaluation strategy
        if load_best_model_at_end:
            if formatted_val_dataset is not None:
                training_args_dict["evaluation_strategy"] = save_strategy
            else:
                print(
                    "Warning: No validation dataset provided. Setting evaluation_strategy to 'no'."
                )
                training_args_dict["evaluation_strategy"] = "no"
                training_args_dict[
                    "load_best_model_at_end"
                ] = False  # Can't load best model without validation
        else:
            training_args_dict["evaluation_strategy"] = (
                save_strategy if formatted_val_dataset is not None else "no"
            )

        # Create training arguments
        training_args = TrainingArguments(**training_args_dict)

        # Create data collator for efficient batching
        logger.info("Setting up data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficient tensor core utilization
        )

        # Initialize Trainer with all components
        logger.info("Initializing Trainer...")
        self.trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=formatted_train_dataset,
            eval_dataset=formatted_val_dataset,  # This can be None
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=self.tokenizer,
        )

        # Apply response-only training if configured
        if responses_only_config and responses_only_config.get("enabled", False):
            logger.info("Applying Unsloth's train_on_responses_only feature")

            instruction_token = responses_only_config.get("instruction_token", "<|im_start|>user\n")
            response_token = responses_only_config.get("response_token", "<|im_start|>assistant\n")
            instruction_token_id = responses_only_config.get("instruction_token_id", None)
            response_token_id = responses_only_config.get("response_token_id", None)

            logger.info(f"Using instruction token: {instruction_token}")
            logger.info(f"Using response token: {response_token}")

            if instruction_token_id is not None:
                logger.info(f"Using instruction token ID: {instruction_token_id}")
            if response_token_id is not None:
                logger.info(f"Using response token ID: {response_token_id}")

            # Create a backup of the original trainer in case we need it
            original_trainer = self.trainer

            # Apply the response-only training transformation
            try:
                self.trainer = train_on_responses_only(
                    self.trainer,
                    instruction_part=instruction_token,
                    response_part=response_token,
                    instruction_token_id=instruction_token_id,
                    response_token_id=response_token_id,
                )
                logger.info("Successfully applied response-only training transformation")
            except Exception as e:
                logger.error(f"Failed to apply response-only training: {str(e)}")
                logger.info("Falling back to standard training")
                self.trainer = original_trainer

        # Log the final trainer configuration to verify settings
        logger.info(f"Final evaluation_strategy: {self.trainer.args.evaluation_strategy}")

        # Run training
        logger.info("Starting training process...")
        train_result = self.trainer.train()

        # Update model reference and return results
        self.model = model_to_train
        return train_result

    def push_to_hub(self):
        """Push model to HuggingFace Hub"""
        if not self.destination_hub_config:
            raise ValueError("destination_hub_config must be provided to push to hub")

        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")

        try:
            print(f"Pushing model to hub: {self.destination_hub_config.model_id}")
            self.trainer.push_to_hub(
                commit_message="Model trained with QwenTrainer",
                private=self.destination_hub_config.private,
            )
            print("Successfully pushed to hub!")
        except Exception as e:
            print(f"Failed to push to hub: {str(e)}")
            raise

    def save_results(self, results: Dict[str, Any], output_dir: str = "./results") -> str:
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results with max sequence length included
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
            "max_sequence_length": self.max_seq_length,  # Track max sequence length
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast

                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    # Keep as-is if conversion fails
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json

            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file
