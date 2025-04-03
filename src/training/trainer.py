import logging
import os
import shutil
from datetime import datetime
from sys import argv
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from callbacks import (
    EarlyStoppingCallback,
    LRMonitorCallback,
    ModelLoadingAlertCallback,
    ValidationCallback,
)
from src.model.qwen_handler import HubConfig, QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser
from src.utils.auth import setup_authentication

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
        metric_key_prefix="eval",
    ):
        """
        Run validation on the given dataset and return metrics.

        Args:
            val_dataset: The validation dataset
            metric_key_prefix: Prefix for metric keys in the output dictionary

        Returns:
            Dictionary of metrics with values
        """
        logger.info(f"Running validation on dataset with {len(val_dataset)} examples")

        # Use the trainer's evaluate method if available
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "evaluate")
        ):
            logger.info("Using trainer's evaluate method for validation")
            return self.trainer.evaluate(
                eval_dataset=val_dataset, metric_key_prefix=metric_key_prefix
            )

        # If there's no trainer or it doesn't have evaluate, calculate metrics manually
        logger.info("No trainer with evaluate method found. Using manual validation.")

        # Create a simple data loader for validation
        from torch.utils.data import DataLoader

        val_dataloader = DataLoader(
            val_dataset, batch_size=8, shuffle=False  # Use a reasonable batch size for validation
        )

        # Ensure model is in evaluation mode
        model = self.model if self.peft_model is None else self.peft_model
        model.eval()

        import torch
        import torch.nn.functional as F

        total_loss = 0.0
        total_samples = 0

        # Evaluate in batches
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch = {
                        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                    }

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss

                # Update metrics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

        # Create metrics dictionary
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }

        logger.info(f"Validation results: {metrics}")
        return metrics

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

    def prepare_model_for_training(self) -> Any:
        """
        Prepare model for training with LoRA optimizations.
        """
        try:
            print("\033[92mPreparing model with LoRA optimizations...\033[0m")

            # Configure LoRA settings
            from peft import get_peft_model

            if self.lora_config:
                # Extract LoRA parameters
                r = self.lora_config.r
                lora_alpha = self.lora_config.lora_alpha
                lora_dropout = self.lora_config.lora_dropout
                target_modules = self.lora_config.target_modules

                # Apply LoRA with Unsloth optimizations if available
                try:
                    import inspect

                    from unsloth import FastLanguageModel

                    print("\033[92mApplying Unsloth optimizations with LoRA...\033[0m")

                    # Add debug information about the function signature
                    print("\033[93mDebugging Function Parameters:\033[0m")
                    signature = inspect.signature(FastLanguageModel.get_peft_model)
                    print(f"Function signature: {signature}")

                    # Print parameters we're trying to pass
                    print(f"Parameters being used:")
                    print(f"- r: {r}")
                    print(f"- lora_alpha: {lora_alpha}")
                    print(f"- lora_dropout: {lora_dropout}")
                    print(f"- target_modules: {target_modules}")
                    print(f"- use_gradient_checkpointing: unsloth")
                    print(f"- random_state: 42")

                    # Try to get the actual LoraConfig parameters
                    from peft import LoraConfig

                    print("\033[93mLoraConfig accepted parameters:\033[0m")
                    lora_signature = inspect.signature(LoraConfig.__init__)
                    print(f"LoraConfig signature: {lora_signature}")

                    # Explicitly rewrite the whole try-except block to fix indentation errors
                    try:
                        # Try Unsloth's FastLanguageModel.get_peft_model first
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
                        print("\033[92mSuccessfully created PEFT model!\033[0m")
                    except TypeError as e:
                        # Fall back to standard LoRA if FastLanguageModel.get_peft_model fails
                        print(f"\033[91mDetailedError: {e}\033[0m")
                        print("\033[93mAttempting minimal configuration...\033[0m")
                        # Create a standard LoraConfig
                        lora_config = LoraConfig(
                            r=r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            target_modules=target_modules,
                            bias="none",
                            task_type="CAUSAL_LM",
                        )
                        print("Created LoraConfig successfully")

                        # Use standard get_peft_model
                        self.peft_model = get_peft_model(self.model, lora_config)
                        print(
                            "\033[92mSuccessfully created PEFT model with fallback method!\033[0m"
                        )

                    self.peft_model.is_unsloth_model = True
                    self.peft_model.is_qlora_model = False

                    # Enable memory efficient optimizations
                    if hasattr(self.peft_model, "enable_input_require_grads"):
                        self.peft_model.enable_input_require_grads()

                    # Print model information
                    total_params = sum(p.numel() for p in self.peft_model.parameters())
                    trainable_params = sum(
                        p.numel() for p in self.peft_model.parameters() if p.requires_grad
                    )

                    print(f"\033[92mLoRA model prepared:")
                    print(f"- Trainable params: {trainable_params:,}")
                    print(f"- Total params: {total_params:,}")
                    print(f"- Trainable%: {100 * trainable_params / total_params:.4f}")
                    print(f"- Using {'BF16' if is_bf16_supported() else 'FP16'} precision")
                    print(f"- Gradient checkpointing: enabled\033[0m")

                    return self.peft_model
                except ImportError:
                    print("\033[93mUnsloth not available, falling back to standard LoRA...\033[0m")
                    self.peft_model = get_peft_model(self.model, self.lora_config)
                    self.peft_model.is_unsloth_model = False
                    self.peft_model.is_qlora_model = False

                # Enable memory efficient optimizations
                if hasattr(self.peft_model, "enable_input_require_grads"):
                    self.peft_model.enable_input_require_grads()

                # Print model information
                total_params = sum(p.numel() for p in self.peft_model.parameters())
                trainable_params = sum(
                    p.numel() for p in self.peft_model.parameters() if p.requires_grad
                )

                print(f"\033[92mLoRA model prepared:")
                print(f"- Trainable params: {trainable_params:,}")
                print(f"- Total params: {total_params:,}")
                print(f"- Trainable%: {100 * trainable_params / total_params:.4f}")
                print(f"- Using {'BF16' if is_bf16_supported() else 'FP16'} precision")
                print(f"- Gradient checkpointing: enabled\033[0m")

                return self.peft_model
            else:
                # Fallback to standard model preparation
                self.model.train()
                self.model.is_unsloth_model = False
                self.model.is_qlora_model = False
                return self.model

        except Exception as e:
            print(f"\033[91mFailed to prepare model with LoRA: {e}\033[0m")
            import traceback

            print(f"\033[91mError details: {traceback.format_exc()}\033[0m")
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
        Prepare dataset for training with optimized processing and caching.
        Filters out examples that exceed maximum sequence length.
        """
        logger.info("Preparing dataset for training...")
        initial_size = len(dataset)

        # Create cache directory with unique identifier
        cache_dir = os.path.join("outputs", "dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Create cache key based on dataset and processing parameters
        import hashlib

        cache_key = hashlib.md5(
            f"{dataset._fingerprint}_{self.max_seq_length}_{self.prompt_creator.prompt_type}".encode()
        ).hexdigest()
        cache_path = os.path.join(cache_dir, f"processed_{cache_key}.arrow")

        # Try to load cached dataset
        if os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached dataset from {cache_path}")
                return Dataset.load_from_disk(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Process dataset with optimizations
        def format_example(examples):
            # Process multiple examples at once for efficiency
            batch_size = len(examples["question"])
            formatted_examples = []
            skipped_indices = []

            for i in range(batch_size):
                try:
                    # Create prompt and completion
                    prompt = self.prompt_creator.create_training_prompt(
                        question=examples["question"][i], choices=examples["choices"][i]
                    )

                    # Correctly access the yml_str and answer columns
                    yml_str = examples["yml_str"][i] if "yml_str" in examples else None
                    answer = examples["answer"][i] if "answer" in examples else None

                    # Create completion based on available data
                    completion = (
                        yml_str if yml_str else f"The answer is {answer}." if answer else ""
                    )

                    if not completion:
                        logger.warning(f"Example {i} has no completion data (yml_str or answer)")
                        skipped_indices.append(i)
                        continue

                    # Create conversation format
                    conversation = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ]

                    # Apply chat template without tokenization first
                    full_text = self.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=False
                    )

                    # Check sequence length
                    tokens = self.tokenizer.encode(full_text)
                    if len(tokens) > self.max_seq_length:
                        skipped_indices.append(i)
                        logger.warning(
                            f"Skipping example {i}: length {len(tokens)} exceeds maximum {self.max_seq_length}"
                        )
                        continue

                    # If length is okay, add to formatted examples
                    formatted_examples.append(
                        {"text": full_text, "original_index": i, "token_length": len(tokens)}
                    )

                except Exception as e:
                    logger.warning(f"Error processing example {i}: {e}")
                    skipped_indices.append(i)

            if not formatted_examples:
                return {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "skipped": [True],
                    "original_indices": [],
                    "token_lengths": [],
                }

            # Batch tokenize valid examples
            texts = [ex["text"] for ex in formatted_examples]
            tokenized = self.tokenizer(
                texts,
                max_length=self.max_seq_length,
                truncation=False,  # No truncation needed since we already filtered
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": tokenized.input_ids,
                "attention_mask": tokenized.attention_mask,
                "labels": tokenized.input_ids.clone(),
                "skipped": [False] * len(formatted_examples),  # Return a list of booleans
                "original_indices": [ex["original_index"] for ex in formatted_examples],
                "token_lengths": [ex["token_length"] for ex in formatted_examples],
            }

        # Process dataset with optimized batch processing
        logger.info("Processing dataset and filtering long sequences...")
        formatted_dataset = dataset.map(
            format_example,
            batched=True,
            batch_size=100,  # Process 100 examples at once
            num_proc=1,  # Use 1 CPU core to avoid pickling issues
            remove_columns=dataset.column_names,
            load_from_cache_file=True,
            desc="Processing dataset",
        )

        # Filter out examples that were skipped
        formatted_dataset = formatted_dataset.filter(
            lambda x: not x.get("skipped", False), desc="Removing skipped examples"
        )

        # Log statistics
        final_size = len(formatted_dataset)
        filtered_count = initial_size - final_size
        logger.info(
            f"\nDataset preparation complete:"
            f"\n- Initial size: {initial_size}"
            f"\n- Final size: {final_size}"
            f"\n- Filtered out: {filtered_count} examples ({(filtered_count/initial_size)*100:.2f}%)"
            f"\n- Reason: Exceeded maximum sequence length of {self.max_seq_length}"
        )

        # Calculate and log length statistics
        if "token_lengths" in formatted_dataset.features:
            lengths = formatted_dataset["token_lengths"]
            length_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            }
            logger.info(
                f"\nSequence length statistics:"
                f"\n- Minimum: {length_stats['min']}"
                f"\n- Maximum: {length_stats['max']}"
                f"\n- Mean: {length_stats['mean']:.1f}"
                f"\n- Median: {length_stats['median']}"
            )

        # Save processed dataset to cache
        try:
            formatted_dataset.save_to_disk(cache_path)
            logger.info(f"Saved processed dataset to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save dataset cache: {e}")

        # Log to wandb if available
        try:
            import wandb

            if wandb.run:
                wandb.log(
                    {
                        "dataset/initial_size": initial_size,
                        "dataset/final_size": final_size,
                        "dataset/filtered_count": filtered_count,
                        "dataset/filtered_percentage": (filtered_count / initial_size) * 100,
                        "dataset/max_sequence_length": self.max_seq_length,
                    }
                )
                if "token_lengths" in formatted_dataset.features:
                    wandb.log(
                        {
                            "dataset/length_stats": length_stats,
                            "dataset/length_histogram": wandb.Histogram(lengths),
                        }
                    )
        except ImportError:
            pass

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
                    "type": "cosine_with_warmup",
                    "warmup_ratio": warmup_ratio,
                    "warmup_steps": warmup_steps,
                }
            }
        )

        return default_config

    def train(self, train_dataset, val_split=0.1, output_dir="./outputs", **kwargs):
        """Train the model on the given dataset"""
        logger.info("Preparing dataset for training...")

        # Prepare model
        model_to_train = self.prepare_model_for_training()
        if model_to_train is None:
            raise RuntimeError("Model preparation failed")

        # Initialize formatted_val_dataset to avoid UnboundLocalError
        formatted_val_dataset = None

        # Format datasets for training
        formatted_train_dataset = self._prepare_dataset_for_training(train_dataset)

        # Create validation dataset if val_split is provided
        if val_split > 0:
            # Split train dataset to create validation dataset
            logger.info(f"Splitting train dataset with val_split={val_split}")
            split_datasets = formatted_train_dataset.train_test_split(
                test_size=val_split, seed=kwargs.get("random_seed", 42)
            )
            formatted_train_dataset = split_datasets["train"]
            formatted_val_dataset = split_datasets["test"]
            # Save the validation dataset on the trainer instance
            self.val_dataset = formatted_val_dataset
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

        if kwargs.get("push_to_hub_strategy") not in hub_strategy_mapping:
            raise ValueError(
                f"Invalid push_to_hub_strategy: {kwargs.get('push_to_hub_strategy')}. "
                "Valid values are: 'end', 'best', 'all', 'no'"
            )

        # Default optimizer config if not provided
        optimizer_config = kwargs.get(
            "optimizer_config",
            {
                "optimizer_type": "lion_8bit",  # Changed default to Lion 8-bit
                "weight_decay": 0.1,  # Increased for Lion optimizer
                "beta1": 0.95,  # Recommended for Lion
                "beta2": 0.98,  # Recommended for Lion
                "epsilon": 1e-8,
                "max_grad_norm": 1.0,
                "optim_bits": 8,
            },
        )

        logger.info(
            f"Using optimizer: {optimizer_config['optimizer_type']} with weight decay {optimizer_config['weight_decay']}"
        )

        # Default LR scheduler config if not provided
        lr_scheduler_config = kwargs.get(
            "lr_scheduler_config",
            {
                "lr_scheduler_type": "cosine",  # Default to cosine scheduler
                "num_cycles": 1,
                "power": 1.0,
                "last_epoch": -1,
            },
        )

        logger.info(f"Using LR scheduler: {lr_scheduler_config['lr_scheduler_type']}")

        # Calculate warmup steps and ratio
        if kwargs.get("max_steps") is not None and kwargs.get("max_steps") > 0:
            total_steps = kwargs.get("max_steps")
        else:
            total_steps = (
                len(formatted_train_dataset)
                // (
                    kwargs.get("per_device_train_batch_size", 16)
                    * kwargs.get("gradient_accumulation_steps", 1)
                )
                * kwargs.get("num_train_epochs", 1)
            )

        # Handle warmup steps calculation
        if kwargs.get("warmup_steps") is None or kwargs.get("warmup_steps") < 0:
            if kwargs.get("warmup_ratio") is None:
                warmup_ratio = 0.1  # Default warmup ratio
            kwargs["warmup_steps"] = int(total_steps * warmup_ratio)
        else:
            # Calculate effective ratio if warmup_steps is explicitly set
            warmup_ratio = kwargs["warmup_steps"] / total_steps if total_steps > 0 else 0.0

        logger.info(
            f"Using {kwargs['warmup_steps']} warmup steps ({warmup_ratio*100:.1f}% of {total_steps} total steps)"
        )

        # Configure validation strategy
        training_args_dict = {
            # Basic training configuration
            "output_dir": output_dir,
            "per_device_train_batch_size": kwargs.get("per_device_train_batch_size", 16),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
            "learning_rate": kwargs.get("learning_rate", 2e-4),
            "num_train_epochs": kwargs.get("num_train_epochs", 1),
            "warmup_steps": kwargs["warmup_steps"],
            # Learning rate schedule
            "lr_scheduler_type": lr_scheduler_config["lr_scheduler_type"],
            # Logging and saving configuration
            "logging_steps": kwargs.get("logging_steps", 100),
            "save_steps": kwargs.get("save_steps", 1000),
            "save_strategy": kwargs.get("save_strategy", "steps"),
            "save_total_limit": kwargs.get("save_total_limit", 1),
            # Model selection configuration
            "load_best_model_at_end": kwargs.get("load_best_model_at_end", True),
            "metric_for_best_model": kwargs.get("metric_for_best_model", "loss"),
            "greater_is_better": kwargs.get("greater_is_better", False),
            # Mixed precision training
            "fp16": not is_bf16_supported(),
            "bf16": is_bf16_supported(),
            # Lion 8-bit optimizer configuration
            "optim": optimizer_config["optimizer_type"],
            "weight_decay": optimizer_config["weight_decay"],
            "adam_beta1": optimizer_config["beta1"],
            "adam_beta2": optimizer_config["beta2"],
            "adam_epsilon": optimizer_config["epsilon"],
            # Gradient clipping and stability
            "max_grad_norm": optimizer_config["max_grad_norm"],
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            # Integration configuration
            "report_to": kwargs.get("report_to", "wandb"),
            "push_to_hub": bool(self.destination_hub_config),
            "hub_model_id": self.destination_hub_config.model_id
            if self.destination_hub_config
            else None,
            "hub_token": self.destination_hub_config.token if self.destination_hub_config else None,
            "hub_strategy": hub_strategy_mapping[kwargs.get("push_to_hub_strategy", "end")],
            # Dataset configuration
            "remove_unused_columns": kwargs.get("remove_unused_columns", False),
            "dataloader_num_workers": kwargs.get("dataloader_num_workers", 4),
            "dataloader_pin_memory": kwargs.get("dataloader_pin_memory", True),
            # Set random seed
            "seed": kwargs.get("random_seed", 42),
            # Additional stability settings
            "full_determinism": kwargs.get("full_determinism", False),
            "torch_compile": kwargs.get("torch_compile", False),
            "use_cpu": kwargs.get("use_cpu", False),
        }

        # Set evaluation strategy appropriately
        # If load_best_model_at_end is True, we must use the same strategy as save_strategy
        # Otherwise, we can use the user-specified strategy
        if kwargs.get("load_best_model_at_end", True):
            training_args_dict["evaluation_strategy"] = kwargs.get("save_strategy", "steps")
            training_args_dict["eval_steps"] = kwargs.get(
                "eval_steps", kwargs.get("save_steps", 1000)
            )
            training_args_dict["eval_delay"] = kwargs.get("eval_delay", 0)
        else:
            # When handling validation manually, we can set evaluation_strategy to "no"
            training_args_dict["evaluation_strategy"] = "no"

        # Handle max_steps
        if kwargs.get("max_steps") is not None and kwargs.get("max_steps") > 0:
            training_args_dict["max_steps"] = kwargs.get("max_steps")
            training_args_dict.pop("num_train_epochs", None)

        # Create training arguments
        training_args = TrainingArguments(**training_args_dict)

        # Create data collator for efficient batching
        logger.info("Setting up data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficient tensor core utilization
        )

        # Initialize trainer with validation callback
        validation_callback = None
        if kwargs.get("callbacks"):
            for callback in kwargs["callbacks"]:
                if isinstance(callback, ValidationCallback):
                    validation_callback = callback
                    break

        if not validation_callback:
            logger.warning(
                "No ValidationCallback found in callbacks. Validation may not work properly."
            )

        # Initialize callbacks list if None
        if kwargs.get("callbacks") is None:
            kwargs["callbacks"] = []

        # Add safety checkpoint callback
        from src.callbacks import SafetyCheckpointCallback

        safety_checkpoint = SafetyCheckpointCallback(
            save_steps=kwargs.get("save_steps", 1000),
            save_total_limit=kwargs.get("save_total_limit", 1),
        )
        kwargs["callbacks"].append(safety_checkpoint)

        # Initialize trainer
        self.trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=formatted_train_dataset,
            eval_dataset=formatted_val_dataset,  # We handle validation through callback
            data_collator=data_collator,
            callbacks=kwargs["callbacks"],
            tokenizer=self.tokenizer,
        )

        # Store output_dir for validation callback
        self.output_dir = output_dir

        # Check if optimizer type is explicitly set to lion_8bit
        if optimizer_config["optimizer_type"].startswith("lion"):
            logger.info(f"Ensuring use of {optimizer_config['optimizer_type']} optimizer...")
            # Make sure the trainer's args have the correct optimizer
            self.trainer.args.optim = optimizer_config["optimizer_type"]
            # Set other Lion optimizer parameters
            self.trainer.args.adam_beta1 = optimizer_config["beta1"]  # Lion beta1
            self.trainer.args.adam_beta2 = optimizer_config["beta2"]  # Lion beta2

        # Explicitly initialize optimizer and scheduler
        self.trainer.create_optimizer_and_scheduler(
            num_training_steps=kwargs.get("max_steps") or -1
        )

        # Double-check that we're using the right optimizer
        if hasattr(self.trainer, "optimizer"):
            optimizer_class = self.trainer.optimizer.__class__.__name__
            logger.info(f"Confirmed optimizer: {optimizer_class}")

            # If not using the intended optimizer, try to create it manually
            if (
                optimizer_config["optimizer_type"].startswith("lion")
                and "Lion" not in optimizer_class
            ):
                try:
                    logger.warning(
                        f"Optimizer mismatch! Expected Lion but got {optimizer_class}. Trying to fix..."
                    )
                    # Import Lion optimizer
                    from bitsandbytes.optim import Lion8bit

                    # Get all parameter groups
                    param_groups = self.trainer.optimizer.param_groups

                    # Create Lion8bit optimizer
                    new_optimizer = Lion8bit(
                        param_groups,
                        lr=kwargs.get("learning_rate", 2e-4),
                        betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                        weight_decay=optimizer_config["weight_decay"],
                    )

                    # Replace the optimizer
                    self.trainer.optimizer = new_optimizer
                    logger.info(
                        f"Successfully replaced optimizer with {new_optimizer.__class__.__name__}"
                    )

                    # Recreate scheduler with new optimizer
                    from transformers import get_scheduler

                    self.trainer.lr_scheduler = get_scheduler(
                        lr_scheduler_config["lr_scheduler_type"],
                        optimizer=self.trainer.optimizer,
                        num_warmup_steps=kwargs["warmup_steps"],
                        num_training_steps=total_steps,
                    )
                except Exception as e:
                    logger.error(f"Failed to manually create Lion optimizer: {e}")
                    logger.warning(f"Continuing with original optimizer: {optimizer_class}")
            elif not optimizer_config["optimizer_type"].startswith(
                optimizer_class.lower().replace("8bit", "").replace("_", "")
            ):
                logger.warning(
                    f"Using {optimizer_class} instead of requested {optimizer_config['optimizer_type']}. This may impact performance."
                )

        if not hasattr(self.trainer, "lr_scheduler") or self.trainer.lr_scheduler is None:
            logger.warning("LR scheduler not initialized properly, creating manually...")
            from transformers import get_scheduler

            # Create optimizer if not exists
            if not hasattr(self.trainer, "optimizer"):
                self.trainer.create_optimizer()

            # Calculate num_training_steps
            num_update_steps_per_epoch = len(formatted_train_dataset) // (
                kwargs.get("per_device_train_batch_size", 16)
                * kwargs.get("gradient_accumulation_steps", 1)
            )
            num_training_steps = kwargs.get("num_train_epochs", 1) * num_update_steps_per_epoch

            # Create scheduler
            self.trainer.lr_scheduler = get_scheduler(
                lr_scheduler_config["lr_scheduler_type"],
                optimizer=self.trainer.optimizer,
                num_warmup_steps=kwargs["warmup_steps"],
                num_training_steps=num_training_steps,
            )
            logger.info(
                f"Created {lr_scheduler_config['lr_scheduler_type']} scheduler with {kwargs['warmup_steps']} warmup steps"
            )

        # Apply response-only training if configured
        if self.responses_only_config and self.responses_only_config.get("enabled", False):
            logger.info("Applying Unsloth's train_on_responses_only feature")

            instruction_token = self.responses_only_config.get(
                "instruction_token", "<|im_start|>user\n"
            )
            response_token = self.responses_only_config.get(
                "response_token", "<|im_start|>assistant\n"
            )
            instruction_token_id = self.responses_only_config.get("instruction_token_id", None)
            response_token_id = self.responses_only_config.get("response_token_id", None)

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
        try:
            train_result = self.trainer.train()
            return train_result
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def push_to_hub(self, commit_message=None):
        """Push model to HuggingFace Hub"""
        if not self.destination_hub_config:
            raise ValueError("destination_hub_config must be provided to push to hub")

        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Model not initialized or is None")

        try:
            # Create a custom model card
            model_card = self._create_model_card()

            # Save model card to output directory
            output_dir = getattr(self, "output_dir", "./outputs")
            readme_path = os.path.join(output_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(model_card)

            logger.info(f"Created model card at {readme_path}")

            message = commit_message or "Model trained with QwenTrainer"
            try:
                import wandb

                if wandb.run is not None:
                    wandb_link = wandb.run.get_url()
                    message += f" | WandB Run: {wandb_link}"
            except Exception as e:
                pass

            print(
                f"Pushing model to hub: {self.destination_hub_config.model_id} with message: {message}"
            )

            # If we have HF Trainer with push_to_hub method, use it
            if hasattr(self, "trainer") and hasattr(self.trainer, "push_to_hub"):
                self.trainer.push_to_hub(commit_message=message)
            # Otherwise use the model's push_to_hub method
            elif hasattr(self.model, "push_to_hub"):
                self.model.push_to_hub(
                    repo_id=self.destination_hub_config.model_id,
                    token=self.destination_hub_config.token,
                    commit_message=message,
                )
            else:
                raise ValueError("Neither trainer nor model has push_to_hub method")

            print("Successfully pushed to hub!")
        except Exception as e:
            print(f"Failed to push to hub: {str(e)}")
            raise

    def _create_model_card(self):
        """Create a comprehensive model card with training details and validation results"""
        # Get model name
        model_name = self.destination_hub_config.model_id.split("/")[-1]
        base_model = getattr(
            self.model,
            "name_or_path",
            self.model.config.name_or_path if hasattr(self.model, "config") else "unknown",
        )

        # Get training parameters
        training_params = {}
        if hasattr(self, "trainer") and hasattr(self.trainer, "args"):
            args = self.trainer.args
            training_params = {
                "learning_rate": getattr(args, "learning_rate", "N/A"),
                "train_batch_size": getattr(args, "per_device_train_batch_size", "N/A"),
                "eval_batch_size": getattr(args, "per_device_eval_batch_size", 8),
                "seed": getattr(args, "seed", 42),
                "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", "N/A"),
                "total_train_batch_size": getattr(args, "per_device_train_batch_size", 1)
                * getattr(args, "gradient_accumulation_steps", 1),
                "optimizer": f"Use {getattr(args, 'optim', 'N/A')} and the args are: No additional optimizer arguments",
                "lr_scheduler_type": getattr(args, "lr_scheduler_type", "N/A"),
                "lr_scheduler_warmup_steps": getattr(args, "warmup_steps", "N/A"),
                "num_epochs": getattr(args, "num_train_epochs", "N/A"),
                "mixed_precision_training": "Native AMP",
            }

        # Get framework versions
        import datasets
        import peft
        import tokenizers
        import torch
        import transformers

        framework_versions = {
            "PEFT": getattr(peft, "__version__", "N/A"),
            "Transformers": getattr(transformers, "__version__", "N/A"),
            "Pytorch": getattr(torch, "__version__", "N/A"),
            "Datasets": getattr(datasets, "__version__", "N/A"),
            "Tokenizers": getattr(tokenizers, "__version__", "N/A"),
        }

        # Get validation metrics
        metrics = {}
        if hasattr(self, "validation_stats") and self.validation_stats:
            metrics = self.validation_stats

        # Get WandB URL
        wandb_url = ""
        try:
            import wandb

            if wandb.run is not None:
                wandb_url = wandb.run.get_url()
        except:
            pass

        # Get dataset information
        dataset_info = ""
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            train_size = len(self.train_dataset)
            dataset_info += f"Training dataset size: {train_size} examples\n"

        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            val_size = len(self.val_dataset)
            dataset_info += f"Validation dataset size: {val_size} examples\n"

        # Check if we have any example completions
        example_completions = []
        if hasattr(self, "debug_examples") and self.debug_examples:
            example_completions = self.debug_examples

        # Create the model card content
        model_card = f"""# {model_name}

This model is a fine-tuned version of {base_model} on a coding multiple-choice dataset. It uses the LoRA method with {'Unsloth optimizations' if getattr(self.model, 'is_unsloth_model', False) else 'standard PEFT implementation'}.

## Model Performance

"""
        if metrics:
            model_card += "### Validation Metrics\n"
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    model_card += f"- {key}: {value:.4f}\n"
                else:
                    model_card += f"- {key}: {value}\n"
            model_card += "\n"

        model_card += """## Model Description

This model is fine-tuned to excel at coding-related multiple-choice tasks. It demonstrates:
- Strong reasoning capabilities for code understanding
- Ability to evaluate and select the most appropriate answer from multiple choices
- Code explanation and interpretation skills

## Training and Evaluation Data

"""
        model_card += dataset_info
        model_card += """
The model was trained on the coding-mcq-reasoning dataset, which contains coding-related multiple choice questions with reasoning steps. Each example includes a question, multiple choices, and well-reasoned explanations.

## Training Procedure

"""
        model_card += "### Training Hyperparameters\n\n"
        model_card += "The following hyperparameters were used during training:\n\n"
        for param, value in training_params.items():
            model_card += f"- {param}: {value}\n"

        model_card += "\n### Framework Versions\n\n"
        for framework, version in framework_versions.items():
            model_card += f"{framework} {version}\n"

        if wandb_url:
            model_card += f"\n## Experiment Tracking\n\nThis model's training process is tracked on Weights & Biases: [View the full experiment]({wandb_url})\n"

        if example_completions:
            model_card += "\n## Example Completions\n\n"
            # Limited to 3 examples to keep the card readable
            for i, example in enumerate(example_completions[:3]):
                if "raw_text" in example:
                    model_card += (
                        f"### Example {i+1}\n\n```\n{example['raw_text'][:500]}...\n```\n\n"
                    )

        model_card += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")

# Format a multiple choice question
question = "What is the time complexity of the following algorithm: for(int i=0; i<n; i++) { for(int j=i; j<n; j++) { // O(1) operation } }?"
choices = ["A. O(n)", "B. O(n^2)", "C. O(n log n)", "D. O(n(n+1)/2)"]

prompt = f"Question: {question}\nChoices:\n" + "\n".join(choices) + "\nAnswer:"

# Generate a response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

- The model is specialized for coding multiple-choice questions and may not perform as well on open-ended coding tasks
- Performance may vary based on the similarity of questions to the training data
- The model is not a substitute for human judgment in complex coding scenarios
"""

        return model_card

    def save_model(self, output_dir):
        """Save model and tokenizer to the specified directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            logger.info(f"Saving model to {output_dir}")

            # If we're using a PEFT/LoRA model, save it with its specific method
            if hasattr(self, "peft_model") and self.peft_model is not None:
                self.peft_model.save_pretrained(output_dir)
                logger.info("Saved PEFT/LoRA model")
            # Otherwise try to save the model directly if it has save_pretrained
            elif hasattr(self, "model") and hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(output_dir)
                logger.info("Saved base model")
            else:
                raise ValueError("No suitable model found to save")

            # Save the tokenizer if available
            if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_dir)
                logger.info("Saved tokenizer")

            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

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
