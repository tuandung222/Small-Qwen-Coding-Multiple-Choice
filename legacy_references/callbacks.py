import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from email.headerregistry import DateHeader
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb

# Check for optional dependencies
try:
    from prettytable import PrettyTable

    PRETTYTABLE_AVAILABLE = True
except ImportError:
    PRETTYTABLE_AVAILABLE = False

# pandas is already imported at the top
PANDAS_AVAILABLE = True

from src.model.qwen_handler import QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser

# Setup logger
logger = logging.getLogger(__name__)


class ValidationCallback(TrainerCallback):
    """
    Enhanced callback for validation, metric tracking, and model pushing.

    Features:
    1. Regular validation every N steps
    2. Comprehensive metric calculation
    3. Automatic model pushing on improvement
    4. Detailed logging to WandB
    5. Early stopping support
    6. Initial validation before training starts
    7. Caching of validation results
    """

    def __init__(
        self,
        trainer_instance,
        validation_steps: int = 50,
        push_to_hub: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
        validate_at_start: bool = True,
        minimal_validating: bool = True,
        max_validation_samples: int = 60,
    ):
        self.trainer = trainer_instance
        self.validation_steps = validation_steps
        self.push_to_hub = push_to_hub
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.validate_at_start = validate_at_start
        self.minimal_validating = minimal_validating
        self.max_validation_samples = max_validation_samples

        # Initialize tracking
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_checkpoint = None
        self.validation_history = []
        self.no_improvement_count = 0

        # Cache management
        self.validation_cache = {}  # Store validation results by step
        self.cache_valid_steps = validation_steps  # How long cache remains valid
        self.last_validation_step = -1  # Last step where validation was performed

        # Ensure we have a directory for storing metrics
        self.metrics_dir = None
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "output_dir"):
            self.metrics_dir = os.path.join(self.trainer.args.output_dir, "validation_metrics")
            os.makedirs(self.metrics_dir, exist_ok=True)
            logger.info(f"Validation metrics will be saved to {self.metrics_dir}")

    def _prepare_validation_dataset(self, dataset):
        """
        Prepare validation dataset, optionally limiting its size for minimal validation.

        Args:
            dataset: Original validation dataset

        Returns:
            Dataset: Prepared validation dataset
        """
        if dataset is None:
            return None

        if not self.minimal_validating:
            return dataset

        # Limit validation dataset size
        if len(dataset) > self.max_validation_samples:
            logger.info(
                f"Limiting validation dataset from {len(dataset)} to {self.max_validation_samples} samples"
            )
            # Use random indices for better representation
            indices = random.sample(range(len(dataset)), self.max_validation_samples)
            limited_dataset = dataset.select(indices)
            logger.info(f"Created minimal validation dataset with {len(limited_dataset)} samples")
            return limited_dataset

        return dataset

    def should_validate(self, current_step: int) -> bool:
        """
        Determine if validation should be run at the current step.

        Args:
            current_step: Current training step

        Returns:
            bool: True if validation should be run, False otherwise
        """
        # Always validate at step 0 if validate_at_start is True
        if current_step == 0 and self.validate_at_start:
            return True

        # Run validation if current step is a multiple of validation_steps
        if current_step % self.validation_steps == 0:
            return True

        return False

    def get_cached_or_validate(self, state: TrainerState) -> Dict[str, Any]:
        """
        Get validation results from cache or run validation if needed.

        Args:
            state: Current training state

        Returns:
            Dict containing validation metrics
        """
        current_step = state.global_step

        # Check if we should run validation
        if self.should_validate(current_step):
            # Run validation and update cache
            logger.info(f"Running scheduled validation at step {current_step}")
            metrics = self._run_validation()
            self.validation_cache[current_step] = {
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            self.last_validation_step = current_step
            return metrics

        # Try to use cached results from the most recent validation
        if self.last_validation_step >= 0:
            steps_since_last = current_step - self.last_validation_step
            if steps_since_last < self.cache_valid_steps:
                logger.info(
                    f"Using cached validation results from step {self.last_validation_step} "
                    f"({steps_since_last} steps ago)"
                )
                return self.validation_cache[self.last_validation_step]["metrics"]

        # If cache is expired or no validation has been run yet, run validation
        logger.info(
            f"Cache expired or no validation yet, running validation at step {current_step}"
        )
        metrics = self._run_validation()
        self.validation_cache[current_step] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.last_validation_step = current_step
        return metrics

    def _cleanup_old_cache(self, max_entries: int = 5):
        """Clean up old cache entries to prevent memory bloat."""
        if len(self.validation_cache) > max_entries:
            # Keep only the most recent entries
            sorted_steps = sorted(self.validation_cache.keys())
            for step in sorted_steps[:-max_entries]:
                del self.validation_cache[step]

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run initial validation before training starts if validate_at_start is True."""
        # Verify that we have access to the necessary components
        self._verify_components()

        # Try to create validation dataset if needed
        if hasattr(self.trainer, "val_dataset") and self.trainer.val_dataset is None:
            logger.info(
                "ValidationCallback: attempting to create validation dataset from train dataset"
            )
            try:
                # Check if trainer has train_dataset
                if (
                    hasattr(self.trainer, "train_dataset")
                    and self.trainer.train_dataset is not None
                ):
                    # Get val_split from trainer if available
                    val_split = 0.1  # Default
                    if hasattr(self.trainer, "val_split"):
                        val_split = self.trainer.val_split
                    elif hasattr(self.trainer, "args") and hasattr(self.trainer.args, "val_split"):
                        val_split = self.trainer.args.val_split

                    # Set seed for reproducibility
                    random_seed = 42
                    if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "seed"):
                        random_seed = self.trainer.args.seed

                    logger.info(
                        f"Splitting train dataset with val_split={val_split}, seed={random_seed}"
                    )

                    # Split the dataset
                    from datasets import Dataset

                    if isinstance(self.trainer.train_dataset, Dataset):
                        split_datasets = self.trainer.train_dataset.train_test_split(
                            test_size=val_split, seed=random_seed
                        )
                        # Use only the test split for validation
                        val_dataset = split_datasets["test"]

                        # Apply minimal validation if enabled
                        val_dataset = self._prepare_validation_dataset(val_dataset)

                        self.trainer.val_dataset = val_dataset
                        logger.info(
                            f"Created validation dataset with {len(self.trainer.val_dataset)} examples"
                        )
                    else:
                        logger.warning("Cannot split non-Dataset train_dataset")
            except Exception as e:
                logger.error(f"Failed to create validation dataset: {e}")
                import traceback

                logger.error(f"Dataset creation error details: {traceback.format_exc()}")

        if self.validate_at_start:
            logger.info("Running initial validation before training starts...")
            try:
                # Run validation
                metrics = self._run_validation()

                # Check if metrics are empty (validation failed)
                if not metrics or len(metrics) == 0:
                    logger.warning("Initial validation returned no metrics. Using dummy values.")
                    # Create dummy metrics for logging purposes
                    metrics = {
                        "eval_loss": 0.0,
                        "eval_perplexity": 1.0,
                        "validation_step": 0,
                        "validation_epoch": 0,
                        "note": "Initial validation - dummy metrics",
                    }
                elif "eval_loss" in metrics and metrics["eval_loss"] == 0.0:
                    # This could indicate the validation didn't actually run properly
                    logger.warning(
                        "Initial validation returned suspicious metrics (loss=0). This might indicate an issue."
                    )
                    metrics["note"] = "Initial validation - suspicious metrics"
                else:
                    logger.info("Initial validation completed successfully.")
                    metrics.setdefault("note", "Initial validation successful")

                # Log metrics
                self._log_metrics(metrics, 0)  # Use step 0 for initial validation

                # Check for improvement only if we have real metrics
                if "note" not in metrics or "dummy" not in metrics["note"]:
                    current_metric = metrics.get(self.metric_for_best)
                    if current_metric is not None:
                        improved = self._check_improvement(current_metric)
                        if improved:
                            self._handle_improvement(metrics, 0)
                        else:
                            # If no improvement, still update best metric
                            self.best_metric = current_metric
                            logger.info(f"Initial {self.metric_for_best}: {self.best_metric:.4f}")
                else:
                    logger.warning("Skipping best model update due to dummy metrics.")
            except Exception as e:
                logger.error(f"Initial validation failed with error: {e}")
                import traceback

                logger.error(f"Error details: {traceback.format_exc()}")
                logger.info("Continuing with training despite initial validation failure.")

            logger.info("Initial validation phase completed.")

    def _verify_components(self):
        """Verify that we have access to the necessary components for validation."""
        # Check trainer
        if self.trainer is None:
            logger.error("ValidationCallback is missing trainer instance!")
            return False

        # Check model
        if not hasattr(self.trainer, "model") or self.trainer.model is None:
            logger.error("ValidationCallback cannot access model! Training may fail.")
            return False

        # Check validation dataset
        if hasattr(self.trainer, "val_dataset"):
            if self.trainer.val_dataset is None:
                logger.warning(
                    "ValidationCallback: validation dataset is None. Will create from val_split if possible."
                )

                # Try to check if there's a validation split value to confirm that's expected
                if hasattr(self.trainer, "val_split"):
                    logger.info(f"Expected behavior: val_split is set to {self.trainer.val_split}")
                else:
                    logger.warning("No val_split attribute found on trainer.")
            else:
                dataset_size = (
                    len(self.trainer.val_dataset)
                    if hasattr(self.trainer.val_dataset, "__len__")
                    else "unknown"
                )
                logger.info(
                    f"ValidationCallback: found validation dataset with {dataset_size} examples."
                )

                # Try to get more details about the dataset
                try:
                    if hasattr(self.trainer.val_dataset, "features"):
                        logger.info(
                            f"Dataset features: {list(self.trainer.val_dataset.features.keys())}"
                        )

                    # Check if the dataset has the expected column structure
                    if hasattr(self.trainer.val_dataset, "column_names"):
                        logger.info(f"Dataset columns: {self.trainer.val_dataset.column_names}")
                    elif hasattr(self.trainer.val_dataset, "__getitem__"):
                        # Try to check first example structure
                        first_example = self.trainer.val_dataset[0]
                        if isinstance(first_example, dict):
                            logger.info(f"First example keys: {list(first_example.keys())}")

                            # Check specifically for input_ids and attention_mask which are needed for evaluation
                            required_keys = ["input_ids", "attention_mask", "labels"]
                            missing_keys = [
                                key for key in required_keys if key not in first_example
                            ]
                            if missing_keys:
                                logger.warning(
                                    f"Dataset is missing these required keys: {missing_keys}"
                                )
                            else:
                                logger.info("Dataset contains all required keys for evaluation.")
                except Exception as e:
                    logger.warning(f"Error inspecting validation dataset: {e}")

                return True
        else:
            logger.warning(
                "ValidationCallback cannot access validation dataset! Validation will not work correctly."
            )
            return False

        return True

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run validation based on validation_steps interval."""
        try:
            # Get validation results (from cache or new run)
            metrics = self.get_cached_or_validate(state)

            # Clean up old cache entries
            self._cleanup_old_cache()

            # If no metrics available, skip further processing
            if not metrics:
                return control

            # Always save metrics history, even if it's not the best
            self._save_metrics_history(metrics, state.global_step)

            # Only log metrics if we actually ran validation this step
            if state.global_step == self.last_validation_step:
                self._log_metrics(metrics, state.global_step)

            # Check for improvement
            current_metric = metrics.get(self.metric_for_best)
            if current_metric is not None:
                improved = self._check_improvement(current_metric)
                if improved:
                    self._handle_improvement(metrics, state.global_step)
                    self.no_improvement_count = 0
                else:
                    logger.info(
                        f"No improvement in {self.metric_for_best}: current={current_metric:.4f}, best={self.best_metric:.4f}"
                    )
                    self.no_improvement_count += 1
                    # Check for early stopping
                    if self.no_improvement_count >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {self.no_improvement_count} validations without improvement"
                        )
                        control.should_training_stop = True

        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            import traceback

            logger.debug(f"Validation error details: {traceback.format_exc()}")

        return control

    def _run_validation(self) -> Dict[str, float]:
        """Run comprehensive validation."""
        try:
            # Get validation dataset
            val_dataset = self.trainer.val_dataset
            if val_dataset is None:
                logger.warning("Validation dataset is None. Cannot perform validation.")
                return {}

            # Check if validation dataset is empty
            if hasattr(val_dataset, "__len__") and len(val_dataset) == 0:
                logger.warning(
                    "Validation dataset is empty (zero length). Cannot perform validation."
                )
                return {}

            # Log validation dataset info for debugging
            dataset_size = len(val_dataset) if hasattr(val_dataset, "__len__") else "unknown"
            logger.info(f"Starting validation on dataset with {dataset_size} examples...")

            # Generate completion for a random example
            try:
                # Select random example
                import random

                random_idx = random.randint(0, len(val_dataset) - 1)
                example = val_dataset[random_idx]

                # Get question and choices
                if isinstance(example, dict):
                    # Try to extract original question and choices
                    input_ids = example.get("input_ids", None)
                    if input_ids is not None:
                        full_text = self.trainer.tokenizer.decode(input_ids)
                        # Try to parse the question and choices from the text
                        try:
                            # Assuming format: "Question: ... Choices: A. ... B. ... C. ..."
                            parts = full_text.split("Question: ")
                            if len(parts) > 1:
                                question_part = parts[1].split("Choices:")[0].strip()
                                choices_part = parts[1].split("Choices:")[1].strip()
                            else:
                                question_part = "Could not parse question"
                                choices_part = "Could not parse choices"
                        except:
                            question_part = "Error parsing question"
                            choices_part = "Error parsing choices"
                    else:
                        question_part = example.get("question", "No question found")
                        choices_part = example.get("choices", "No choices found")
                else:
                    question_part = "Could not extract question"
                    choices_part = "Could not extract choices"

                # Generate completion
                model = self.trainer.model
                tokenizer = self.trainer.tokenizer

                # Create prompt using the same format as training
                prompt = f"Question: {question_part}\n\nChoices:\n{choices_part}"

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Generate with standard parameters
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # Decode completion
                completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove the prompt from completion
                if completion.startswith(prompt):
                    completion = completion[len(prompt) :].strip()

                # Save completion to file
                validation_dir = os.path.join(self.trainer.args.output_dir, "validation_examples")
                os.makedirs(validation_dir, exist_ok=True)

                example_data = {
                    "step": self.trainer.state.global_step,
                    "example_index": random_idx,
                    "question": question_part,
                    "choices": choices_part,
                    "model_completion": completion,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save to JSON file
                example_file = os.path.join(
                    validation_dir, f"validation_example_step_{self.trainer.state.global_step}.json"
                )
                with open(example_file, "w") as f:
                    json.dump(example_data, f, indent=2)

                # Log to wandb
                try:
                    import wandb

                    if wandb.run:
                        wandb.log(
                            {
                                "validation/example/step": self.trainer.state.global_step,
                                "validation/example/question": question_part,
                                "validation/example/choices": choices_part,
                                "validation/example/completion": completion,
                                "validation/example": wandb.Table(
                                    columns=["Field", "Content"],
                                    data=[
                                        ["Question", question_part],
                                        ["Choices", choices_part],
                                        ["Model Completion", completion],
                                    ],
                                ),
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to log example to wandb: {e}")

                logger.info("\n=== Random Validation Example ===")
                logger.info(f"Question: {question_part}")
                logger.info(f"Choices: {choices_part}")
                logger.info(f"Model Completion: {completion}")
                logger.info("===============================")

            except Exception as e:
                logger.warning(f"Failed to generate validation example: {e}")
                import traceback

                logger.debug(f"Generation error details: {traceback.format_exc()}")

            # Run evaluation with response-only loss calculation
            logger.info("Running evaluation on validation dataset...")

            # Try to use custom evaluation with progress bar if possible
            try:
                metrics = self._evaluate_with_progress(val_dataset)
                logger.info(
                    f"Custom evaluation with progress bar completed. Found {len(metrics)} metrics."
                )
            except Exception as e:
                logger.warning(
                    f"Custom evaluation with progress bar failed: {e}. Falling back to standard evaluation."
                )
                # Fall back to standard evaluation
                metrics = self.trainer.evaluate(
                    val_dataset,
                    metric_key_prefix="eval",
                    compute_loss_on_response_only=True,
                )
                logger.info(f"Standard evaluation completed. Found {len(metrics)} metrics.")

            # If metrics is None or empty, return empty dict
            if metrics is None:
                logger.warning("Evaluation returned None. Returning empty metrics.")
                return {}

            # Check if eval_loss exists
            if "eval_loss" not in metrics:
                logger.warning("eval_loss not found in metrics. Evaluation may have failed.")

            # Calculate perplexity from response-only loss
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Add validation step information
            metrics["validation_step"] = self.trainer.state.global_step
            metrics["validation_epoch"] = self.trainer.state.epoch

            # Add some model information to metrics
            try:
                metrics["model_name"] = self.trainer.model.config.name_or_path
                metrics["model_type"] = (
                    self.trainer.model.config.model_type
                    if hasattr(self.trainer.model.config, "model_type")
                    else "unknown"
                )
                # Count trainable parameters
                trainable_params = sum(
                    p.numel() for p in self.trainer.model.parameters() if p.requires_grad
                )
                total_params = sum(p.numel() for p in self.trainer.model.parameters())
                metrics["trainable_params"] = trainable_params
                metrics["total_params"] = total_params
                metrics["trainable_percentage"] = round(
                    100 * trainable_params / max(1, total_params), 2
                )
            except Exception as e:
                logger.warning(f"Error adding model information to metrics: {e}")

            # Save sample validation inputs and outputs
            try:
                self._save_validation_samples(val_dataset)
            except Exception as e:
                logger.warning(f"Error saving validation samples: {e}")

            # Log success
            logger.info(f"Validation complete. Found {len(metrics)} metrics.")

            return metrics  # Return the metrics dict

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            import traceback

            logger.error(f"Validation error details: {traceback.format_exc()}")
            return {}

    def _evaluate_with_progress(self, eval_dataset) -> Dict[str, float]:
        """Custom evaluation with optimized batch processing."""
        model = self.trainer.model
        model.eval()

        # Verify dataset is usable
        if eval_dataset is None or not hasattr(eval_dataset, "__len__") or len(eval_dataset) == 0:
            logger.error("Invalid evaluation dataset")
            raise ValueError("Invalid evaluation dataset")

        # Create an optimized dataloader
        try:
            eval_batch_size = (
                self.trainer.args.per_device_eval_batch_size * 2
            )  # Double batch size for eval
            eval_dataloader = DateHeader(
                eval_dataset,
                batch_size=eval_batch_size,
                collate_fn=self.trainer.data_collator,
                num_workers=4,  # Use multiple workers
                pin_memory=True,  # Pin memory for faster GPU transfer
                prefetch_factor=2,  # Prefetch batches
                persistent_workers=True,  # Keep workers alive
            )
            logger.info(f"Created optimized eval dataloader with {len(eval_dataloader)} batches")
        except Exception as e:
            logger.error(f"Error creating evaluation dataloader: {e}")
            raise

        # Setup metrics and device
        device = model.device
        losses = []
        num_eval_steps = 0

        # Optimize CUDA operations
        torch.backends.cudnn.benchmark = True  # Enable CUDNN autotuner

        # Use automatic mixed precision for evaluation
        from torch.cuda.amp import autocast

        scaler = torch.cuda.amp.GradScaler()

        # Batch accumulator for memory efficiency
        accumulated_loss = 0.0
        accumulation_count = 0
        accumulation_steps = 4  # Adjust based on memory

        try:
            with torch.no_grad(), autocast(enabled=True):
                for step, batch in enumerate(tqdm(eval_dataloader, desc="Validation", position=0)):
                    try:
                        # Efficient batch transfer to GPU
                        batch = {
                            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }

                        # Forward pass with memory optimization
                        outputs = model(**batch)
                        loss = outputs.loss.detach().float()

                        # Accumulate loss
                        accumulated_loss += loss.item()
                        accumulation_count += 1

                        # Process accumulated batches
                        if accumulation_count >= accumulation_steps:
                            avg_loss = accumulated_loss / accumulation_count
                            losses.append(avg_loss)
                            accumulated_loss = 0.0
                            accumulation_count = 0

                        num_eval_steps += 1

                    except Exception as batch_error:
                        logger.warning(f"Error processing batch {step}: {batch_error}")
                        continue

                # Process any remaining accumulated loss
                if accumulation_count > 0:
                    avg_loss = accumulated_loss / accumulation_count
                    losses.append(avg_loss)

        except Exception as eval_error:
            logger.error(f"Evaluation error: {eval_error}")
            raise

        # Compute final metrics
        if not losses:
            return {"eval_loss": 0.0, "eval_error": "No valid batches"}

        loss_value = torch.tensor(losses).mean().item()

        # Reset CUDNN benchmark
        torch.backends.cudnn.benchmark = False

        return {
            "eval_loss": loss_value,
            "eval_samples": len(eval_dataset),
            "eval_steps": num_eval_steps,
            "eval_batch_size": eval_batch_size,
        }

    def _save_validation_samples(self, val_dataset, num_samples: int = 3):
        """Save a few sample inputs and outputs from validation."""
        if val_dataset is None or len(val_dataset) == 0:
            return

        # Create directory for validation samples
        output_dir = self.trainer.args.output_dir
        samples_dir = os.path.join(output_dir, "validation_samples")
        os.makedirs(samples_dir, exist_ok=True)

        # Get a few random samples
        import random

        random.seed(42)  # For reproducibility
        sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))

        # Process each sample
        sample_texts = []
        step = self.trainer.state.global_step

        for i, idx in enumerate(sample_indices):
            try:
                example = val_dataset[idx]

                # Get the input text if available
                input_text = ""
                if "text" in example:
                    input_text = example["text"]
                elif "input_ids" in example:
                    input_text = self.trainer.tokenizer.decode(example["input_ids"])

                # Save to file
                sample_file = os.path.join(samples_dir, f"sample_{step}_{i}.txt")
                with open(sample_file, "w", encoding="utf-8") as f:
                    f.write(f"=== VALIDATION SAMPLE {i+1}/{len(sample_indices)} ===\n")
                    f.write(f"Step: {step}\n")
                    f.write("INPUT:\n")
                    f.write(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
                    f.write("\n\n")

                sample_texts.append(f"Sample {i+1}: {input_text[:100]}...")
            except Exception as e:
                logger.warning(f"Error processing validation sample {i}: {e}")

        # Create a summary file
        summary_file = os.path.join(samples_dir, f"samples_summary_step_{step}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Validation Samples Summary (Step {step})\n\n")
            for sample in sample_texts:
                f.write(f"- {sample}\n")

        logger.info(f"Saved {len(sample_texts)} validation samples to {samples_dir}")

    def _calculate_additional_metrics(self, val_dataset) -> Dict[str, float]:
        """Calculate additional validation metrics."""
        metrics = {}
        try:
            # Get validation dataset
            val_dataset = self.trainer.val_dataset
            if val_dataset is None:
                return {}

            # Run evaluation with response-only loss calculation
            metrics = self.trainer.evaluate(
                val_dataset,
                metric_key_prefix="eval",
                compute_loss_on_response_only=True,  # New parameter to indicate response-only loss
            )

            # Calculate perplexity from response-only loss
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Add validation step information
            metrics["validation_step"] = self.trainer.state.global_step
            metrics["validation_epoch"] = self.trainer.state.epoch

        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")

        return metrics

    def _save_metrics_history(self, metrics: Dict[str, Any], step: int):
        """Save metrics to history files regardless of improvement status."""
        if not self.metrics_dir:
            return

        try:
            # Save current metrics as JSON
            metrics_file = os.path.join(self.metrics_dir, f"metrics_step_{step}.json")
            with open(metrics_file, "w") as f:
                import json

                json.dump(metrics, f, indent=2)

            # Update history file
            history_file = os.path.join(self.metrics_dir, "metrics_history.json")
            history_data = []

            # Read existing history if available
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r") as f:
                        history_data = json.load(f)
                except:
                    history_data = []

            # Add current metrics with step info
            metrics_entry = metrics.copy()
            metrics_entry["step"] = step
            history_data.append(metrics_entry)

            # Write updated history
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Saved metrics history to {history_file}")

            # Create a readable summary table
            if PRETTYTABLE_AVAILABLE:
                summary_file = os.path.join(self.metrics_dir, "metrics_summary.txt")
                table = PrettyTable()
                table.field_names = ["Step", self.metric_for_best, "Improved"]

                for entry in history_data:
                    entry_step = entry.get("step", 0)
                    entry_metric = entry.get(self.metric_for_best, "N/A")
                    # Determine if this was an improvement
                    if entry_step == 0:
                        is_improved = "Initial"
                    else:
                        prev_best = (
                            min(
                                [
                                    e.get(self.metric_for_best, float("inf"))
                                    for e in history_data
                                    if e.get("step", 0) < entry_step
                                ]
                            )
                            if not self.greater_is_better
                            else max(
                                [
                                    e.get(self.metric_for_best, float("-inf"))
                                    for e in history_data
                                    if e.get("step", 0) < entry_step
                                ]
                            )
                        )
                        if self.greater_is_better:
                            is_improved = "Yes" if entry_metric > prev_best else "No"
                        else:
                            is_improved = "Yes" if entry_metric < prev_best else "No"

                    table.add_row(
                        [
                            entry_step,
                            f"{entry_metric:.4f}"
                            if isinstance(entry_metric, float)
                            else entry_metric,
                            is_improved,
                        ]
                    )

                with open(summary_file, "w") as f:
                    f.write(str(table))
                logger.info(f"Updated metrics summary table in {summary_file}")

        except Exception as e:
            logger.warning(f"Error saving metrics history: {e}")
            import traceback

            logger.debug(f"Error details: {traceback.format_exc()}")

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best so far."""
        # Special case for first validation
        if len(self.validation_history) == 0:
            logger.info(f"First validation with {self.metric_for_best}={current_metric:.4f}")
            return True

        if self.greater_is_better:
            is_better = current_metric > (self.best_metric + self.early_stopping_min_delta)
        else:
            is_better = current_metric < (self.best_metric - self.early_stopping_min_delta)

        if is_better:
            logger.info(
                f"Improved {self.metric_for_best}: {self.best_metric:.4f} -> {current_metric:.4f}"
            )

        return is_better

    def _handle_improvement(self, metrics: Dict[str, float], step: int):
        """Handle model improvement."""
        # Update best metric
        self.best_metric = metrics[self.metric_for_best]

        # Save checkpoint
        checkpoint_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        self.trainer.save_model(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        self.best_checkpoint = checkpoint_dir

        # Push to hub if configured
        if self.push_to_hub and hasattr(self.trainer, "push_to_hub"):
            try:
                logger.info("Pushing best model to hub...")
                self.trainer.push_to_hub(
                    commit_message=f"Best model at step {step} with {self.metric_for_best}={self.best_metric:.4f}"
                )
                logger.info("Successfully pushed to hub")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")

        # Log improvement
        logger.info(
            f"New best model at step {step} with {self.metric_for_best}={self.best_metric:.4f}"
        )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Summarize validation history at end of training."""
        # Create validation summary
        summary = {
            "best_metric": self.best_metric,
            "best_step": next(
                (
                    h["step"]
                    for h in self.validation_history
                    if h["metrics"][self.metric_for_best] == self.best_metric
                ),
                None,
            ),
            "total_validations": len(self.validation_history),
            "validation_trend": [
                h["metrics"][self.metric_for_best] for h in self.validation_history
            ],
        }

        # Log summary
        logger.info("\nValidation Summary:")
        logger.info(f"Best {self.metric_for_best}: {summary['best_metric']:.4f}")
        logger.info(f"Best step: {summary['best_step']}")
        logger.info(f"Total validations: {summary['total_validations']}")

        # Log to WandB
        try:
            if wandb.run is not None:
                wandb.run.summary.update(summary)
        except ImportError:
            pass

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to various destinations."""
        # Log to console
        logger.info(f"\nValidation metrics at step {step}:")
        for key, value in metrics.items():
            # Format differently based on value type
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            else:
                # Just convert to string for non-numeric values
                logger.info(f"{key}: {str(value)}")

        # Log to wandb
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

        # Store in history
        self.validation_history.append({"step": step, "metrics": metrics})


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for implementing early stopping during training.

    This callback monitors validation metrics and stops training when no improvement
    is seen for a specified number of validation rounds (patience).

    Key Features:
    1. Configurable Patience:
       - Set how many validation rounds to wait for improvement
       - Default patience is 3 rounds

    2. Minimum Improvement Delta:
       - Define minimum change required to consider as improvement
       - Helps avoid stopping due to minor fluctuations

    3. Flexible Metric Monitoring:
       - Works with any validation metric
       - Supports both minimization and maximization objectives

    4. Automatic Training Control:
       - Automatically signals training to stop when criteria are met
       - Integrates seamlessly with the training loop

    Args:
        patience (int, optional): Number of validation rounds to wait for improvement.
            Default is 3.
        min_delta (float, optional): Minimum change in metric to qualify as improvement.
            Default is 0.0.

    Attributes:
        patience (int): Number of rounds to wait
        min_delta (float): Minimum improvement required
        best_metric (float): Best metric value seen so far
        no_improvement_count (int): Number of rounds without improvement

    Example:
        ```python
        # Create callback with custom patience
        early_stopping = EarlyStoppingCallback(
            patience=5,      # Wait for 5 rounds
            min_delta=0.01  # Require 1% improvement
        )

        # Add to trainer's callback list
        trainer.train(
            callbacks=[early_stopping],
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        ```

    Note:
        - The callback automatically adapts to whether the metric should be
          maximized or minimized based on the trainer's configuration.
        - Training will stop when no_improvement_count >= patience
        - Set min_delta higher for metrics with high variance to avoid premature stopping
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.no_improvement_count = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """
        Called after each validation step to check for early stopping conditions.

        This method:
        1. Retrieves the current validation metric
        2. Compares it with the best metric considering min_delta
        3. Updates the no-improvement counter
        4. Signals training to stop if patience is exceeded

        Args:
            args: Training arguments containing configuration
            state: Current training state
            control: Training control object
            metrics: Dictionary of current metrics
            **kwargs: Additional arguments
        """
        # Get validation metric
        metric_to_check = args.metric_for_best_model
        metric_value = metrics.get(metric_to_check)

        if metric_value is not None:
            # Check if this is the best model
            if args.greater_is_better:
                is_improvement = metric_value > (self.best_metric + self.min_delta)
            else:
                is_improvement = metric_value < (self.best_metric - self.min_delta)

            if is_improvement:
                self.best_metric = metric_value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Stop training if no improvement for patience epochs
            if self.no_improvement_count >= self.patience:
                control.should_training_stop = True


class LRMonitorCallback(TrainerCallback):
    """Custom callback to track learning rates during training."""

    def __init__(self, trainer=None):
        self.trainer = trainer

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step % args.logging_steps == 0:
            try:
                # Get learning rate scheduler
                lr_scheduler = self.trainer.lr_scheduler
                optimizer = self.trainer.optimizer

                # Get current learning rate
                if hasattr(lr_scheduler, "get_last_lr"):
                    lrs = lr_scheduler.get_last_lr()
                    current_lr = lrs[0] if lrs else None
                else:
                    # Fallback - try to get from optimizer
                    current_lr = optimizer.param_groups[0]["lr"]

                # Log to wandb
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "trainer/learning_rate": current_lr,
                                "trainer/global_step": state.global_step,
                                "trainer/epoch": state.epoch,
                                "trainer/total_steps": state.max_steps,
                                "trainer/percent_complete": state.global_step
                                / state.max_steps
                                * 100
                                if state.max_steps
                                else 0,
                            }
                        )

                        # Also log optimizer parameters
                        if optimizer and hasattr(optimizer, "param_groups"):
                            for i, param_group in enumerate(optimizer.param_groups):
                                # Log parameters like weight decay, momentum, etc.
                                for key, value in param_group.items():
                                    if key != "params" and not isinstance(value, (list, tuple)):
                                        wandb.log({f"optimizer/group{i}_{key}": value})
                except ImportError:
                    logger.warning("wandb not installed, skipping logging")
            except Exception as e:
                logger.warning(f"Error logging learning rate: {e}")
        return control


class PromptMonitorCallback(TrainerCallback):
    """Custom callback to show random prompts during training with enhanced visualization."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        logging_steps: int = 10,
        save_to_file: bool = True,
        log_to_wandb: bool = True,
        max_prompts_to_save: int = 100,
        analyze_tokens: bool = True,
        show_token_stats: bool = True,
        output_dir: Optional[str] = None,
        track_diversity: bool = True,
        track_quality: bool = True,
        enable_interactive: bool = False,
        categorize_prompts: bool = True,
        enable_comparison: bool = False,
    ):
        """
        Initialize the prompt monitor callback with enhanced features.

        Args:
            dataset: The training dataset
            tokenizer: The tokenizer used for the model
            logging_steps: Number of steps between showing prompts
            save_to_file: Whether to save prompts to a file in the experiment folder
            log_to_wandb: Whether to log prompts to wandb
            max_prompts_to_save: Maximum number of prompts to save to file
            analyze_tokens: Whether to analyze token distribution
            show_token_stats: Whether to show token statistics in terminal
            output_dir: Directory to save prompt files (defaults to trainer output_dir)
            track_diversity: Whether to track prompt diversity over time
            track_quality: Whether to track prompt quality metrics
            enable_interactive: Whether to enable interactive prompt selection
            categorize_prompts: Whether to categorize prompts automatically
            enable_comparison: Whether to enable prompt comparison features
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.logging_steps = logging_steps
        self.last_prompt = None
        self.last_prompt_idx = None
        self.save_to_file = save_to_file
        self.log_to_wandb = log_to_wandb
        self.max_prompts_to_save = max_prompts_to_save
        self.analyze_tokens = analyze_tokens
        self.show_token_stats = show_token_stats
        self.output_dir = output_dir
        self.prompt_history = []
        self.trainer = None
        self.prompt_file = None
        self.prompt_file_path = None

        # New features
        self.track_diversity = track_diversity
        self.track_quality = track_quality
        self.enable_interactive = enable_interactive
        self.categorize_prompts = categorize_prompts
        self.enable_comparison = enable_comparison

        # Initialize diversity tracking
        self.diversity_scores = []
        self.last_prompt_embedding = None

        # Initialize quality tracking
        self.quality_scores = []

        # Initialize categorization
        self.prompt_categories = {}
        self.category_counts = {}

        # Initialize comparison data
        self.comparison_prompts = []

        # For interactive mode
        self.interactive_mode = False
        self.marked_prompts = set()

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Initialize prompt file at the beginning of training."""
        if self.save_to_file and self.output_dir is None:
            self.output_dir = args.output_dir

        if self.save_to_file and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.prompt_file_path = os.path.join(self.output_dir, "prompt_history.json")
            self.prompt_file = open(self.prompt_file_path, "w")
            self.prompt_file.write("[\n")  # Start JSON array
            logger.info(f"Saving prompts to {self.prompt_file_path}")

            # Create additional files for enhanced features
            if self.track_diversity:
                self.diversity_file_path = os.path.join(self.output_dir, "prompt_diversity.json")
                with open(self.diversity_file_path, "w") as f:
                    f.write("[]")
                logger.info(f"Created diversity tracking file: {self.diversity_file_path}")

            if self.track_quality:
                self.quality_file_path = os.path.join(self.output_dir, "prompt_quality.json")
                with open(self.quality_file_path, "w") as f:
                    f.write("[]")
                logger.info(f"Created quality tracking file: {self.quality_file_path}")

            if self.categorize_prompts:
                self.categories_file_path = os.path.join(self.output_dir, "prompt_categories.json")
                with open(self.categories_file_path, "w") as f:
                    f.write("{}")
                logger.info(f"Created categories file: {self.categories_file_path}")

            # Add an initial example if dataset is available
            try:
                if self.dataset and len(self.dataset) > 0:
                    idx = 0
                    example = self.dataset[idx]
                    prompt = example.get("text", "No text available")

                    # Create an initial prompt entry
                    if prompt:
                        self._process_prompt(prompt, idx, state.global_step, state.epoch)
                        logger.info("Added initial prompt example to monitoring files")
            except Exception as e:
                logger.warning(f"Failed to add initial prompt example: {e}")

        return control

    def _process_prompt(self, prompt: str, idx: int, step: int, epoch: float) -> Dict[str, Any]:
        """Process a prompt and generate all necessary data."""
        # Analyze tokens if enabled
        token_analysis = {}
        if self.analyze_tokens:
            token_analysis = self._analyze_tokens(prompt)

        # Calculate quality metrics if enabled
        quality_metrics = {}
        if self.track_quality:
            quality_metrics = self._calculate_prompt_quality(prompt)
            self.quality_scores.append({"step": step, "metrics": quality_metrics})

            # Update quality file immediately
            if hasattr(self, "quality_file_path") and os.path.exists(self.quality_file_path):
                try:
                    import json

                    with open(self.quality_file_path, "w") as f:
                        json.dump(self.quality_scores, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update quality file: {e}")

        # Calculate diversity score if enabled
        diversity_score = 0.0
        if self.track_diversity:
            diversity_score = self._calculate_prompt_diversity(prompt)
            self.diversity_scores.append({"step": step, "score": diversity_score})

            # Update diversity file immediately
            if hasattr(self, "diversity_file_path") and os.path.exists(self.diversity_file_path):
                try:
                    import json

                    with open(self.diversity_file_path, "w") as f:
                        json.dump(self.diversity_scores, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update diversity file: {e}")

        # Categorize prompt if enabled
        category = "unknown"
        if self.categorize_prompts:
            category = self._categorize_prompt(prompt)
            self.category_counts[category] = self.category_counts.get(category, 0) + 1

            # Update categories file immediately
            if hasattr(self, "categories_file_path") and os.path.exists(self.categories_file_path):
                try:
                    import json

                    with open(self.categories_file_path, "w") as f:
                        json.dump(self.category_counts, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to update categories file: {e}")

        # Create prompt data
        prompt_data = {
            "step": step,
            "epoch": epoch,
            "prompt": prompt,
            "example_idx": idx,
            "token_analysis": token_analysis,
            "quality_metrics": quality_metrics,
            "diversity_score": diversity_score,
            "category": category,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        self._save_prompt_to_file(prompt_data)

        # Log to wandb
        self._log_to_wandb(prompt_data)

        return prompt_data

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Show a random prompt at each logging step with enhanced visualization."""
        if state.global_step % self.logging_steps == 0:
            try:
                # Sample a random example that's different from the last one
                max_attempts = 5  # Limit attempts to avoid infinite loop
                for _ in range(max_attempts):
                    idx = random.randint(0, len(self.dataset) - 1)
                    if idx != self.last_prompt_idx:
                        break

                example = self.dataset[idx]
                self.last_prompt_idx = idx

                # Get the prompt text
                prompt = example.get("text", "")
                if not prompt:
                    logger.warning(f"Example {idx} has no text. Skipping.")
                    return control

                # Only show if it's different from the last one
                if prompt != self.last_prompt:
                    # Get current training loss from state if available
                    training_loss = None
                    if hasattr(state, "log_history") and state.log_history:
                        try:
                            for entry in reversed(state.log_history):
                                if "loss" in entry:
                                    training_loss = entry["loss"]
                                    break
                        except:
                            pass

                    # Process the prompt
                    prompt_data = self._process_prompt(prompt, idx, state.global_step, state.epoch)

                    # Add training loss to prompt data if available
                    if training_loss is not None:
                        prompt_data["training_loss"] = training_loss

                        # Also log to wandb
                        try:
                            import wandb

                            if wandb.run is not None:
                                wandb.log(
                                    {
                                        "prompts/with_loss/step": state.global_step,
                                        "prompts/with_loss/loss": training_loss,
                                        "prompts/with_loss/token_count": prompt_data[
                                            "token_analysis"
                                        ]["token_count"],
                                    }
                                )
                        except:
                            pass

                    # Handle interactive mode if enabled
                    if self.enable_interactive and self.interactive_mode:
                        self._handle_interactive_mode(prompt_data)

                    # Update last prompt
                    self.last_prompt = prompt
            except Exception as e:
                logger.warning(f"Error showing prompt: {e}")

        return control

    def _analyze_tokens(self, text: str) -> Dict[str, Any]:
        """Analyze token distribution and provide comprehensive metrics."""
        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)
            decoded_tokens = [self.tokenizer.decode([t]) for t in tokens]

            # Basic token counts
            token_count = len(tokens)
            unique_tokens = len(set(tokens))
            unique_ratio = unique_tokens / token_count if token_count > 0 else 0

            # Count token frequencies
            token_freqs = {}
            for t in tokens:
                token_freqs[t] = token_freqs.get(t, 0) + 1

            # Get top tokens by frequency (include decoded version for readability)
            top_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)[:20]
            top_tokens_decoded = []
            for token_id, count in top_tokens:
                try:
                    token_text = self.tokenizer.decode([token_id])
                    # Replace control characters with descriptive text
                    token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
                    if not token_text or token_text.isspace():
                        token_text = f"[{token_id}]"
                    top_tokens_decoded.append((token_text, count))
                except:
                    top_tokens_decoded.append((str(token_id), count))

            # Analyze token length distribution
            token_lengths = [len(t) for t in decoded_tokens]
            length_dist = {}
            for length in token_lengths:
                length_dist[length] = length_dist.get(length, 0) + 1

            # Calculate sequence entropy (measure of information content)
            import math

            entropy = 0
            for count in token_freqs.values():
                prob = count / token_count
                entropy -= prob * math.log2(prob)

            # Calculate average token length
            avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0

            # Count special tokens
            special_tokens = 0
            if hasattr(self.tokenizer, "special_tokens_map"):
                special_token_ids = set()
                for token_name, token_value in self.tokenizer.special_tokens_map.items():
                    try:
                        # Handle both single tokens and lists of tokens
                        if isinstance(token_value, str):
                            token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                            if token_id != self.tokenizer.unk_token_id:
                                special_token_ids.add(token_id)
                        elif isinstance(token_value, list):
                            for t in token_value:
                                token_id = self.tokenizer.convert_tokens_to_ids(t)
                                if token_id != self.tokenizer.unk_token_id:
                                    special_token_ids.add(token_id)
                    except:
                        pass

                special_tokens = sum(1 for t in tokens if t in special_token_ids)

            # Calculate vocabulary coverage
            vocab_size = getattr(self.tokenizer, "vocab_size", 0)
            vocab_coverage = unique_tokens / vocab_size if vocab_size > 0 else 0

            # Return comprehensive analysis
            return {
                "token_count": token_count,
                "unique_tokens": unique_tokens,
                "unique_ratio": unique_ratio,
                "entropy": entropy,
                "avg_token_length": avg_token_length,
                "max_token_length": max(token_lengths) if token_lengths else 0,
                "min_token_length": min(token_lengths) if token_lengths else 0,
                "special_tokens": special_tokens,
                "special_token_ratio": special_tokens / token_count if token_count > 0 else 0,
                "vocab_coverage": vocab_coverage,
                "top_tokens": top_tokens_decoded,
                "token_length_dist": length_dist,
            }
        except Exception as e:
            logger.warning(f"Error analyzing tokens: {e}")
            return {"token_count": 0, "unique_tokens": 0, "unique_ratio": 0, "error": str(e)}

    def _calculate_prompt_quality(self, text: str) -> Dict[str, float]:
        """Calculate quality metrics for the prompt."""
        # Basic quality metrics
        metrics = {
            "length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": sum(len(word) for word in text.split()) / max(1, len(text.split())),
            "sentence_count": text.count(".") + text.count("!") + text.count("?"),
            "question_count": text.count("?"),
            "code_block_count": text.count("```"),
        }

        # Calculate complexity (simple approximation)
        words = text.split()
        unique_words = set(words)
        metrics["vocabulary_richness"] = len(unique_words) / max(1, len(words))

        # Calculate readability (Flesch-Kincaid approximation)
        if metrics["sentence_count"] > 0:
            metrics["readability"] = (
                0.39 * (len(words) / metrics["sentence_count"])
                + 11.8 * (len(unique_words) / len(words))
                - 15.59
            )
        else:
            metrics["readability"] = 0

        return metrics

    def _calculate_prompt_diversity(self, text: str) -> float:
        """Calculate diversity score compared to previous prompts."""
        if not self.prompt_history:
            return 1.0  # First prompt is considered maximally diverse

        # Simple token-based diversity
        current_tokens = set(self.tokenizer.encode(text))

        # Calculate Jaccard similarity with previous prompts
        similarities = []
        for prev_prompt in self.prompt_history[-5:]:  # Compare with last 5 prompts
            prev_tokens = set(prev_prompt.get("token_analysis", {}).get("token_ids", []))
            if prev_tokens:
                intersection = len(current_tokens.intersection(prev_tokens))
                union = len(current_tokens.union(prev_tokens))
                similarity = intersection / max(1, union)
                similarities.append(similarity)

        # Diversity is inverse of average similarity
        avg_similarity = sum(similarities) / max(1, len(similarities))
        diversity = 1.0 - avg_similarity

        return diversity

    def _categorize_prompt(self, text: str) -> str:
        """Categorize the prompt based on content."""
        # Simple keyword-based categorization
        text_lower = text.lower()

        if "function" in text_lower or "def " in text_lower:
            return "function_definition"
        elif "class" in text_lower or "object" in text_lower:
            return "class_definition"
        elif "error" in text_lower or "exception" in text_lower:
            return "error_handling"
        elif "loop" in text_lower or "for " in text_lower or "while " in text_lower:
            return "loops"
        elif "if " in text_lower or "else" in text_lower:
            return "conditionals"
        elif "import" in text_lower or "from " in text_lower:
            return "imports"
        elif "return" in text_lower:
            return "return_statements"
        elif "?" in text_lower:
            return "questions"
        elif "```" in text_lower:
            return "code_blocks"
        else:
            return "general"

    def _save_prompt_to_file(self, prompt_data: Dict[str, Any]) -> None:
        """Save prompt data to file."""
        if not self.save_to_file or not self.prompt_file:
            return

        # Add comma if not the first entry
        if self.prompt_history:
            self.prompt_file.write(",\n")

        # Write prompt data as JSON
        import json

        json.dump(prompt_data, self.prompt_file, indent=2)

        # Add to history
        self.prompt_history.append(prompt_data)

        # Trim history if needed
        if len(self.prompt_history) > self.max_prompts_to_save:
            logger.warning(f"Reached maximum prompts to save ({self.max_prompts_to_save})")

        # Update category counts
        if self.categorize_prompts and "category" in prompt_data:
            category = prompt_data["category"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def _log_to_wandb(self, prompt_data: Dict[str, Any]) -> None:
        """Log prompt data to wandb with enhanced visualizations and metrics."""
        if not self.log_to_wandb:
            return

        try:
            import wandb

            if wandb.run is not None:
                # Create a table for the prompt with more detailed information
                columns = [
                    "step",
                    "epoch",
                    "prompt",
                    "token_count",
                    "unique_tokens",
                    "category",
                    "diversity_score",
                ]

                data = [
                    prompt_data["step"],
                    prompt_data.get("epoch", 0),
                    prompt_data["prompt"],
                    prompt_data["token_analysis"]["token_count"],
                    prompt_data["token_analysis"]["unique_tokens"],
                    prompt_data.get("category", "unknown"),
                    prompt_data.get("diversity_score", 0.0),
                ]

                # Create detailed prompt table for this step
                prompt_table = wandb.Table(columns=columns, data=[data])

                # Log basic metrics
                log_data = {
                    # Current prompt details as table
                    "prompts/current": prompt_table,
                    # Basic token statistics
                    "prompts/token_count": prompt_data["token_analysis"]["token_count"],
                    "prompts/unique_tokens": prompt_data["token_analysis"]["unique_tokens"],
                    "prompts/unique_ratio": prompt_data["token_analysis"].get("unique_ratio", 0),
                    # Tracking step and epoch
                    "prompts/step": prompt_data["step"],
                    "prompts/epoch": prompt_data.get("epoch", 0),
                }

                # Log example index if available
                if "example_idx" in prompt_data:
                    log_data["prompts/example_idx"] = prompt_data["example_idx"]

                # Create histograms for token lengths if available
                if "token_length_dist" in prompt_data["token_analysis"]:
                    length_dist = prompt_data["token_analysis"]["token_length_dist"]
                    length_table = wandb.Table(
                        columns=["length", "count"], data=[[k, v] for k, v in length_dist.items()]
                    )
                    log_data["prompts/token_length_dist"] = wandb.plot.bar(
                        length_table, "length", "count", title="Token Length Distribution"
                    )

                # Top tokens visualization
                if "top_tokens" in prompt_data["token_analysis"]:
                    top_tokens = prompt_data["token_analysis"]["top_tokens"]
                    if isinstance(top_tokens, dict):
                        top_tokens = list(top_tokens.items())
                    elif (
                        isinstance(top_tokens, list)
                        and len(top_tokens) > 0
                        and isinstance(top_tokens[0], tuple)
                    ):
                        # Already in the right format
                        pass
                    else:
                        # Convert to proper format if needed
                        try:
                            top_tokens = [(str(t), c) for t, c in top_tokens]
                        except:
                            top_tokens = []

                    if top_tokens:
                        # Limit to top 15 tokens for cleaner visualization
                        if len(top_tokens) > 15:
                            top_tokens = top_tokens[:15]

                        tokens_table = wandb.Table(
                            columns=["token", "count"], data=[[k, v] for k, v in top_tokens]
                        )
                        log_data["prompts/top_tokens"] = wandb.plot.bar(
                            tokens_table, "token", "count", title="Top Tokens in Prompt"
                        )

                # Log quality metrics as radar chart if available
                if "quality_metrics" in prompt_data and prompt_data["quality_metrics"]:
                    # Log individual metrics
                    for metric, value in prompt_data["quality_metrics"].items():
                        log_data[f"prompts/quality/{metric}"] = value

                    # Create radar chart of all metrics
                    quality_data = [[k, v] for k, v in prompt_data["quality_metrics"].items()]
                    if quality_data:
                        quality_table = wandb.Table(columns=["metric", "value"], data=quality_data)
                        log_data["prompts/quality_radar"] = wandb.plot.bar(
                            quality_table, "metric", "value", title="Prompt Quality Metrics"
                        )

                # Log diversity score if available
                if "diversity_score" in prompt_data:
                    log_data["prompts/diversity_score"] = prompt_data["diversity_score"]

                # Track history of prompt categories as a stacked bar chart
                if self.category_counts:
                    category_table = wandb.Table(
                        data=[[k, v] for k, v in self.category_counts.items()],
                        columns=["category", "count"],
                    )
                    log_data["prompts/category_distribution"] = wandb.plot.bar(
                        category_table, "category", "count", title="Prompt Categories"
                    )

                # Create a histogram of all prompt lengths seen so far
                if self.prompt_history:
                    try:
                        prompt_lengths = [
                            p["token_analysis"]["token_count"] for p in self.prompt_history
                        ]
                        length_data = [[i, l] for i, l in enumerate(prompt_lengths)]
                        length_histogram = wandb.Table(
                            columns=["index", "length"], data=length_data
                        )
                        log_data["prompts/length_history"] = wandb.plot.line(
                            length_histogram, "index", "length", title="Prompt Length History"
                        )
                    except Exception as e:
                        logger.warning(f"Error creating prompt length history: {e}")

                # Log all data to wandb
                wandb.log(log_data)

                # Optionally save the full prompt to wandb artifacts for later inspection
                if (
                    prompt_data["step"] % (self.logging_steps * 10) == 0
                ):  # Save every 10th logged prompt
                    try:
                        import json
                        import tempfile

                        # Create a temporary file to store the prompt data
                        with tempfile.NamedTemporaryFile(
                            mode="w", delete=False, suffix=".json"
                        ) as tmp:
                            json.dump(prompt_data, tmp, indent=2)
                            prompt_file = tmp.name

                        # Create and save an artifact
                        artifact = wandb.Artifact(
                            name=f"prompt_step_{prompt_data['step']}",
                            type="prompt",
                            description=f"Training prompt at step {prompt_data['step']}",
                        )
                        artifact.add_file(prompt_file)
                        wandb.log_artifact(artifact)

                        # Clean up temporary file
                        os.unlink(prompt_file)
                    except Exception as e:
                        logger.warning(f"Error saving prompt artifact: {e}")

        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")
            import traceback

            logger.debug(f"Wandb logging error details: {traceback.format_exc()}")

    def _handle_interactive_mode(self, prompt_data: Dict[str, Any]) -> None:
        """Handle interactive prompt selection if enabled."""
        if not self.enable_interactive:
            return

        try:
            import sys

            print("\n" + "=" * 80)
            print("INTERACTIVE PROMPT MODE")
            print("=" * 80)
            print(f"Step: {prompt_data['step']}")
            print(f"Category: {prompt_data.get('category', 'unknown')}")
            print(f"Token Count: {prompt_data['token_analysis']['token_count']}")
            print("-" * 80)
            print(prompt_data["prompt"])
            print("-" * 80)
            print("Options:")
            print("  [s] Save this prompt for later comparison")
            print("  [m] Mark this prompt as interesting")
            print("  [c] Continue to next prompt")
            print("  [q] Quit interactive mode")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "s":
                self.comparison_prompts.append(prompt_data)
                print(f"Saved prompt for comparison. Total saved: {len(self.comparison_prompts)}")
            elif choice == "m":
                self.marked_prompts.add(prompt_data["step"])
                print("Marked prompt as interesting")
            elif choice == "q":
                self.interactive_mode = False
                print("Exiting interactive mode")

        except Exception as e:
            logger.warning(f"Error in interactive mode: {e}")

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Close prompt file at the end of training and save additional data."""
        if self.save_to_file and self.prompt_file:
            self.prompt_file.write("\n]")  # End JSON array
            self.prompt_file.close()
            logger.info(f"Saved {len(self.prompt_history)} prompts to {self.prompt_file_path}")

            # Save additional data with final updates
            if self.track_diversity and hasattr(self, "diversity_file_path"):
                try:
                    import json

                    with open(self.diversity_file_path, "w") as f:
                        json.dump(self.diversity_scores, f, indent=2)
                    logger.info(f"Saved final diversity scores to {self.diversity_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save final diversity scores: {e}")

            if self.track_quality and hasattr(self, "quality_file_path"):
                try:
                    import json

                    with open(self.quality_file_path, "w") as f:
                        json.dump(self.quality_scores, f, indent=2)
                    logger.info(f"Saved final quality scores to {self.quality_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save final quality scores: {e}")

            if self.categorize_prompts and hasattr(self, "categories_file_path"):
                try:
                    import json

                    with open(self.categories_file_path, "w") as f:
                        json.dump(self.category_counts, f, indent=2)
                    logger.info(f"Saved final category counts to {self.categories_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save final category counts: {e}")

            if self.enable_comparison and self.comparison_prompts:
                try:
                    import json

                    comparison_file = os.path.join(self.output_dir, "prompt_comparisons.json")
                    with open(comparison_file, "w") as f:
                        json.dump(self.comparison_prompts, f, indent=2)
                    logger.info(f"Saved comparison data to {comparison_file}")
                except Exception as e:
                    logger.warning(f"Failed to save comparison data: {e}")

        return control


class ModelLoadingAlertCallback(TrainerCallback):
    """Callback to alert when model loading method changes and warn about QLoRA optimizations."""

    def __init__(self, use_unsloth: bool = True):
        """
        Initialize the model loading alert callback.

        Args:
            use_unsloth: Whether Unsloth was attempted for model loading
        """
        self.use_unsloth = use_unsloth
        self.alert_shown = False
        self.trainer = None
        self.warning_count = 0
        self.max_warnings = 3  # Show warning multiple times during training

    def _calculate_memory_usage(self, model):
        """Calculate approximate memory usage for QLoRA."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Estimate memory usage for QLoRA
            base_model_memory = total_params * 0.5 / (1024**3)  # GB for 4-bit model
            lora_memory = trainable_params * 4 / (1024**3)  # GB for LoRA weights (FP32)
            optimizer_memory = trainable_params * 8 / (1024**3)  # GB for Adam states
            activation_memory = 2  # GB (approximate)

            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "base_model_memory_gb": base_model_memory,
                "lora_memory_gb": lora_memory,
                "optimizer_memory_gb": optimizer_memory,
                "activation_memory_gb": activation_memory,
                "total_memory_gb": base_model_memory
                + lora_memory
                + optimizer_memory
                + activation_memory,
            }
        except Exception as e:
            logger.warning(f"Failed to calculate memory usage: {e}")
            return None

    def _show_performance_warning(self, model):
        """Show detailed performance warning with QLoRA-specific information."""
        memory_info = self._calculate_memory_usage(model)

        warning_msg = "\n" + "!" * 80 + "\n"
        warning_msg += "⚠️  CRITICAL PERFORMANCE WARNING  ⚠️\n"
        warning_msg += "!" * 80 + "\n\n"

        warning_msg += "Training is proceeding WITHOUT proper QLoRA optimizations! This means:\n"
        warning_msg += "1. 💾 SIGNIFICANTLY HIGHER MEMORY USAGE:\n"
        if memory_info:
            warning_msg += (
                f"   - Base model memory (4-bit): {memory_info['base_model_memory_gb']:.2f} GB\n"
            )
            warning_msg += f"   - LoRA weights memory: {memory_info['lora_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Optimizer states: {memory_info['optimizer_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Activation memory: {memory_info['activation_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Total estimated memory: {memory_info['total_memory_gb']:.2f} GB\n"
            warning_msg += f"   - With proper QLoRA, this could be reduced by up to 80%!\n"

        warning_msg += "\n2. ⚡ TRAINING INEFFICIENCIES:\n"
        warning_msg += "   - Missing 4-bit quantization optimizations\n"
        warning_msg += "   - Suboptimal memory access patterns\n"
        warning_msg += "   - Higher CPU-GPU transfer overhead\n"

        warning_msg += "\n3. 🔧 MISSING QLORA FEATURES:\n"
        warning_msg += "   - No double quantization\n"
        warning_msg += "   - No NF4 data type benefits\n"
        warning_msg += "   - No paged optimizer states\n"
        warning_msg += "   - No efficient k-bit training\n"

        warning_msg += "\nRECOMMENDED ACTIONS:\n"
        warning_msg += "1. Stop training (Ctrl+C)\n"
        warning_msg += "2. Ensure proper QLoRA setup:\n"
        warning_msg += "   - Install bitsandbytes>=0.39.0\n"
        warning_msg += "   - Install transformers>=4.31.0\n"
        warning_msg += "   - Install accelerate>=0.20.3\n"
        warning_msg += "   - Install peft>=0.4.0\n"
        warning_msg += "3. Check BitsAndBytes configuration\n"
        warning_msg += "4. Verify k-bit training preparation\n"
        warning_msg += "5. Restart training with proper QLoRA settings\n"

        warning_msg += "\n" + "!" * 80 + "\n"

        print(warning_msg)

        # Also log to wandb if available
        try:
            import wandb

            if wandb.run:
                wandb.alert(
                    title="⚠️ Training Without QLoRA Optimizations",
                    text=warning_msg,
                    level=wandb.AlertLevel.ERROR,
                )

                # Log memory metrics
                if memory_info:
                    wandb.log(
                        {
                            "memory/base_model_gb": memory_info["base_model_memory_gb"],
                            "memory/lora_weights_gb": memory_info["lora_memory_gb"],
                            "memory/optimizer_gb": memory_info["optimizer_memory_gb"],
                            "memory/activation_gb": memory_info["activation_memory_gb"],
                            "memory/total_gb": memory_info["total_memory_gb"],
                        }
                    )
        except ImportError:
            pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Show alert at the beginning of training if QLoRA was not properly configured."""
        if self.use_unsloth and not self.alert_shown:
            try:
                if self.trainer is None:
                    logger.warning("Trainer not available in ModelLoadingAlertCallback")
                    return control

                if not hasattr(self.trainer, "model") or self.trainer.model is None:
                    logger.warning("Model not available in ModelLoadingAlertCallback")
                    return control

                model = self.trainer.model
                if not hasattr(model, "is_qlora_model") or not model.is_qlora_model:
                    self._show_performance_warning(model)
                    self.alert_shown = True
                    self.warning_count += 1

            except Exception as e:
                logger.warning(f"Error checking model loading method: {e}")
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Show periodic warnings during training."""
        if self.use_unsloth and self.warning_count < self.max_warnings:
            if state.global_step % 100 == 0:  # Show warning every 100 steps
                try:
                    model = self.trainer.model
                    if not hasattr(model, "is_qlora_model") or not model.is_qlora_model:
                        self._show_performance_warning(model)
                        self.warning_count += 1
                except Exception as e:
                    logger.warning(f"Error checking model during training: {e}")
        return control


class CheckpointCallback(TrainerCallback):
    """Callback for managing checkpoints with safety considerations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_metric = float("inf")
        self.best_checkpoint = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Save best checkpoint based on evaluation metrics."""
        if "eval_loss" in metrics and metrics["eval_loss"] < self.best_metric:
            self.best_metric = metrics["eval_loss"]
            self.best_checkpoint = os.path.join(
                self.output_dir, f"best-checkpoint-{state.global_step}"
            )
            # Save best model
            kwargs["trainer"].save_model(self.best_checkpoint)
            logger.info(
                f"New best model saved at step {state.global_step} "
                f"with eval_loss: {self.best_metric:.4f}"
            )

        return control
