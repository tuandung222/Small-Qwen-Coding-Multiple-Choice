"""
Validation callback for comprehensive model evaluation and tracking.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class ValidationCallback(BaseCallback):
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
        super().__init__()
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
            import random

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

            # Run evaluation with response-only loss calculation
            metrics = self.trainer.evaluate(
                val_dataset,
                metric_key_prefix="eval",
                compute_loss_on_response_only=True,
            )

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

            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            import traceback

            logger.error(f"Validation error details: {traceback.format_exc()}")
            return {}

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

    def _save_metrics_history(self, metrics: Dict[str, Any], step: int):
        """Save metrics to history files regardless of improvement status."""
        if not self.metrics_dir:
            return

        try:
            # Save current metrics as JSON
            metrics_file = os.path.join(self.metrics_dir, f"metrics_step_{step}.json")
            with open(metrics_file, "w") as f:
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

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
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
            import wandb

            if wandb.run is not None:
                wandb.run.summary.update(summary)
        except ImportError:
            pass
