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
        
        # Initialize validation dataset
        self.val_dataset = None

        # Initialize tracking
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_checkpoint = None
        self.validation_history = []
        self.no_improvement_count = 0
        
        # Track last validation step
        self.last_validation_step = -1

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

    def get_cached_or_validate(self, state: Optional[TrainerState] = None) -> Dict[str, float]:
        """Validate model, if needed. Returns metrics."""
        # Get current step if available, otherwise default to 0
        current_step = state.global_step if state and hasattr(state, "global_step") else 0
        
        # Check if validation dataset exists and contains examples
        if self._ensure_validation_dataset():
            # Always run validation without caching
            logger.info(f"Running validation at step {current_step}")
            metrics = self._run_validation()
            self.last_validation_step = current_step
            
            # Call _handle_validation to handle metrics logging and other validation operations
            if metrics:
                self._handle_validation(metrics, current_step)
                
            return metrics
            
        # Return empty dict if we shouldn't validate at this step
        return {}

    def _run_validation(self) -> Dict[str, float]:
        """Run comprehensive validation."""
        try:
            # Try to sync datasets one more time before validation
            self._sync_datasets_with_trainer()
            
            # Get validation dataset - prioritize trainer's dataset
            val_dataset = None
            
            # Check for trainer's validation dataset first (preferred)
            if hasattr(self.trainer, "val_dataset") and self.trainer.val_dataset is not None:
                val_dataset = self.trainer.val_dataset
                logger.info(f"Using trainer's validation dataset with {len(val_dataset)} examples")
            # Fall back to callback's internal validation dataset
            elif hasattr(self, "val_dataset") and self.val_dataset is not None:
                val_dataset = self.val_dataset
                # Also set it on the trainer for future use
                self.trainer.val_dataset = self.val_dataset
                logger.info(f"Using validation callback's internal validation dataset with {len(self.val_dataset)} examples")
            
            # Apply minimal validation sampling if enabled, limiting to max_validation_samples
            val_dataset = self._prepare_validation_dataset(val_dataset)

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
            metrics = {}
            if hasattr(self.trainer, 'evaluate'):
                metrics = self.trainer.evaluate(
                    val_dataset,
                    metric_key_prefix="eval",
                    compute_loss_on_response_only=True,
                )
            elif hasattr(self.trainer, 'validate'):
                metrics = self.trainer.validate(
                    val_dataset,
                    metric_key_prefix="eval",
                )
            else:
                logger.error("Trainer has neither 'evaluate' nor 'validate' method")
                return {}

            # Calculate perplexity from response-only loss
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Add validation step information - without using undefined state variable
            if hasattr(self.trainer, "state"):
                metrics["validation_step"] = self.trainer.state.global_step
                metrics["validation_epoch"] = self.trainer.state.epoch
            else:
                # Get step from the last validation step or default to 0
                metrics["validation_step"] = getattr(self, "last_validation_step", 0)
                metrics["validation_epoch"] = 0.0

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

    def _check_improvement(self, current_metric: float, step: float = None) -> bool:
        """Check if current metric is better than best so far. Optionally log the step at which validation is performed."""
        # Special case for first validation
        if len(self.validation_history) == 0:
            if step is not None:
                logger.info(f"First validation at step {step} with {self.metric_for_best}={current_metric:.4f}")
            else:
                logger.info(f"First validation with {self.metric_for_best}={current_metric:.4f}")
            return True

        if self.greater_is_better:
            is_better = current_metric > (self.best_metric + self.early_stopping_min_delta)
        else:
            is_better = current_metric < (self.best_metric - self.early_stopping_min_delta)

        if is_better:
            if step is not None:
                logger.info(f"Improved {self.metric_for_best} at step {step}: {self.best_metric:.4f} -> {current_metric:.4f}")
            else:
                logger.info(f"Improved {self.metric_for_best}: {self.best_metric:.4f} -> {current_metric:.4f}")

        return is_better

    def _handle_improvement(self, metrics: Dict[str, float], step: int):
        """Handle model improvement."""
        # Update best metric
        self.best_metric = metrics[self.metric_for_best]

        # Save checkpoint
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "output_dir"):
            # Using standard Trainer with args attribute
            checkpoint_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        else:
            # Fallback for QwenTrainer which doesn't have args attribute
            # Try to get output_dir from the output_dir attribute directly or use a default
            output_dir = getattr(self.trainer, "output_dir", None)
            if output_dir is None:
                # Check if state has output_dir
                if hasattr(self.trainer, "state") and hasattr(self.trainer.state, "output_dir"):
                    output_dir = self.trainer.state.output_dir
                else:
                    # Last resort: use a default directory
                    output_dir = os.path.join("outputs", "checkpoints")
                    os.makedirs(output_dir, exist_ok=True)
                    logger.warning(f"Using fallback output directory: {output_dir}")
            
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        
        # Save the model
        try:
            # Check if the trainer is QwenTrainer
            if hasattr(self.trainer, 'save_model') and not isinstance(self.trainer.save_model, bool):
                # QwenTrainer has its own save_model method
                self.trainer.save_model(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
                self.best_checkpoint = checkpoint_dir
            else:
                # Standard HF Trainer - save model directly
                self.trainer.model.save_pretrained(checkpoint_dir)
                if hasattr(self.trainer, "tokenizer") and hasattr(self.trainer.tokenizer, "save_pretrained"):
                    self.trainer.tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
                self.best_checkpoint = checkpoint_dir
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Try alternative saving method
            try:
                if hasattr(self.trainer, "model") and hasattr(self.trainer.model, "save_pretrained"):
                    self.trainer.model.save_pretrained(checkpoint_dir)
                    # Save tokenizer if available
                    if hasattr(self.trainer, "tokenizer") and hasattr(self.trainer.tokenizer, "save_pretrained"):
                        self.trainer.tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint using alternate method to {checkpoint_dir}")
                    self.best_checkpoint = checkpoint_dir
            except Exception as save_error:
                logger.error(f"All checkpoint saving methods failed: {save_error}")

        # Push to hub if configured
        if self.push_to_hub and hasattr(self.trainer, "push_to_hub"):
            try:
                logger.info("Pushing best model to hub...")
                # Avoid passing private parameter which causes issues with create_model_card
                commit_msg = f"Best model at step {step} with {self.metric_for_best}={self.best_metric:.4f}"
                
                # Check if this is QwenTrainer or standard HF Trainer 
                if hasattr(self.trainer, "destination_hub_config"):
                    # QwenTrainer - call with correct parameters
                    self.trainer.push_to_hub(commit_message=commit_msg)
                else:
                    # Standard HF Trainer - avoid private parameter
                    self.trainer.push_to_hub(commit_message=commit_msg)
                
                logger.info("Successfully pushed to hub")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")
                # Print detailed error for debugging
                import traceback
                logger.error(f"Hub push error details: {traceback.format_exc()}")

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
        
        # Synchronize datasets with the trainer to ensure consistency
        self._sync_datasets_with_trainer()

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
                    # Try to create val_dataset using val_split attribute
                    from datasets import Dataset
                    
                    if hasattr(self.trainer.args, "val_split") and self.trainer.args.val_split > 0:
                        train_dataset = self.trainer.train_dataset
                        dataset_size = len(train_dataset)
                        val_size = int(dataset_size * self.trainer.args.val_split)
                        
                        if val_size > 0:
                            import random
                            random.seed(self.trainer.args.seed if hasattr(self.trainer.args, "seed") else 42)
                            
                            # Create validation indices
                            all_indices = list(range(dataset_size))
                            val_indices = random.sample(all_indices, val_size)
                            
                            # Create validation dataset
                            self.val_dataset = Dataset.from_dict(
                                train_dataset.select(val_indices).to_dict()
                            )
                            
                            # Also set it on the trainer for consistency
                            self.trainer.val_dataset = self.val_dataset
                            
                            logger.info(f"Created validation dataset with {len(self.val_dataset)} examples using val_split={self.trainer.args.val_split}")
                        else:
                            logger.warning(f"Calculated val_size is {val_size}, which is too small to create a validation dataset")
                    else:
                        logger.warning("No val_split attribute found on trainer.args")
            except Exception as e:
                logger.error(f"Error creating validation dataset: {e}")

        # Run initial validation if enabled
        if self.validate_at_start:
            logger.info("Running initial validation before training starts...")
            metrics = self._run_validation()

            if not metrics:
                logger.warning("Initial validation returned no metrics. Using dummy values.")
                # Use dummy values for initial validation
                metrics = {
                    self.metric_for_best: 0.0,
                    "eval_loss": 0.0,
                    "eval_perplexity": 1.0,
                    "validation_step": 0.0,
                    "validation_epoch": 0.0,
                    "note": "Initial validation - dummy metrics",
                }

            # Log metrics
            self._log_metrics(metrics, 0)

            # Update best metric
            if metrics.get("note") == "Initial validation - dummy metrics":
                logger.warning("Skipping best model update due to dummy metrics.")
            else:
                self._update_best_metric(metrics, 0)

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
            # Ensure datasets are synced with the trainer before validation
            self._sync_datasets_with_trainer()
            
            # Get validation results without caching
            # This will call _handle_validation internally, so we don't need to handle metrics again here
            metrics = self.get_cached_or_validate(state)

            # If no metrics available, skip further processing
            if not metrics:
                return control

            # Check for early stopping if needed
            if hasattr(self, 'consecutive_non_improvements') and hasattr(self, 'early_stopping_patience'):
                if (self.early_stopping_patience > 0 and 
                    self.consecutive_non_improvements >= self.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered after {self.consecutive_non_improvements} "
                        f"validations without improvement."
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

    def _update_best_metric(self, metrics: Dict[str, Any], step: int):
        """Update the best metric and handle model improvement if better."""
        if self.metric_for_best not in metrics:
            logger.warning(f"{self.metric_for_best} not found in metrics. Cannot update best.")
            return
            
        current_metric = metrics[self.metric_for_best]
        improved = self._check_improvement(current_metric, step)
        
        if improved:
            self._handle_improvement(metrics, step)
            self.no_improvement_count = 0
            logger.info(f"Updated best {self.metric_for_best} to {current_metric:.4f} at step {step}")
        else:
            logger.info(f"No improvement over best {self.metric_for_best}: {self.best_metric:.4f}")
            self.no_improvement_count += 1

    def _sync_datasets_with_trainer(self):
        """
        Synchronize validation dataset with the trainer.
        Ensures both the callback and trainer are using the same validation dataset.
        """
        # Check if the trainer has a validation dataset but we don't
        if (self.val_dataset is None and hasattr(self.trainer, "val_dataset") 
            and self.trainer.val_dataset is not None):
            logger.info("Synchronizing validation dataset from trainer to callback")
            self.val_dataset = self.trainer.val_dataset
            
        # Check if we have a validation dataset but the trainer doesn't
        elif (self.val_dataset is not None and hasattr(self.trainer, "val_dataset") 
              and self.trainer.val_dataset is None):
            logger.info("Synchronizing validation dataset from callback to trainer")
            self.trainer.val_dataset = self.val_dataset
            
        # Log validation dataset information if available
        if self.val_dataset is not None:
            logger.info(f"ValidationCallback now has access to validation dataset with {len(self.val_dataset)} examples")

    def _handle_validation(self, metrics: Dict[str, float], step: int):
        """Handle validation results."""
        # Log metrics
        self._log_metrics(metrics, step)
        
        # Check if metrics have improved over the best so far
        has_improved = self._check_improvement(metrics[self.metric_for_best], step)
        
        # If metrics have improved, handle the improvement
        if has_improved:
            self._handle_improvement(metrics, step)
        
        # Log random example completions for tracing
        self._log_example_completions(step)
        
        # Reset consecutive non-improvement counter or increment it
        if has_improved:
            self.consecutive_non_improvements = 0
        else:
            self.consecutive_non_improvements += 1
            logger.info(
                f"No improvement for {self.consecutive_non_improvements} validation rounds. "
                f"Early stopping patience: {self.early_stopping_patience}"
            )
        
        # Check early stopping
        if (
            self.early_stopping_patience > 0
            and self.consecutive_non_improvements >= self.early_stopping_patience
        ):
            logger.info(
                f"Early stopping triggered after {self.consecutive_non_improvements} "
                f"validation rounds without improvement."
            )
            return {"stop_training": True}
        
        return {"stop_training": False}
    
    def _log_example_completions(self, step: int):
        """Sample random examples from validation dataset and log full completions.
        
        Args:
            step: Current training step
        """
        try:
            # Only proceed if validation dataset is available
            if not hasattr(self, "val_dataset") or self.val_dataset is None:
                logger.warning("No validation dataset available for example completion logging")
                return
                
            # Configure number of examples to sample
            num_examples = min(3, len(self.val_dataset))
            if num_examples == 0:
                return
                
            # Sample random indices
            import random
            import os
            import torch
            indices = random.sample(range(len(self.val_dataset)), num_examples)
            
            # Get the examples
            examples = [self.val_dataset[idx] for idx in indices]
            
            # Create directory for saving completions
            log_dir = os.path.join(
                self.trainer.output_dir if hasattr(self.trainer, "output_dir") else "outputs",
                "example_completions"
            )
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate completions
            model = self.trainer.model
            tokenizer = self.trainer.tokenizer
            
            completion_logs = []
            
            # Log header
            logger.info(f"\n{'='*40}\nGenerating example completions at step {step}\n{'='*40}")
            
            for i, example in enumerate(examples):
                # Get input data
                input_ids = example["input_ids"].unsqueeze(0)  # Add batch dimension
                attention_mask = example["attention_mask"].unsqueeze(0)
                
                # Ensure tensors are on the right device
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                # Find response token position
                if hasattr(self.trainer, "responses_only_config") and self.trainer.responses_only_config:
                    response_token = self.trainer.responses_only_config.get("response_token", "<|im_start|>assistant\n")
                else:
                    response_token = "<|im_start|>assistant\n"  # Default
                
                # Decode input to find user query
                input_text = tokenizer.decode(input_ids[0])
                
                # Extract user query and model's expected completion
                parts = input_text.split(response_token)
                user_query = parts[0] if len(parts) > 0 else input_text
                expected_completion = parts[1] if len(parts) > 1 else ""
                
                # Generate completion with the model
                with torch.inference_mode():
                    try:
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                        # Get only the new tokens (the completion)
                        new_tokens = outputs[0][input_ids.shape[1]:]
                        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        
                        # Log to console
                        logger.info(f"\nExample {i+1}/{num_examples}:")
                        logger.info(f"User Query: {user_query[:100]}...")
                        logger.info(f"Generated: {generated_text[:100]}...")
                        
                        # Save to file
                        example_log_path = os.path.join(log_dir, f"step_{step}_example_{i+1}.txt")
                        with open(example_log_path, "w", encoding="utf-8") as f:
                            f.write(f"STEP: {step}\n")
                            f.write(f"USER QUERY:\n{user_query}\n\n")
                            f.write(f"EXPECTED COMPLETION:\n{expected_completion}\n\n")
                            f.write(f"GENERATED COMPLETION:\n{generated_text}\n")
                        
                        # Add to logs for wandb
                        completion_logs.append({
                            "step": step,
                            "example_idx": indices[i],
                            "user_query": user_query,
                            "expected_completion": expected_completion,
                            "generated_completion": generated_text,
                        })
                    except Exception as e:
                        logger.error(f"Error generating completion for example {i}: {e}")
            
            # Log to wandb if available
            try:
                import wandb
                if wandb.run:
                    # Create table for easy viewing
                    table_data = []
                    for log in completion_logs:
                        table_data.append([
                            log["step"],
                            log["example_idx"],
                            log["user_query"][:200] + "..." if len(log["user_query"]) > 200 else log["user_query"],
                            log["expected_completion"][:200] + "..." if len(log["expected_completion"]) > 200 else log["expected_completion"],
                            log["generated_completion"][:200] + "..." if len(log["generated_completion"]) > 200 else log["generated_completion"],
                        ])
                    
                    columns = ["Step", "Example Index", "User Query", "Expected", "Generated"]
                    wandb.log({
                        f"validation/example_completions_step_{step}": wandb.Table(
                            data=table_data,
                            columns=columns
                        )
                    })
                    
                    # Also log as text for easier reading
                    for i, log in enumerate(completion_logs):
                        wandb.log({
                            f"validation/example_{i+1}/step_{step}": wandb.Html(
                                f"<h3>Example {i+1} at step {step}</h3>"
                                f"<h4>User Query:</h4><pre>{log['user_query']}</pre>"
                                f"<h4>Expected:</h4><pre>{log['expected_completion']}</pre>"
                                f"<h4>Generated:</h4><pre>{log['generated_completion']}</pre>"
                            )
                        })
                    
                    logger.info(f"Logged {len(completion_logs)} example completions to wandb")
            except ImportError:
                logger.info("wandb not available for logging example completions")
                
            logger.info(f"Example completions saved to {log_dir}")
            
        except Exception as e:
            logger.error(f"Error logging example completions: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _ensure_validation_dataset(self):
        """Ensure that a valid, non-empty validation dataset exists."""
        if hasattr(self.trainer, "val_dataset") and self.trainer.val_dataset is not None:
            try:
                return len(self.trainer.val_dataset) > 0
            except Exception:
                return False
        elif self.val_dataset is not None:
            try:
                return len(self.val_dataset) > 0
            except Exception:
                return False
        return False

    def _log_validation_completions(self, completions: list) -> None:
        """Log validation completions by (1) dumping to a local file, (2) logging to wandb, and (3) printing to console."""
        import os
        import wandb
        
        # Determine the output directory from trainer's args; fallback to './outputs'
        output_dir = getattr(self.trainer.args, 'output_dir', './outputs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Dump completions to a local file
        log_file = os.path.join(output_dir, 'validation_completions.txt')
        try:
            with open(log_file, 'w') as f:
                for item in completions:
                    f.write(f"Prompt: {item['prompt']}\n")
                    f.write(f"Completion: {item['completion']}\n")
                    f.write(f"{'-'*50}\n")
            self.logger.info(f"Validation completions dumped to {log_file}")
        except Exception as e:
            self.logger.error(f"Error writing validation completions to file: {e}")

        # Log completions to wandb
        try:
            if wandb.run is not None:
                table_data = [[item['prompt'], item['completion']] for item in completions]
                wandb.log({"validation/completions": wandb.Table(data=table_data, columns=["Prompt", "Completion"])})
                self.logger.info("Validation completions logged to wandb.")
            else:
                self.logger.warning("wandb run not active; skipping wandb logging for validation completions.")
        except Exception as e:
            self.logger.error(f"Error logging validation completions to wandb: {e}")

        # Print completions to console
        for item in completions:
            print("\n" + "="*30 + " Validation Completion " + "="*30)
            print(f"Prompt: {item['prompt']}")
            print(f"Completion: {item['completion']}")
            print("="*70 + "\n")
