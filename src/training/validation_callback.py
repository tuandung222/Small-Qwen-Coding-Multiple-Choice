"""Enhanced validation callback with comprehensive metrics and model pushing."""

import logging
import os
from typing import Dict, Optional

import torch
import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EnhancedValidationCallback(TrainerCallback):
    """
    Enhanced callback for validation, metric tracking, and model pushing.

    Features:
    1. Regular validation every N steps
    2. Comprehensive metric calculation
    3. Automatic model pushing on improvement
    4. Detailed logging to WandB

    Args:
        trainer_instance: The trainer instance
        validation_steps: Number of steps between validations
        push_to_hub: Whether to push improved models to hub
        metric_for_best: Metric to track for best model
        greater_is_better: Whether higher metric is better
    """

    def __init__(
        self,
        trainer_instance,
        validation_steps: int = 50,
        push_to_hub: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.trainer = trainer_instance
        self.validation_steps = validation_steps
        self.push_to_hub = push_to_hub
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better

        # Initialize tracking
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_checkpoint = None
        self.validation_history = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run validation every validation_steps."""
        if state.global_step % self.validation_steps == 0:
            # Run validation
            metrics = self._run_validation()

            # Log metrics
            self._log_metrics(metrics, state.global_step)

            # Check for improvement
            current_metric = metrics.get(self.metric_for_best)
            if current_metric is not None:
                improved = self._check_improvement(current_metric)
                if improved:
                    self._handle_improvement(metrics, state.global_step)

    def _run_validation(self) -> Dict[str, float]:
        """Run comprehensive validation."""
        try:
            # Get validation dataset
            val_dataset = self.trainer.val_dataset
            if val_dataset is None:
                return {}

            # Run evaluation
            metrics = self.trainer.evaluate(val_dataset)

            # Calculate additional metrics
            metrics.update(self._calculate_additional_metrics(val_dataset))

            return metrics
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {}

    def _calculate_additional_metrics(self, val_dataset) -> Dict[str, float]:
        """Calculate additional validation metrics."""
        metrics = {}
        try:
            # Accuracy
            if hasattr(self.trainer, "compute_accuracy"):
                metrics["eval_accuracy"] = self.trainer.compute_accuracy(val_dataset)

            # Perplexity
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Answer preservation rate (if applicable)
            if hasattr(self.trainer, "compute_answer_preservation"):
                metrics["eval_answer_preservation"] = self.trainer.compute_answer_preservation(
                    val_dataset
                )

            # Reasoning quality metrics (if applicable)
            if hasattr(self.trainer, "compute_reasoning_metrics"):
                reasoning_metrics = self.trainer.compute_reasoning_metrics(val_dataset)
                metrics.update(reasoning_metrics)

        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")

        return metrics

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to various destinations."""
        # Log to WandB
        try:
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

        # Log to console
        logger.info(f"Validation metrics at step {step}:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        # Store in history
        self.validation_history.append({"step": step, "metrics": metrics})

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best so far."""
        if self.greater_is_better:
            return current_metric > self.best_metric
        return current_metric < self.best_metric

    def _handle_improvement(self, metrics: Dict[str, float], step: int):
        """Handle model improvement."""
        # Update best metric
        self.best_metric = metrics[self.metric_for_best]

        # Save checkpoint
        checkpoint_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        self.trainer.save_checkpoint(checkpoint_dir)
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

                # Create validation trend plot
                wandb.log(
                    {
                        "validation_trend": wandb.plot.line(
                            table_data=[
                                [h["step"], h["metrics"][self.metric_for_best]]
                                for h in self.validation_history
                            ],
                            columns=["step", self.metric_for_best],
                            title=f"{self.metric_for_best} Trend",
                        )
                    }
                )
        except ImportError:
            pass
