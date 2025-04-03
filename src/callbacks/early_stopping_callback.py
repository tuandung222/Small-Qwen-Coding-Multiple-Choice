"""
Early stopping callback to prevent overfitting.
"""

from typing import Dict

from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for implementing early stopping during training.

    Monitors validation metrics and stops training when no improvement
    is seen for a specified number of validation rounds (patience).

    Args:
        patience (int): Number of validation rounds to wait for improvement
        min_delta (float): Minimum change required to qualify as improvement
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        super().__init__()
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
        """Check for improvement after each validation."""
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
                logger.info(f"New best {metric_to_check}: {self.best_metric:.4f}")
            else:
                self.no_improvement_count += 1
                logger.info(
                    f"No improvement in {metric_to_check}: "
                    f"current={metric_value:.4f}, best={self.best_metric:.4f}, "
                    f"patience={self.no_improvement_count}/{self.patience}"
                )

            # Stop training if no improvement for patience epochs
            if self.no_improvement_count >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.no_improvement_count} "
                    f"validations without improvement"
                )
                control.should_training_stop = True

        return control
