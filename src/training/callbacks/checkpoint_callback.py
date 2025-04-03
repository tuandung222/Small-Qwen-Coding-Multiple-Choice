"""
Checkpoint callback for best model saving.
"""

import os

from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class CheckpointCallback(BaseCallback):
    """
    Callback for managing checkpoints with best model tracking.

    Features:
    - Tracks best model based on evaluation metrics
    - Saves best model checkpoints
    - Logs improvement metrics
    """

    def __init__(self, output_dir: str):
        super().__init__()
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

            # Log best model metrics
            metrics = {
                "checkpoints/best_eval_loss": self.best_metric,
                "checkpoints/best_step": state.global_step,
            }
            self._log_to_wandb(metrics, state.global_step)

        return control

    def get_best_checkpoint_path(self) -> str:
        """Get the path to the best checkpoint."""
        return self.best_checkpoint if self.best_checkpoint else None
