"""
Learning rate monitoring callback.
"""

from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class LRMonitorCallback(BaseCallback):
    """
    Callback to track learning rates during training.

    Features:
    - Monitors learning rate changes
    - Logs to wandb with detailed metrics
    - Tracks optimizer parameters
    """

    def __init__(self, trainer=None):
        super().__init__()
        self.trainer = trainer

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Log learning rate and optimizer metrics at each logging step."""
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

                # Prepare metrics
                metrics = {
                    "trainer/learning_rate": current_lr,
                    "trainer/global_step": state.global_step,
                    "trainer/epoch": state.epoch,
                    "trainer/total_steps": state.max_steps,
                    "trainer/percent_complete": (
                        state.global_step / state.max_steps * 100 if state.max_steps else 0
                    ),
                }

                # Log optimizer parameters if available
                if optimizer and hasattr(optimizer, "param_groups"):
                    for i, param_group in enumerate(optimizer.param_groups):
                        # Log parameters like weight decay, momentum, etc.
                        for key, value in param_group.items():
                            if key != "params" and not isinstance(value, (list, tuple)):
                                metrics[f"optimizer/group{i}_{key}"] = value

                # Log to wandb
                self._log_to_wandb(metrics, state.global_step)

            except Exception as e:
                logger.warning(f"Error logging learning rate: {e}")

        return control
