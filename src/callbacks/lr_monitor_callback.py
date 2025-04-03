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
                # Check if trainer is accessible
                if self.trainer is None and "trainer" in kwargs:
                    self.trainer = kwargs["trainer"]

                # Get current learning rate - multiple ways to try
                current_lr = None

                # Method 1: From state (most reliable)
                if hasattr(state, "learning_rate"):
                    current_lr = state.learning_rate

                # Method 2: From lr_scheduler
                elif (
                    hasattr(self.trainer, "lr_scheduler") and self.trainer.lr_scheduler is not None
                ):
                    lr_scheduler = self.trainer.lr_scheduler
                    if hasattr(lr_scheduler, "get_last_lr"):
                        lrs = lr_scheduler.get_last_lr()
                        current_lr = lrs[0] if lrs else None

                # Method 3: From optimizer
                elif hasattr(self.trainer, "optimizer") and self.trainer.optimizer is not None:
                    optimizer = self.trainer.optimizer
                    if hasattr(optimizer, "param_groups") and len(optimizer.param_groups) > 0:
                        current_lr = optimizer.param_groups[0].get("lr")

                # Method 4: From Hugging Face Trainer's get_lr_scheduler
                elif hasattr(self.trainer, "trainer") and hasattr(
                    self.trainer.trainer, "lr_scheduler"
                ):
                    lr_scheduler = self.trainer.trainer.lr_scheduler
                    if hasattr(lr_scheduler, "get_last_lr"):
                        lrs = lr_scheduler.get_last_lr()
                        current_lr = lrs[0] if lrs else None

                # If we couldn't get the LR, log a warning and return
                if current_lr is None:
                    logger.warning(
                        "Could not determine learning rate - no lr_scheduler or optimizer found"
                    )
                    return control

                # Get optimizer if available
                optimizer = None
                if hasattr(self.trainer, "optimizer"):
                    optimizer = self.trainer.optimizer
                elif hasattr(self.trainer, "trainer") and hasattr(
                    self.trainer.trainer, "optimizer"
                ):
                    optimizer = self.trainer.trainer.optimizer

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
