"""
Safety checkpoint callback for regular model saving.
"""

import os
import shutil

from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class SafetyCheckpointCallback(BaseCallback):
    """
    Callback to save checkpoints at regular intervals for safety.

    Features:
    - Regular checkpoint saving at specified intervals
    - Automatic cleanup of old checkpoints
    - Configurable number of checkpoints to keep
    """

    def __init__(self, save_steps: int = 30, save_total_limit: int = 5):
        super().__init__()
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.saved_checkpoints = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save checkpoint every save_steps."""
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            # Create checkpoint directory
            checkpoint_dir = os.path.join(args.output_dir, f"safety-checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Get trainer instance - try from kwargs or fallback to attribute
            trainer = None
            if "trainer" in kwargs:
                trainer = kwargs["trainer"]
            elif hasattr(self, "trainer"):
                trainer = self.trainer

            if trainer is None:
                logger.warning(
                    f"SafetyCheckpointCallback: No trainer found at step {state.global_step}. Cannot save checkpoint."
                )
                return control

            # Save model
            trainer.save_model(checkpoint_dir)
            logger.info(f"Saved safety checkpoint to {checkpoint_dir}")

            # Add to saved checkpoints list
            self.saved_checkpoints.append(checkpoint_dir)

            # Remove old checkpoints if exceeding limit
            while len(self.saved_checkpoints) > self.save_total_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                try:
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old safety checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

            # Log checkpoint info to wandb
            try:
                metrics = {
                    "checkpoints/total_saved": len(self.saved_checkpoints),
                    "checkpoints/last_save_step": state.global_step,
                }
                self._log_to_wandb(metrics, state.global_step)
            except Exception as e:
                logger.warning(f"Failed to log checkpoint info to wandb: {e}")

        return control
