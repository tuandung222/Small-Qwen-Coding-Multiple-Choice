import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import TrainerCallback

import wandb

from .auth import setup_authentication


class WandBCallback(TrainerCallback):
    """Callback for logging training metrics to W&B"""

    def __init__(self, logger):
        self.logger = logger

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model info at start of training"""
        # Ensure wandb run is initialized
        try:
            if wandb.run is None:
                # Initialize wandb run automatically
                model_name = getattr(args, "hub_model_id", "unknown-model")
                self.logger.init_run(model_name)
                print(f"WandB run initialized: {wandb.run.name}")

            if model:
                self.logger.log_model_info(model)
        except Exception as e:
            print(f"Warning: WandB initialization in callback failed: {e}")
            print("Training will continue without W&B logging")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training"""
        if logs:
            try:
                if wandb.run is None:
                    print(
                        "Warning: WandB run not initialized during logging. Logging to wandb will be skipped."
                    )
                    return
                wandb.log(logs)
            except Exception as e:
                print(f"Warning: Error logging to WandB: {e}")
                print("Continuing training without WandB logging")

    def on_train_end(self, args, state, control, **kwargs):
        """Finish logging when training ends"""
        self.logger.finish_run()

    def log_training_metrics(
        self, logs: Dict[str, Any], state: Any, model: Optional[torch.nn.Module] = None
    ):
        """Log training metrics including gradients and memory"""
        if (
            not hasattr(self.logger.config, "log_training")
            or not self.logger.config.log_training
            or not wandb.run
        ):
            return

        # Log basic training metrics
        if "loss" in logs:
            logs["training/loss"] = logs["loss"]

        # Log learning rate
        if hasattr(state, "learning_rate"):
            logs["training/learning_rate"] = state.learning_rate

        # Log gradient statistics if enabled
        if (
            hasattr(self.logger.config, "log_gradients")
            and self.logger.config.log_gradients
            and model is not None
        ):
            grad_norm = 0.0
            param_norm = 0.0
            total_params = 0

            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_norm += param.data.norm(2).item() ** 2
                    total_params += 1

            if total_params > 0:
                grad_norm = grad_norm**0.5
                param_norm = param_norm**0.5
                logs["training/gradient_norm"] = grad_norm
                logs["training/parameter_norm"] = param_norm
                logs["training/grad_param_ratio"] = grad_norm / param_norm if param_norm > 0 else 0

        # Log memory usage if enabled
        if (
            hasattr(self.logger.config, "log_memory")
            and self.logger.config.log_memory
            and torch.cuda.is_available()
        ):
            logs["training/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
            logs["training/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**2  # MB

        # Log training progress
        if hasattr(state, "max_steps") and state.max_steps > 0:
            logs["training/progress_percentage"] = 100 * state.global_step / state.max_steps

        # Log training time
        if hasattr(self.logger, "train_start_time"):
            logs["training/total_hours"] = (time.time() - self.logger.train_start_time) / 3600
        if hasattr(self.logger, "step_start_time"):
            logs["training/step_time"] = time.time() - self.logger.step_start_time
            self.logger.step_start_time = time.time()

        try:
            wandb.log(logs)
        except Exception as e:
            print(f"Warning: Error logging training metrics to WandB: {e}")
