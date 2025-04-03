"""
Callback for logging training metrics to Weights & Biases.
"""

import time
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback

import wandb
from src.utils.wandb_logger import WandBLogger


class WandBCallback(TrainerCallback):
    """
    Callback for logging training metrics to W&B with enhanced features.

    Features:
    1. Training metrics logging
    2. Gradient statistics tracking
    3. Memory usage monitoring
    4. Example predictions visualization
    5. Model information logging
    6. Training progress tracking
    """

    def __init__(self, logger: WandBLogger):
        self.logger = logger
        self.train_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model info at start of training"""
        if model:
            self.logger.log_model_info(model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training"""
        if logs:
            # Add training progress information
            if hasattr(state, "max_steps") and state.max_steps > 0:
                logs["training/progress_percentage"] = 100 * state.global_step / state.max_steps

            # Add timing information
            logs["training/total_hours"] = (time.time() - self.train_start_time) / 3600
            logs["training/step_time"] = time.time() - self.step_start_time

            # Log to wandb
            wandb.log(logs)

            # Update step timing
            self.step_start_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        """Finish logging when training ends"""
        # Log final training time
        wandb.log(
            {
                "training/total_time_hours": (time.time() - self.train_start_time) / 3600,
                "training/final_step": state.global_step,
            }
        )

        # Finish the wandb run
        self.logger.finish_run()

    def log_training_metrics(
        self, logs: Dict[str, Any], state: Any, model: Optional[torch.nn.Module] = None
    ):
        """Log comprehensive training metrics including gradients and memory"""
        if not self.logger.config.log_training or not wandb.run:
            return

        # Log basic training metrics
        if "loss" in logs:
            logs["training/loss"] = logs["loss"]

        # Log learning rate
        if hasattr(state, "learning_rate"):
            logs["training/learning_rate"] = state.learning_rate

        # Log gradient statistics if enabled
        if self.logger.config.log_gradients and model is not None:
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
        if self.logger.config.log_memory and torch.cuda.is_available():
            logs["training/gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
            logs["training/gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**2  # MB

        # Log training progress
        if hasattr(state, "max_steps") and state.max_steps > 0:
            logs["training/progress_percentage"] = 100 * state.global_step / state.max_steps

        # Log training time
        logs["training/total_hours"] = (time.time() - self.train_start_time) / 3600
        logs["training/step_time"] = time.time() - self.step_start_time

        wandb.log(logs)
        self.step_start_time = time.time()

    def log_validation_metrics(self, metrics: Dict[str, float], epoch: int, step: int):
        """Log validation metrics with enhanced tracking"""
        if not self.logger.config.log_validation or not wandb.run:
            return

        log_data = {
            "validation/accuracy": metrics.get("accuracy", 0.0),
            "validation/combined_score": metrics.get("combined_score", 0.0),
            "epoch": epoch + 1,
            "step": step,
        }

        # Add quality metrics if available
        if "reasoning_quality" in metrics:
            log_data["validation/reasoning_quality"] = metrics["reasoning_quality"]
        if "quality_accuracy" in metrics:
            log_data["validation/quality_accuracy"] = metrics["quality_accuracy"]

        # Add any additional metrics
        for key, value in metrics.items():
            if key not in ["accuracy", "combined_score", "reasoning_quality", "quality_accuracy"]:
                log_data[f"validation/{key}"] = value

        wandb.log(log_data)

    def log_examples(self, examples: List[Dict[str, Any]], step: int):
        """Log example predictions with enhanced visualization"""
        if not self.logger.config.log_examples or not wandb.run:
            return

        # Create a table of examples
        columns = ["id", "question", "true_answer", "predicted_answer", "reasoning", "is_correct"]
        example_table = wandb.Table(columns=columns)

        for ex in examples:
            example_table.add_data(
                ex["id"],
                ex["question"],
                ex["true_answer"],
                ex["predicted_answer"],
                ex["reasoning"],
                ex["is_correct"],
            )

        # Log table and summary statistics
        wandb.log(
            {
                f"examples/val_{step}": example_table,
                "examples/correct_count": sum(1 for ex in examples if ex["is_correct"]),
                "examples/total_count": len(examples),
                "examples/accuracy": sum(1 for ex in examples if ex["is_correct"]) / len(examples),
            }
        )
