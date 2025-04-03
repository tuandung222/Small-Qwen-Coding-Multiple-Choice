"""
Logger class for Weights & Biases integration.
"""

import time
from typing import Any, Dict, Optional

import torch

import wandb
from src.utils.auth import setup_authentication

from .wandb_config import WandBConfig


class WandBLogger:
    """
    Enhanced logger for W&B integration with comprehensive tracking features.

    Features:
    1. Automatic authentication
    2. Run management
    3. Model information logging
    4. Training metrics tracking
    5. Memory usage monitoring
    6. Gradient statistics
    """

    def __init__(self, config: WandBConfig):
        """
        Initialize WandB logger.

        Args:
            config: WandBConfig instance with logging configuration
        """
        # Setup authentication
        setup_authentication()

        self.config = config
        self.run = None
        self.train_start_time = time.time()
        self.step_start_time = time.time()

    def setup(self):
        """Setup W&B run with configuration."""
        self.run = wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            entity=self.config.entity,
            tags=self.config.tags,
            notes=self.config.notes,
            config=self.config.config,
        )

        # Log configuration
        if self.run:
            wandb.config.update(self.config.to_dict())

    def init_run(self, model_name: str):
        """
        Initialize a new W&B run with model information.

        Args:
            model_name: Name of the model being trained
        """
        # Set run name if not provided
        if not self.config.run_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.config.run_name = f"{model_name}_{timestamp}"

        # Initialize run
        self.setup()

        # Log initial information
        if self.run:
            wandb.log(
                {
                    "model/name": model_name,
                    "training/start_time": time.time(),
                }
            )

    def log_model_info(self, model: torch.nn.Module):
        """
        Log comprehensive model information to W&B.

        Args:
            model: PyTorch model being trained
        """
        if not self.run:
            raise RuntimeError("W&B run not initialized. Call init_run first.")

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Log model architecture information
        model_info = {
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/frozen_parameters": frozen_params,
            "model/trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        }

        # Log layer information if available
        if hasattr(model, "config"):
            model_info.update(
                {
                    "model/hidden_size": getattr(model.config, "hidden_size", None),
                    "model/num_layers": getattr(model.config, "num_hidden_layers", None),
                    "model/attention_heads": getattr(model.config, "num_attention_heads", None),
                }
            )

        # Log memory requirements
        if torch.cuda.is_available():
            model_info.update(
                {
                    "model/initial_gpu_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                    "model/initial_gpu_reserved": torch.cuda.memory_reserved() / 1024**2,  # MB
                }
            )

        # Update wandb summary
        self.run.summary.update(model_info)

        # Also log as a step metric
        wandb.log(model_info)

    def finish_run(self):
        """Finish the W&B run with summary statistics."""
        if self.run:
            # Log final timing information
            wandb.log(
                {
                    "training/end_time": time.time(),
                    "training/total_duration": time.time() - self.train_start_time,
                }
            )

            # Update summary with final metrics
            self.run.summary.update(
                {
                    "training/total_time": time.time() - self.train_start_time,
                    "training/end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # Finish the run
            self.run.finish()
            self.run = None
