from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from src.data.prompt_creator import PromptCreator
from src.data.response_parser import ResponseParser
from src.model.qwen_handler import QwenModelHandler


class ValidationCallback(TrainerCallback):
    """
    Callback for monitoring and managing model validation during training.

    This callback provides functionality to:
    1. Track validation metrics throughout training
    2. Identify and save the best performing model checkpoint
    3. Log validation results for monitoring

    The callback works in conjunction with TrainingArguments to:
    - Monitor specific metrics (configured via metric_for_best_model)
    - Support both minimization and maximization objectives (via greater_is_better)
    - Save best performing checkpoints automatically

    Key Features:
    - Automatic best model tracking
    - Flexible metric monitoring
    - Integration with model checkpointing
    - Support for custom validation schedules

    Args:
        trainer_instance: Instance of QwenTrainer managing the training process

    Attributes:
        trainer: Reference to the trainer instance
        best_metric (float): Best validation metric achieved so far
        best_checkpoint (str): Path to the best performing checkpoint

    Example:
        ```python
        # Create and configure the callback
        validation_callback = ValidationCallback(trainer_instance=trainer)

        # Add to trainer's callback list
        trainer.train(
            callbacks=[validation_callback],
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )

        # Access best results after training
        print(f"Best metric: {validation_callback.best_metric}")
        print(f"Best checkpoint: {validation_callback.best_checkpoint}")
        ```
    """

    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
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
        """
        Called after each validation step during training.

        This method:
        1. Retrieves the current validation metric
        2. Compares it with the best metric so far
        3. Updates the best checkpoint if improvement is found
        4. Logs the results for tracking

        Args:
            args: Training arguments containing configuration
            state: Current training state
            control: Training control object
            metrics: Dictionary of current metrics
            **kwargs: Additional arguments
        """
        # Get validation metric
        metric_to_check = args.metric_for_best_model
        metric_value = metrics.get(metric_to_check)

        if metric_value is not None:
            # Check if this is the best model
            if args.greater_is_better:
                is_best = metric_value > self.best_metric
            else:
                is_best = metric_value < self.best_metric

            if is_best:
                self.best_metric = metric_value
                self.best_checkpoint = state.best_model_checkpoint

                # Log best metric
                metrics["best_" + metric_to_check] = self.best_metric


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for implementing early stopping during training.

    This callback monitors validation metrics and stops training when no improvement
    is seen for a specified number of validation rounds (patience).

    Key Features:
    1. Configurable Patience:
       - Set how many validation rounds to wait for improvement
       - Default patience is 3 rounds

    2. Minimum Improvement Delta:
       - Define minimum change required to consider as improvement
       - Helps avoid stopping due to minor fluctuations

    3. Flexible Metric Monitoring:
       - Works with any validation metric
       - Supports both minimization and maximization objectives

    4. Automatic Training Control:
       - Automatically signals training to stop when criteria are met
       - Integrates seamlessly with the training loop

    Args:
        patience (int, optional): Number of validation rounds to wait for improvement.
            Default is 3.
        min_delta (float, optional): Minimum change in metric to qualify as improvement.
            Default is 0.0.

    Attributes:
        patience (int): Number of rounds to wait
        min_delta (float): Minimum improvement required
        best_metric (float): Best metric value seen so far
        no_improvement_count (int): Number of rounds without improvement

    Example:
        ```python
        # Create callback with custom patience
        early_stopping = EarlyStoppingCallback(
            patience=5,      # Wait for 5 rounds
            min_delta=0.01  # Require 1% improvement
        )

        # Add to trainer's callback list
        trainer.train(
            callbacks=[early_stopping],
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        ```

    Note:
        - The callback automatically adapts to whether the metric should be
          maximized or minimized based on the trainer's configuration.
        - Training will stop when no_improvement_count >= patience
        - Set min_delta higher for metrics with high variance to avoid premature stopping
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
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
        """
        Called after each validation step to check for early stopping conditions.

        This method:
        1. Retrieves the current validation metric
        2. Compares it with the best metric considering min_delta
        3. Updates the no-improvement counter
        4. Signals training to stop if patience is exceeded

        Args:
            args: Training arguments containing configuration
            state: Current training state
            control: Training control object
            metrics: Dictionary of current metrics
            **kwargs: Additional arguments
        """
        # Get validation metric
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
            else:
                self.no_improvement_count += 1

            # Stop training if no improvement for patience epochs
            if self.no_improvement_count >= self.patience:
                control.should_training_stop = True
