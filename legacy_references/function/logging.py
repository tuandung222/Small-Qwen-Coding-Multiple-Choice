import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

import wandb

logger = logging.getLogger(__name__)


def setup_logging(
    output_dir: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Set up logging configuration

    Args:
        output_dir: Directory for log files
        level: Logging level
        log_file: Name of log file
        log_format: Format string for log messages
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Set up root logger
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(output_dir, log_file or "train.log")),
            ],
        )

        logger.info(f"Logging set up in {output_dir}")

    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise


def setup_wandb(
    project_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    job_type: str = "train",
    resume: Optional[str] = None,
) -> None:
    """
    Set up Weights & Biases logging

    Args:
        project_name: Name of the project
        config: Configuration dictionary
        tags: Optional tags for the run
        group: Optional group for the run
        job_type: Type of job
        resume: Optional run ID to resume
    """
    try:
        wandb.init(
            project=project_name,
            config=config,
            tags=tags,
            group=group,
            job_type=job_type,
            resume=resume,
        )

        logger.info(f"Initialized W&B project: {project_name}")

    except Exception as e:
        logger.error(f"Error setting up W&B: {str(e)}")
        raise


def setup_tensorboard(
    output_dir: str,
    comment: Optional[str] = None,
) -> SummaryWriter:
    """
    Set up TensorBoard logging

    Args:
        output_dir: Directory for TensorBoard logs
        comment: Optional comment for the run

    Returns:
        SummaryWriter: TensorBoard writer
    """
    try:
        log_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir, comment=comment)
        logger.info(f"Initialized TensorBoard in {log_dir}")

        return writer

    except Exception as e:
        logger.error(f"Error setting up TensorBoard: {str(e)}")
        raise


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
    log_wandb: bool = True,
    log_tensorboard: bool = True,
    writer: Optional[SummaryWriter] = None,
) -> None:
    """
    Log metrics to various backends

    Args:
        metrics: Dictionary of metrics
        step: Current step
        prefix: Prefix for metric names
        log_wandb: Whether to log to W&B
        log_tensorboard: Whether to log to TensorBoard
        writer: TensorBoard writer
    """
    try:
        # Log to W&B
        if log_wandb and wandb.run is not None:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)

        # Log to TensorBoard
        if log_tensorboard and writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"{prefix}{k}", v, step)

        # Log to console
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Metrics at step {step}: {metrics_str}")

    except Exception as e:
        logger.error(f"Error logging metrics: {str(e)}")
        raise


def log_model_graph(
    model: torch.nn.Module,
    input_shape: tuple,
    writer: SummaryWriter,
) -> None:
    """
    Log model graph to TensorBoard

    Args:
        model: Model to log
        input_shape: Shape of input tensor
        writer: TensorBoard writer
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Add graph to TensorBoard
        writer.add_graph(model, dummy_input)
        writer.close()

        logger.info("Logged model graph to TensorBoard")

    except Exception as e:
        logger.error(f"Error logging model graph: {str(e)}")
        raise


def log_gradients(
    model: torch.nn.Module,
    step: int,
    writer: SummaryWriter,
) -> None:
    """
    Log gradient statistics to TensorBoard

    Args:
        model: Model to log gradients for
        step: Current step
        writer: TensorBoard writer
    """
    try:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(
                    f"gradients/{name}",
                    param.grad,
                    step,
                )

        logger.info(f"Logged gradients at step {step}")

    except Exception as e:
        logger.error(f"Error logging gradients: {str(e)}")
        raise


def log_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    writer: SummaryWriter,
) -> None:
    """
    Log learning rate to TensorBoard

    Args:
        optimizer: Optimizer to log learning rate for
        step: Current step
        writer: TensorBoard writer
    """
    try:
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(
                f"learning_rate/group_{i}",
                param_group["lr"],
                step,
            )

        logger.info(f"Logged learning rate at step {step}")

    except Exception as e:
        logger.error(f"Error logging learning rate: {str(e)}")
        raise


def log_memory_usage(
    step: int,
    writer: SummaryWriter,
) -> None:
    """
    Log memory usage to TensorBoard

    Args:
        step: Current step
        writer: TensorBoard writer
    """
    try:
        if torch.cuda.is_available():
            # Log GPU memory
            writer.add_scalar(
                "memory/allocated_gb",
                torch.cuda.memory_allocated() / 1024**3,
                step,
            )
            writer.add_scalar(
                "memory/cached_gb",
                torch.cuda.memory_reserved() / 1024**3,
                step,
            )

        # Log CPU memory
        import psutil

        process = psutil.Process()
        writer.add_scalar(
            "memory/cpu_gb",
            process.memory_info().rss / 1024**3,
            step,
        )

        logger.info(f"Logged memory usage at step {step}")

    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")
        raise


def log_training_time(
    start_time: float,
    step: int,
    writer: SummaryWriter,
) -> None:
    """
    Log training time to TensorBoard

    Args:
        start_time: Start time of training
        step: Current step
        writer: TensorBoard writer
    """
    try:
        elapsed_time = time.time() - start_time
        writer.add_scalar("time/elapsed_seconds", elapsed_time, step)
        writer.add_scalar("time/steps_per_second", step / elapsed_time, step)

        logger.info(f"Logged training time at step {step}")

    except Exception as e:
        logger.error(f"Error logging training time: {str(e)}")
        raise
