import logging
from typing import Any, Dict, Optional

import torch

from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def setup_mixed_precision(
    model: torch.nn.Module,
    training_config: TrainingConfig,
) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Setup mixed precision training

    Args:
        model: Model to configure
        training_config: Training configuration

    Returns:
        Optional[torch.cuda.amp.GradScaler]: GradScaler if using mixed precision
    """
    try:
        if training_config.fp16 or training_config.bf16:
            if training_config.fp16:
                logger.info("Using FP16 mixed precision")
                dtype = torch.float16
            else:
                logger.info("Using BF16 mixed precision")
                dtype = torch.bfloat16

            # Convert model to mixed precision
            model = model.to(dtype=dtype)

            # Create GradScaler for FP16
            if training_config.fp16:
                scaler = torch.cuda.amp.GradScaler()
                return scaler

        else:
            logger.info("Using full precision")

        return None

    except Exception as e:
        logger.error(f"Error setting up mixed precision: {str(e)}")
        raise


def train_step_with_mixed_precision(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    inputs: Dict[str, torch.Tensor],
    gradient_accumulation_steps: int = 1,
    step: int = 0,
) -> Dict[str, float]:
    """
    Perform a training step with mixed precision

    Args:
        model: Model to train
        optimizer: Optimizer to use
        scaler: GradScaler for mixed precision
        inputs: Input tensors
        gradient_accumulation_steps: Number of steps to accumulate over
        step: Current step

    Returns:
        Dict[str, float]: Training metrics
    """
    try:
        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            loss = outputs.loss / gradient_accumulation_steps

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Update weights if accumulating
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metrics = {
            "loss": loss.item() * gradient_accumulation_steps,
        }

        return metrics

    except Exception as e:
        logger.error(f"Error in mixed precision training step: {str(e)}")
        raise


def eval_step_with_mixed_precision(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Perform an evaluation step with mixed precision

    Args:
        model: Model to evaluate
        inputs: Input tensors

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    try:
        model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs.loss

        metrics = {
            "eval_loss": loss.item(),
        }

        return metrics

    except Exception as e:
        logger.error(f"Error in mixed precision evaluation step: {str(e)}")
        raise


def check_mixed_precision_support() -> Dict[str, bool]:
    """
    Check if mixed precision is supported on the current device

    Returns:
        Dict[str, bool]: Support status for different precision types
    """
    try:
        support = {
            "fp16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7,
            "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        }

        logger.info("Mixed precision support:")
        logger.info(f"  FP16: {support['fp16']}")
        logger.info(f"  BF16: {support['bf16']}")

        return support

    except Exception as e:
        logger.error(f"Error checking mixed precision support: {str(e)}")
        raise
