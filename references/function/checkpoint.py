import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    checkpoint_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
    save_rng_state: bool = True,
) -> str:
    """
    Save a model checkpoint with associated metadata

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save the checkpoint
        checkpoint_name: Name of the checkpoint
        metadata: Optional metadata to save with the checkpoint
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        save_rng_state: Whether to save RNG state

    Returns:
        str: Path to the saved checkpoint
    """
    try:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save optimizer state if available
        if save_optimizer and hasattr(model, "optimizer"):
            torch.save(model.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

        # Save scheduler state if available
        if save_scheduler and hasattr(model, "scheduler"):
            torch.save(model.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        # Save RNG state
        if save_rng_state:
            rng_states = {
                "python": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(rng_states, os.path.join(checkpoint_dir, "rng_state.pt"))

        # Save metadata
        if metadata:
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        return checkpoint_dir

    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise


def load_checkpoint(
    checkpoint_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_rng_state: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a model checkpoint with associated metadata

    Args:
        checkpoint_path: Path to the checkpoint
        model: The model to load the checkpoint into
        tokenizer: The tokenizer to load the checkpoint into
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        load_rng_state: Whether to load RNG state
        device: Device to load the model onto

    Returns:
        Dict[str, Any]: Metadata from the checkpoint
    """
    try:
        # Load model and tokenizer
        model = model.from_pretrained(checkpoint_path)
        tokenizer = tokenizer.from_pretrained(checkpoint_path)

        if device:
            model = model.to(device)

        # Load optimizer state if available
        if load_optimizer and os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
            optimizer_state = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
            if hasattr(model, "optimizer"):
                model.optimizer.load_state_dict(optimizer_state)

        # Load scheduler state if available
        if load_scheduler and os.path.exists(os.path.join(checkpoint_path, "scheduler.pt")):
            scheduler_state = torch.load(os.path.join(checkpoint_path, "scheduler.pt"))
            if hasattr(model, "scheduler"):
                model.scheduler.load_state_dict(scheduler_state)

        # Load RNG state
        if load_rng_state and os.path.exists(os.path.join(checkpoint_path, "rng_state.pt")):
            rng_states = torch.load(os.path.join(checkpoint_path, "rng_state.pt"))
            torch.set_rng_state(rng_states["python"])
            if torch.cuda.is_available() and rng_states["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_states["cuda"])

        # Load metadata
        metadata = {}
        if os.path.exists(os.path.join(checkpoint_path, "metadata.json")):
            with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
                metadata = json.load(f)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return metadata

    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise


def save_best_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    metrics: Dict[str, float],
    is_best: bool,
    save_total_limit: int = 5,
) -> None:
    """
    Save the best model based on metrics

    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save the model
        metrics: Dictionary of metrics
        is_best: Whether this is the best model so far
        save_total_limit: Maximum number of checkpoints to keep
    """
    try:
        # Create best model directory
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

        # Save metrics
        with open(os.path.join(best_model_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save timestamp
        with open(os.path.join(best_model_dir, "timestamp.txt"), "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # If this is the best model, create a symlink
        if is_best:
            latest_link = os.path.join(output_dir, "latest")
            if os.path.exists(latest_link) and os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(best_model_dir, latest_link, target_is_directory=True)

        # Clean up old checkpoints if needed
        if save_total_limit > 0:
            checkpoints = sorted(
                [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]),
            )
            if len(checkpoints) > save_total_limit:
                for checkpoint in checkpoints[:-save_total_limit]:
                    shutil.rmtree(os.path.join(output_dir, checkpoint))

        logger.info(f"Saved best model to {best_model_dir}")

    except Exception as e:
        logger.error(f"Error saving best model: {str(e)}")
        raise


def save_training_state(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    output_dir: str,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Save the current training state

    Args:
        model: The model
        optimizer: The optimizer
        scheduler: The learning rate scheduler
        epoch: Current epoch
        global_step: Current global step
        output_dir: Directory to save the state
        metrics: Optional metrics to save

    Returns:
        str: Path to the saved state
    """
    try:
        # Create state directory
        state_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(state_dir, exist_ok=True)

        # Save model state
        model.save_pretrained(state_dir)

        # Save optimizer and scheduler state
        torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(state_dir, "scheduler.pt"))

        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if metrics:
            training_state["metrics"] = metrics

        with open(os.path.join(state_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)

        logger.info(f"Saved training state to {state_dir}")
        return state_dir

    except Exception as e:
        logger.error(f"Error saving training state: {str(e)}")
        raise


def load_training_state(
    state_path: str,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training state

    Args:
        state_path: Path to the state
        model: The model to load the state into
        optimizer: The optimizer to load the state into
        scheduler: The scheduler to load the state into
        device: Device to load the model onto

    Returns:
        Dict[str, Any]: Training state
    """
    try:
        # Load model state
        model = model.from_pretrained(state_path)
        if device:
            model = model.to(device)

        # Load optimizer and scheduler state
        optimizer.load_state_dict(torch.load(os.path.join(state_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(state_path, "scheduler.pt")))

        # Load training state
        with open(os.path.join(state_path, "training_state.json"), "r") as f:
            training_state = json.load(f)

        logger.info(f"Loaded training state from {state_path}")
        return training_state

    except Exception as e:
        logger.error(f"Error loading training state: {str(e)}")
        raise
