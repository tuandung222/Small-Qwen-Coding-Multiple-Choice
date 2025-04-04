import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments

from src.config.training_config import TrainingConfig, ValidationConfig, WandBConfig

from .logging import log_gradients, log_learning_rate, log_memory_usage, log_metrics
from .optimization import OptimizerConfig, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for model training
    """

    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None


def setup_training_args(
    training_config: TrainingConfig,
    validation_config: ValidationConfig,
    wandb_config: WandBConfig,
    output_dir: str,
) -> TrainingArguments:
    """
    Setup training arguments for the Trainer

    Args:
        training_config: Training configuration
        validation_config: Validation configuration
        wandb_config: Weights & Biases configuration
        output_dir: Directory to save outputs

    Returns:
        TrainingArguments: Configured training arguments
    """
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=validation_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=validation_config.eval_steps,
            evaluation_strategy=validation_config.evaluation_strategy,
            save_strategy=training_config.save_strategy,
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=validation_config.load_best_model_at_end,
            metric_for_best_model=validation_config.metric_for_best_model,
            greater_is_better=validation_config.greater_is_better,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            gradient_checkpointing=training_config.gradient_checkpointing,
            report_to=wandb_config.report_to if wandb_config.enabled else "none",
            run_name=wandb_config.run_name,
            group=wandb_config.group,
            tags=wandb_config.tags,
        )

        logger.info("Training arguments:")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Number of epochs: {training_config.num_train_epochs}")
        logger.info(f"Batch size: {training_config.per_device_train_batch_size}")
        logger.info(f"Learning rate: {training_config.learning_rate}")
        logger.info(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")

        return training_args

    except Exception as e:
        logger.error(f"Error setting up training arguments: {str(e)}")
        raise


def setup_wandb(
    wandb_config: WandBConfig,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    lora_config: Dict[str, Any],
) -> None:
    """
    Setup Weights & Biases logging

    Args:
        wandb_config: Weights & Biases configuration
        model_config: Model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
    """
    try:
        if not wandb_config.enabled:
            return

        import wandb

        wandb.init(
            project=wandb_config.project,
            name=wandb_config.run_name,
            group=wandb_config.group,
            tags=wandb_config.tags,
            config={
                "model": model_config,
                "training": training_config,
                "lora": lora_config,
            },
        )

        logger.info(f"Initialized Weights & Biases logging for project: {wandb_config.project}")

    except Exception as e:
        logger.error(f"Error setting up Weights & Biases: {str(e)}")
        raise


def save_training_results(
    results: Dict[str, Any],
    output_dir: str,
    experiment_name: str,
) -> str:
    """
    Save training results to a JSON file

    Args:
        results: Training results to save
        output_dir: Directory to save results
        experiment_name: Name of the experiment

    Returns:
        str: Path to the saved results file
    """
    try:
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, f"{experiment_name}_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved training results to: {results_file}")

        return results_file

    except Exception as e:
        logger.error(f"Error saving training results: {str(e)}")
        raise


def compute_metrics(eval_preds: tuple) -> Dict[str, float]:
    """
    Compute metrics for evaluation

    Args:
        eval_preds: Tuple of predictions and labels

    Returns:
        Dict[str, float]: Computed metrics
    """
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score

        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)

        accuracy = accuracy_score(labels, predictions)

        return {
            "accuracy": accuracy,
        }

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def train_epoch(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: TrainingConfig = None,
    device: torch.device = None,
    epoch: int = 0,
    global_step: int = 0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Train for one epoch

    Args:
        model: Model to train
        train_dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        device: Device to train on
        epoch: Current epoch
        global_step: Current global step
        scaler: Gradient scaler for mixed precision training

    Returns:
        Tuple[int, Dict[str, float]]: Updated global step and metrics
    """
    try:
        model.train()
        total_loss = 0
        metrics = {}

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss

            # Scale loss and backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Scheduler step
                if scheduler is not None:
                    scheduler.step()

                # Reset gradients
                optimizer.zero_grad()

                # Update global step
                global_step += 1

                # Log metrics
                if global_step % config.logging_steps == 0:
                    metrics = {
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]
                        if scheduler
                        else optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "step": global_step,
                    }
                    log_metrics(metrics, global_step)
                    log_gradients(model, global_step)
                    log_learning_rate(optimizer, global_step)
                    log_memory_usage(global_step)

            total_loss += loss.item()

        # Calculate average loss
        metrics["train_loss"] = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} average loss: {metrics['train_loss']:.4f}")

        return global_step, metrics

    except Exception as e:
        logger.error(f"Error in training epoch: {str(e)}")
        raise


def evaluate(
    model: PreTrainedModel,
    eval_dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Evaluate the model

    Args:
        model: Model to evaluate
        eval_dataloader: Evaluation data loader
        device: Device to evaluate on
        scaler: Gradient scaler for mixed precision training

    Returns:
        Dict[str, float]: Evaluation metrics
    """
    try:
        model.eval()
        total_loss = 0
        metrics = {}

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass with mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                        loss = outputs.loss
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                total_loss += loss.item()

        # Calculate average loss
        metrics["eval_loss"] = total_loss / len(eval_dataloader)
        logger.info(f"Evaluation loss: {metrics['eval_loss']:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise


def train(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    config: TrainingConfig = None,
    device: torch.device = None,
    optimizer_config: Optional[OptimizerConfig] = None,
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Train the model

    Args:
        model: Model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        config: Training configuration
        device: Device to train on
        optimizer_config: Optimizer configuration

    Returns:
        Tuple[PreTrainedModel, Dict[str, Any]]: Trained model and training history
    """
    try:
        # Create optimizer
        optimizer = create_optimizer(model, optimizer_config)

        # Create scheduler
        scheduler = create_scheduler(optimizer, config)

        # Create gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        training_history = []

        for epoch in range(config.num_train_epochs):
            # Train epoch
            global_step, train_metrics = train_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=device,
                epoch=epoch,
                global_step=global_step,
                scaler=scaler,
            )

            # Evaluate
            if eval_dataloader is not None and global_step % config.eval_steps == 0:
                eval_metrics = evaluate(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    scaler=scaler,
                )

                # Save best model
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    model.save_pretrained(f"{config.output_dir}/best_model")

            # Save checkpoint
            if global_step % config.save_steps == 0:
                model.save_pretrained(f"{config.output_dir}/checkpoint-{global_step}")

            # Update training history
            metrics = {**train_metrics}
            if eval_dataloader is not None:
                metrics.update(eval_metrics)
            training_history.append(metrics)

        # Save final model
        model.save_pretrained(f"{config.output_dir}/final_model")

        return model, training_history

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


def train_with_validation(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    optimizer_config: Optional[OptimizerConfig] = None,
    early_stopping_patience: int = 3,
) -> Tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Train the model with early stopping

    Args:
        model: Model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        config: Training configuration
        device: Device to train on
        optimizer_config: Optimizer configuration
        early_stopping_patience: Number of epochs to wait before early stopping

    Returns:
        Tuple[PreTrainedModel, Dict[str, Any]]: Trained model and training history
    """
    try:
        # Create optimizer
        optimizer = create_optimizer(model, optimizer_config)

        # Create scheduler
        scheduler = create_scheduler(optimizer, config)

        # Create gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        patience_counter = 0
        training_history = []

        for epoch in range(config.num_train_epochs):
            # Train epoch
            global_step, train_metrics = train_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=device,
                epoch=epoch,
                global_step=global_step,
                scaler=scaler,
            )

            # Evaluate
            eval_metrics = evaluate(
                model=model,
                eval_dataloader=eval_dataloader,
                device=device,
                scaler=scaler,
            )

            # Early stopping check
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                patience_counter = 0
                model.save_pretrained(f"{config.output_dir}/best_model")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Save checkpoint
            if global_step % config.save_steps == 0:
                model.save_pretrained(f"{config.output_dir}/checkpoint-{global_step}")

            # Update training history
            metrics = {**train_metrics, **eval_metrics}
            training_history.append(metrics)

        # Save final model
        model.save_pretrained(f"{config.output_dir}/final_model")

        return model, training_history

    except Exception as e:
        logger.error(f"Error in training with validation: {str(e)}")
        raise
