import os
import time
import wandb
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from transformers import TrainerCallback
from .auth import setup_authentication

class WandBConfig:
    """Configuration for W&B logging"""
    
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_memory: bool = True,
        log_gradients: bool = True,
    ):
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity
        self.tags = tags
        self.notes = notes
        self.config = config
        self.log_memory = log_memory
        self.log_gradients = log_gradients

class WandBLogger:
    """Logger for W&B integration"""
    
    def __init__(self, config: WandBConfig):
        # Setup authentication
        setup_authentication()
        
        self.config = config
        self.run = None
        self.train_start_time = time.time()
        self.step_start_time = time.time()
        
    def setup(self):
        """Setup W&B run"""
        self.run = wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            entity=self.config.entity,
            tags=self.config.tags,
            notes=self.config.notes,
            config=self.config.config,
        )
        
    def init_run(self, model_name: str):
        """Initialize a new W&B run"""
        # Set run name if not provided
        if not self.config.run_name:
            self.config.run_name = f"{model_name}_{wandb.util.generate_id()}"
            
        # Initialize run
        self.setup()
        
    def log_model_info(self, model):
        """Log model information to W&B"""
        if not self.run:
            raise RuntimeError("W&B run not initialized. Call init_run first.")
            
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.run.summary.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        })
        
    def finish_run(self):
        """Finish the W&B run"""
        if self.run:
            self.run.finish()
            self.run = None

class WandBCallback(TrainerCallback):
    """Callback for logging training metrics to W&B"""
    
    def __init__(self, logger: WandBLogger):
        self.logger = logger
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Log model info at start of training"""
        if model:
            self.logger.log_model_info(model)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training"""
        if logs:
            wandb.log(logs)
            
    def on_train_end(self, args, state, control, **kwargs):
        """Finish logging when training ends"""
        self.logger.finish_run()

    def log_training_metrics(self, logs: Dict[str, Any], state: Any, model: Optional[torch.nn.Module] = None):
        """Log training metrics including gradients and memory"""
        if not self.config.log_training or not self.run:
            return

        # Log basic training metrics
        if "loss" in logs:
            logs["training/loss"] = logs["loss"]
        
        # Log learning rate
        if hasattr(state, "learning_rate"):
            logs["training/learning_rate"] = state.learning_rate

        # Log gradient statistics if enabled
        if self.config.log_gradients and model is not None:
            grad_norm = 0.0
            param_norm = 0.0
            total_params = 0
            
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_norm += param.data.norm(2).item() ** 2
                    total_params += 1
            
            if total_params > 0:
                grad_norm = grad_norm ** 0.5
                param_norm = param_norm ** 0.5
                logs["training/gradient_norm"] = grad_norm
                logs["training/parameter_norm"] = param_norm
                logs["training/grad_param_ratio"] = grad_norm / param_norm if param_norm > 0 else 0

        # Log memory usage if enabled
        if self.config.log_memory and torch.cuda.is_available():
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
        """Log validation metrics"""
        if not self.config.log_validation or not self.run:
            return

        log_data = {
            "validation/accuracy": metrics.get("accuracy", 0.0),
            "validation/combined_score": metrics.get("combined_score", 0.0),
            "epoch": epoch + 1,
            "step": step,
        }

        if "reasoning_quality" in metrics:
            log_data["validation/reasoning_quality"] = metrics["reasoning_quality"]
        if "quality_accuracy" in metrics:
            log_data["validation/quality_accuracy"] = metrics["quality_accuracy"]

        wandb.log(log_data)

    def log_examples(self, examples: List[Dict[str, Any]], step: int):
        """Log example predictions"""
        if not self.config.log_examples or not self.run:
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

        wandb.log({f"examples/val_{step}": example_table}) 