import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer
    """
    name: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = None


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """
    Create an optimizer for the model
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer: Created optimizer
    """
    try:
        # Get parameters to optimize
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        if config.name.lower() == "adamw":
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            )
        elif config.name.lower() == "adam":
            from torch.optim import Adam
            optimizer = Adam(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
            )
        elif config.name.lower() == "sgd":
            from torch.optim import SGD
            optimizer = SGD(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                momentum=config.beta1,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.name}")
            
        logger.info(f"Created {config.name} optimizer with learning rate {config.learning_rate}")
        return optimizer
        
    except Exception as e:
        logger.error(f"Error creating optimizer: {str(e)}")
        raise


def create_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig,
    num_training_steps: int,
) -> _LRScheduler:
    """
    Create a learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        config: Optimizer configuration
        num_training_steps: Total number of training steps
        
    Returns:
        _LRScheduler: Created scheduler
    """
    try:
        # Calculate warmup steps
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        # Create scheduler
        if config.lr_scheduler_type.lower() == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_scheduler_type.lower() == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_scheduler_type.lower() == "cosine_with_restarts":
            from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif config.lr_scheduler_type.lower() == "polynomial":
            from transformers import get_polynomial_decay_schedule_with_warmup
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **config.lr_scheduler_kwargs or {},
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.lr_scheduler_type}")
            
        logger.info(f"Created {config.lr_scheduler_type} scheduler with {num_warmup_steps} warmup steps")
        return scheduler
        
    except Exception as e:
        logger.error(f"Error creating scheduler: {str(e)}")
        raise


def clip_gradients(
    model: nn.Module,
    max_grad_norm: float,
) -> None:
    """
    Clip gradients to prevent exploding gradients
    
    Args:
        model: Model to clip gradients for
        max_grad_norm: Maximum gradient norm
    """
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        logger.debug(f"Clipped gradients to max norm {max_grad_norm}")
        
    except Exception as e:
        logger.error(f"Error clipping gradients: {str(e)}")
        raise


def get_lr(optimizer: Optimizer) -> float:
    """
    Get the current learning rate
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        float: Current learning rate
    """
    try:
        for param_group in optimizer.param_groups:
            return param_group["lr"]
            
    except Exception as e:
        logger.error(f"Error getting learning rate: {str(e)}")
        raise


def get_gradient_norm(model: nn.Module) -> float:
    """
    Get the gradient norm
    
    Args:
        model: Model to get gradient norm for
        
    Returns:
        float: Gradient norm
    """
    try:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    except Exception as e:
        logger.error(f"Error getting gradient norm: {str(e)}")
        raise


def get_parameter_norm(model: nn.Module) -> float:
    """
    Get the parameter norm
    
    Args:
        model: Model to get parameter norm for
        
    Returns:
        float: Parameter norm
    """
    try:
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    except Exception as e:
        logger.error(f"Error getting parameter norm: {str(e)}")
        raise


def get_optimizer_stats(optimizer: Optimizer) -> Dict[str, float]:
    """
    Get optimizer statistics
    
    Args:
        optimizer: Optimizer to get statistics for
        
    Returns:
        Dict[str, float]: Optimizer statistics
    """
    try:
        stats = {
            "learning_rate": get_lr(optimizer),
            "momentum": optimizer.param_groups[0].get("momentum", 0.0),
            "beta1": optimizer.param_groups[0].get("betas", (0.9, 0.999))[0],
            "beta2": optimizer.param_groups[0].get("betas", (0.9, 0.999))[1],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
        }
        return stats
        
    except Exception as e:
        logger.error(f"Error getting optimizer stats: {str(e)}")
        raise 