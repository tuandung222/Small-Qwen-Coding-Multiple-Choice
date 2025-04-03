import logging
import torch
from typing import Dict, Any, Optional, List
from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)


def setup_gradient_checkpointing(
    model: torch.nn.Module,
    training_config: TrainingConfig,
) -> None:
    """
    Setup gradient checkpointing for the model
    
    Args:
        model: Model to configure
        training_config: Training configuration
    """
    try:
        if training_config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        else:
            model.gradient_checkpointing_disable()
            logger.info("Disabled gradient checkpointing")
            
    except Exception as e:
        logger.error(f"Error setting up gradient checkpointing: {str(e)}")
        raise


def clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float = 1.0,
) -> None:
    """
    Clip gradients to prevent exploding gradients
    
    Args:
        model: Model to clip gradients for
        max_grad_norm: Maximum gradient norm
    """
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
    except Exception as e:
        logger.error(f"Error clipping gradients: {str(e)}")
        raise


def accumulate_gradients(
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    step: int,
) -> bool:
    """
    Accumulate gradients for gradient accumulation
    
    Args:
        loss: Loss value
        model: Model to accumulate gradients for
        optimizer: Optimizer to use
        gradient_accumulation_steps: Number of steps to accumulate over
        step: Current step
        
    Returns:
        bool: Whether to update weights
    """
    try:
        # Scale loss by number of accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Check if we should update weights
        should_update = (step + 1) % gradient_accumulation_steps == 0
        
        if should_update:
            optimizer.step()
            optimizer.zero_grad()
            
        return should_update
        
    except Exception as e:
        logger.error(f"Error accumulating gradients: {str(e)}")
        raise


def get_gradient_stats(
    model: torch.nn.Module,
) -> Dict[str, float]:
    """
    Get statistics about gradients
    
    Args:
        model: Model to get gradient stats for
        
    Returns:
        Dict[str, float]: Gradient statistics
    """
    try:
        total_norm = 0
        param_norms = []
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_norms.append(param_norm.item())
                
        total_norm = total_norm ** 0.5
        
        stats = {
            "total_norm": total_norm,
            "mean_norm": sum(param_norms) / len(param_norms) if param_norms else 0,
            "max_norm": max(param_norms) if param_norms else 0,
            "min_norm": min(param_norms) if param_norms else 0,
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting gradient stats: {str(e)}")
        raise


def check_gradient_overflow(
    model: torch.nn.Module,
    threshold: float = 1e6,
) -> bool:
    """
    Check if gradients have overflowed
    
    Args:
        model: Model to check gradients for
        threshold: Threshold for overflow
        
    Returns:
        bool: Whether gradients have overflowed
    """
    try:
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return True
                if p.grad.norm().item() > threshold:
                    return True
                    
        return False
        
    except Exception as e:
        logger.error(f"Error checking gradient overflow: {str(e)}")
        raise 