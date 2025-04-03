import logging
import torch
from typing import Dict, Any, Optional, Union
from src.config.training_config import EnvironmentConfig

logger = logging.getLogger(__name__)


def setup_device(
    environment_config: EnvironmentConfig,
) -> torch.device:
    """
    Setup device for training
    
    Args:
        environment_config: Environment configuration
        
    Returns:
        torch.device: Device to use for training
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
            
        return device
        
    except Exception as e:
        logger.error(f"Error setting up device: {str(e)}")
        raise


def setup_multi_gpu(
    model: torch.nn.Module,
    environment_config: EnvironmentConfig,
) -> torch.nn.Module:
    """
    Setup model for multi-GPU training
    
    Args:
        model: Model to configure
        environment_config: Environment configuration
        
    Returns:
        torch.nn.Module: Configured model
    """
    try:
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
            
        return model
        
    except Exception as e:
        logger.error(f"Error setting up multi-GPU: {str(e)}")
        raise


def setup_distributed_training(
    model: torch.nn.Module,
    environment_config: EnvironmentConfig,
) -> tuple:
    """
    Setup distributed training
    
    Args:
        model: Model to configure
        environment_config: Environment configuration
        
    Returns:
        tuple: Configured model and process group
    """
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        
        logger.info(f"Initialized distributed training on rank {local_rank}")
        
        return model, dist.get_world_size()
        
    except Exception as e:
        logger.error(f"Error setting up distributed training: {str(e)}")
        raise


def get_device_memory_stats() -> Dict[str, float]:
    """
    Get memory statistics for the current device
    
    Returns:
        Dict[str, float]: Memory statistics in GB
    """
    try:
        if torch.cuda.is_available():
            stats = {
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "cached": torch.cuda.memory_reserved(0) / 1024**3,
            }
            
            logger.info("Device memory stats:")
            logger.info(f"  Total: {stats['total']:.2f} GB")
            logger.info(f"  Allocated: {stats['allocated']:.2f} GB")
            logger.info(f"  Cached: {stats['cached']:.2f} GB")
            
            return stats
            
        return {}
        
    except Exception as e:
        logger.error(f"Error getting device memory stats: {str(e)}")
        raise


def clear_device_memory() -> None:
    """
    Clear device memory
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared device memory")
            
    except Exception as e:
        logger.error(f"Error clearing device memory: {str(e)}")
        raise


def move_to_device(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Move data to device
    
    Args:
        data: Data to move
        device: Device to move to
        
    Returns:
        Union[torch.Tensor, Dict[str, torch.Tensor]]: Data on device
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: v.to(device) for k, v in data.items()}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    except Exception as e:
        logger.error(f"Error moving data to device: {str(e)}")
        raise 