import logging
import os
import random
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logger.info(f"Set random seed to {seed} for reproducibility")


def setup_environment() -> str:
    """
    Setup training environment and configurations
    
    Returns:
        str: HuggingFace token for authentication
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow!")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Setup authentication and get HuggingFace token
    try:
        from src.utils.auth import setup_authentication
        setup_authentication()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set!")
    except Exception as e:
        logger.error(f"Authentication setup failed: {str(e)}")
        raise

    return hf_token


def is_bf16_supported() -> bool:
    """
    Check if BF16 is supported on the current device
    
    Returns:
        bool: True if BF16 is supported, False otherwise
    """
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except:
        return False


def setup_cuda_environment(
    cuda_visible_devices: Optional[str] = None,
    omp_num_threads: Optional[int] = None,
    cuda_device_max_connections: Optional[int] = None,
    cuda_launch_blocking: Optional[int] = None,
    pytorch_cuda_alloc_conf: Optional[str] = None,
) -> None:
    """
    Setup CUDA environment variables
    
    Args:
        cuda_visible_devices: CUDA visible devices
        omp_num_threads: Number of OpenMP threads
        cuda_device_max_connections: Maximum number of CUDA device connections
        cuda_launch_blocking: CUDA launch blocking
        pytorch_cuda_alloc_conf: PyTorch CUDA allocation configuration
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
    
    if cuda_device_max_connections is not None:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = str(cuda_device_max_connections)
    
    if cuda_launch_blocking is not None:
        os.environ["CUDA_LAUNCH_BLOCKING"] = str(cuda_launch_blocking)
    
    if pytorch_cuda_alloc_conf is not None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
    
    logger.info("CUDA environment variables set") 