import logging
import torch
from typing import Optional, Dict, Any, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.config.training_config import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with LoRA configuration
    
    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
        device_map: Device mapping for model placement
        
    Returns:
        Tuple[Any, Any]: Model and tokenizer
    """
    try:
        logger.info(f"Loading model {model_config.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_id,
            token=model_config.token,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if model_config.use_bf16 else torch.float16,
            trust_remote_code=True,
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_id,
            token=model_config.token,
            trust_remote_code=True,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        # Configure LoRA
        logger.info("Configuring LoRA...")
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise


def setup_optimizer(
    model: Any,
    optimizer_config: Dict[str, Any],
) -> Any:
    """
    Setup optimizer for training
    
    Args:
        model: The model to optimize
        optimizer_config: Optimizer configuration
        
    Returns:
        Any: Configured optimizer
    """
    try:
        from transformers import AdamW, get_scheduler
        from torch.optim import AdamW as TorchAdamW
        
        optimizer_type = optimizer_config.get("type", "adamw")
        learning_rate = optimizer_config.get("learning_rate", 2e-5)
        weight_decay = optimizer_config.get("weight_decay", 0.01)
        
        if optimizer_type == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "torch_adamw":
            optimizer = TorchAdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        logger.info(f"Using optimizer: {optimizer_type}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        
        return optimizer
        
    except Exception as e:
        logger.error(f"Error setting up optimizer: {str(e)}")
        raise


def setup_lr_scheduler(
    optimizer: Any,
    scheduler_config: Dict[str, Any],
    num_training_steps: int,
) -> Any:
    """
    Setup learning rate scheduler
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_config: Scheduler configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Any: Configured learning rate scheduler
    """
    try:
        from transformers import get_scheduler
        
        scheduler_type = scheduler_config.get("type", "cosine")
        num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
        num_warmup_ratio = scheduler_config.get("num_warmup_ratio", 0.1)
        
        if num_warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * num_warmup_ratio)
            
        scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Using scheduler: {scheduler_type}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"Error setting up learning rate scheduler: {str(e)}")
        raise 