import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for model initialization
    """

    model_name_or_path: str
    model_revision: str = "main"
    use_auth_token: bool = False
    use_bf16: bool = False
    use_fp16: bool = False
    use_8bit: bool = False
    use_4bit: bool = False
    use_flash_attention_2: bool = False
    use_xformers: bool = False
    use_gradient_checkpointing: bool = False
    device_map: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    low_cpu_mem_usage: bool = False
    trust_remote_code: bool = True


def initialize_model(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Initialize a model and tokenizer

    Args:
        config: Model configuration

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Initialized model and tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Set up device and dtype
        if config.device_map is None:
            config.device_map = "auto"

        if config.torch_dtype is None:
            if config.use_bf16:
                config.torch_dtype = torch.bfloat16
            elif config.use_fp16:
                config.torch_dtype = torch.float16
            else:
                config.torch_dtype = torch.float32

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            use_auth_token=config.use_auth_token,
            trust_remote_code=config.trust_remote_code,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            revision=config.model_revision,
            use_auth_token=config.use_auth_token,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            trust_remote_code=config.trust_remote_code,
        )

        # Apply optimizations
        if config.use_8bit:
            model = model.quantize(8)
        elif config.use_4bit:
            model = model.quantize(4)

        if config.use_flash_attention_2:
            model = model.to_bettertransformer()

        if config.use_xformers:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            model.config.use_memory_efficient_attention = True
            model.config.attention_implementation = "xformers"
            model.config.attention_op = MemoryEfficientAttentionFlashAttentionOp

        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        logger.info(
            f"Initialized model {config.model_name_or_path} with dtype {config.torch_dtype}"
        )
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise


def initialize_lora(
    model: PreTrainedModel,
    config: Dict[str, Any],
) -> PreTrainedModel:
    """
    Initialize LoRA for a model

    Args:
        model: Base model
        config: LoRA configuration

    Returns:
        PreTrainedModel: Model with LoRA applied
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        # Create LoRA config
        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["lora_target_modules"],
            lora_dropout=config["lora_dropout"],
            bias=config["lora_bias"],
            task_type=TaskType[config["lora_task_type"]],
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        logger.info(f"Initialized LoRA with rank {config['lora_r']}")
        return model

    except Exception as e:
        logger.error(f"Error initializing LoRA: {str(e)}")
        raise


def initialize_optimized_model(
    model: PreTrainedModel,
    config: ModelConfig,
) -> PreTrainedModel:
    """
    Initialize an optimized version of the model

    Args:
        model: Base model
        config: Model configuration

    Returns:
        PreTrainedModel: Optimized model
    """
    try:
        # Apply quantization
        if config.use_8bit:
            from bitsandbytes.nn import Linear8bitLt

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
        elif config.use_4bit:
            from bitsandbytes.nn import Linear4bit

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module = Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

        # Apply attention optimizations
        if config.use_flash_attention_2:
            model = model.to_bettertransformer()

        if config.use_xformers:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            model.config.use_memory_efficient_attention = True
            model.config.attention_implementation = "xformers"
            model.config.attention_op = MemoryEfficientAttentionFlashAttentionOp

        # Apply gradient checkpointing
        if config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        logger.info("Initialized optimized model")
        return model

    except Exception as e:
        logger.error(f"Error initializing optimized model: {str(e)}")
        raise


def initialize_model_parallel(
    model: PreTrainedModel,
    num_gpus: int,
) -> PreTrainedModel:
    """
    Initialize model parallel training

    Args:
        model: Base model
        num_gpus: Number of GPUs to use

    Returns:
        PreTrainedModel: Model parallel model
    """
    try:
        if num_gpus > 1:
            model = nn.DataParallel(model)
            logger.info(f"Initialized model parallel training with {num_gpus} GPUs")
        return model

    except Exception as e:
        logger.error(f"Error initializing model parallel: {str(e)}")
        raise


def initialize_distributed_model(
    model: PreTrainedModel,
    local_rank: int,
) -> PreTrainedModel:
    """
    Initialize distributed training

    Args:
        model: Base model
        local_rank: Local rank of the process

    Returns:
        PreTrainedModel: Distributed model
    """
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        # Initialize process group
        dist.init_process_group(backend="nccl")

        # Move model to GPU
        model = model.to(local_rank)

        # Wrap model in DDP
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

        logger.info(f"Initialized distributed training on rank {local_rank}")
        return model

    except Exception as e:
        logger.error(f"Error initializing distributed model: {str(e)}")
        raise
