import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import Dataset
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler

from src.training.trainer import QwenTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DistributedBackend(Enum):
    """Enum for distributed training backends"""

    TORCH_RUN = "torch_run"
    DEEPSPEED = "deepspeed"
    ACCELERATE = "accelerate"
    NONE = "none"


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training"""

    # Common settings
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    world_size: int = -1  # -1 for auto-detection
    rank: int = -1  # -1 for auto-detection
    local_rank: int = -1  # -1 for auto-detection
    master_addr: str = "localhost"
    master_port: str = "12355"
    init_method: str = "env://"
    mixed_precision: str = "bf16"  # bf16, fp16, or no
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    batch_size: int = 8  # Default batch size per device

    # Distributed backend selection
    distributed_backend: DistributedBackend = DistributedBackend.NONE

    # DDP specific settings
    use_ddp: bool = False

    # FSDP specific settings
    use_fsdp: bool = False
    fsdp_min_num_params: int = 1e6  # Minimum number of parameters to use FSDP
    fsdp_state_dict_type: str = "FULL_STATE_DICT"
    fsdp_offload_params: bool = False
    fsdp_auto_wrap_policy: Optional[str] = None  # transformer, size, or None

    # DeepSpeed specific settings
    deepspeed_config_path: Optional[str] = None
    deepspeed_stage: int = 2  # 0, 1, 2, 3
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_param: bool = False
    deepspeed_zero_reduce_bucket_size: int = 5e8
    deepspeed_zero_reduce_scatter: bool = True
    deepspeed_zero_contiguous_gradients: bool = True

    # Accelerate specific settings
    accelerate_mixed_precision: str = "bf16"  # bf16, fp16, or no
    accelerate_gradient_accumulation_steps: int = 1
    accelerate_gradient_checkpointing: bool = True
    accelerate_dispatch_batches: bool = True
    accelerate_split_batches: bool = False
    accelerate_even_batches: bool = True
    accelerate_project_dir: Optional[str] = None
    accelerate_project_config: Optional[Dict[str, Any]] = None


class DistributedTrainer(QwenTrainer):
    """
    Distributed training handler for Qwen models that inherits from QwenTrainer.

    This class extends QwenTrainer with distributed training capabilities:
    1. Support for multiple distributed backends (DDP, FSDP, DeepSpeed, Accelerate)
    2. Automatic distributed environment setup
    3. Distributed data loading and model preparation
    4. Integration with existing QwenTrainer functionality
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_creator: Any,
        lora_config: Optional[Any] = None,
        destination_hub_config: Optional[Any] = None,
        distributed_config: Optional[DistributedTrainingConfig] = None,
        debug_samples: int = 3,
    ):
        """
        Initialize the DistributedTrainer with model, tokenizer, and configuration.

        Args:
            model: The base model to fine-tune
            tokenizer: The tokenizer associated with the model
            prompt_creator: PromptCreator instance for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient training
            destination_hub_config: Optional configuration for pushing to HuggingFace Hub
            distributed_config: Configuration for distributed training
            debug_samples: Number of random samples to log during training for debugging
        """
        # Initialize the parent QwenTrainer
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt_creator=prompt_creator,
            lora_config=lora_config,
            destination_hub_config=destination_hub_config,
            debug_samples=debug_samples,
        )

        # Store distributed configuration
        self.distributed_config = distributed_config or DistributedTrainingConfig()

        # Initialize distributed environment
        self._setup_distributed()

        # Setup accelerator if using Accelerate
        if self.distributed_config.distributed_backend == DistributedBackend.ACCELERATE:
            self.accelerator = Accelerator(
                mixed_precision=self.distributed_config.accelerate_mixed_precision,
                gradient_accumulation_steps=self.distributed_config.accelerate_gradient_accumulation_steps,
                dispatch_batches=self.distributed_config.accelerate_dispatch_batches,
                split_batches=self.distributed_config.accelerate_split_batches,
                even_batches=self.distributed_config.accelerate_even_batches,
                project_dir=self.distributed_config.accelerate_project_dir,
                project_config=self.distributed_config.accelerate_project_config,
            )
        else:
            self.accelerator = None

    def _setup_distributed(self):
        """Setup distributed training environment"""
        if self.distributed_config.distributed_backend == DistributedBackend.NONE:
            logger.info("No distributed training backend selected, using single GPU/CPU")
            return

        if not dist.is_initialized():
            if self.distributed_config.world_size == -1:
                self.distributed_config.world_size = torch.cuda.device_count()

            if self.distributed_config.rank == -1:
                self.distributed_config.rank = int(os.environ.get("RANK", -1))

            if self.distributed_config.local_rank == -1:
                self.distributed_config.local_rank = int(os.environ.get("LOCAL_RANK", -1))

            dist.init_process_group(
                backend=self.distributed_config.backend,
                init_method=self.distributed_config.init_method,
                world_size=self.distributed_config.world_size,
                rank=self.distributed_config.rank,
            )

            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.distributed_config.local_rank)

            logger.info(
                f"Initialized distributed training: world_size={self.distributed_config.world_size}, "
                f"rank={self.distributed_config.rank}, local_rank={self.distributed_config.local_rank}"
            )

    def prepare_model_for_training(self) -> Any:
        """
        Prepare model for distributed training.

        This method extends the parent class's prepare_model_for_training method
        to add distributed training capabilities.
        """
        # First, prepare the model using the parent class method
        model = super().prepare_model_for_training()

        # Apply distributed training specific preparations
        if (
            self.distributed_config.distributed_backend == DistributedBackend.TORCH_RUN
            or self.distributed_config.use_ddp
        ):
            # DDP preparation
            if self.distributed_config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Wrap model with DDP
            model = DistributedDataParallel(
                model,
                device_ids=[self.distributed_config.local_rank]
                if torch.cuda.is_available()
                else None,
                output_device=self.distributed_config.local_rank
                if torch.cuda.is_available()
                else None,
                find_unused_parameters=True,
            )
        elif self.distributed_config.use_fsdp:
            # FSDP preparation
            from torch.distributed.fsdp import CPUOffload
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, StateDictType
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            if self.distributed_config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Setup FSDP wrapping policy
            if self.distributed_config.fsdp_auto_wrap_policy == "transformer":
                auto_wrap_policy = transformer_auto_wrap_policy
            elif self.distributed_config.fsdp_auto_wrap_policy == "size":
                auto_wrap_policy = size_based_auto_wrap_policy(
                    self.distributed_config.fsdp_min_num_params
                )
            else:
                auto_wrap_policy = None

            # Setup mixed precision
            mixed_precision_config = None
            if self.distributed_config.mixed_precision != "no":
                mixed_precision_config = MixedPrecision(
                    param_dtype=torch.bfloat16
                    if self.distributed_config.mixed_precision == "bf16"
                    else torch.float16,
                    reduce_dtype=torch.bfloat16
                    if self.distributed_config.mixed_precision == "bf16"
                    else torch.float16,
                    buffer_dtype=torch.bfloat16
                    if self.distributed_config.mixed_precision == "bf16"
                    else torch.float16,
                )

            # Setup CPU offload if enabled
            cpu_offload = (
                CPUOffload(offload_params=True)
                if self.distributed_config.fsdp_offload_params
                else None
            )

            # Wrap model with FSDP
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_config,
                cpu_offload=cpu_offload,
                device_id=self.distributed_config.local_rank,
            )
        elif self.distributed_config.distributed_backend == DistributedBackend.DEEPSPEED:
            # DeepSpeed preparation
            try:
                import deepspeed

                self.deepspeed = deepspeed
            except ImportError:
                raise ImportError(
                    "DeepSpeed is not installed. Please install it with 'pip install deepspeed'"
                )

            if self.distributed_config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Create DeepSpeed config if not provided
            if not self.distributed_config.deepspeed_config_path:
                self._create_deepspeed_config()

            # Initialize DeepSpeed engine
            model, optimizer, _, scheduler = self.deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=self.distributed_config.deepspeed_config_path,
            )

            return model

        return model

    def _create_deepspeed_config(self):
        """Create DeepSpeed configuration file"""
        import json
        import tempfile

        deepspeed_config = {
            "train_batch_size": self.distributed_config.batch_size
            * self.distributed_config.world_size,
            "gradient_accumulation_steps": self.distributed_config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-4, "warmup_num_steps": 100},
            },
            "fp16": {"enabled": self.distributed_config.mixed_precision == "fp16"},
            "bf16": {"enabled": self.distributed_config.mixed_precision == "bf16"},
            "zero_optimization": {
                "stage": self.distributed_config.deepspeed_stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True}
                if self.distributed_config.deepspeed_offload_optimizer
                else None,
                "offload_param": {"device": "cpu", "pin_memory": True}
                if self.distributed_config.deepspeed_offload_param
                else None,
                "reduce_bucket_size": self.distributed_config.deepspeed_zero_reduce_bucket_size,
                "reduce_scatter": self.distributed_config.deepspeed_zero_reduce_scatter,
                "contiguous_gradients": self.distributed_config.deepspeed_zero_contiguous_gradients,
            },
            "gradient_checkpointing": self.distributed_config.gradient_checkpointing,
        }

        # Create temporary file for DeepSpeed config
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(deepspeed_config, f, indent=2)

        self.distributed_config.deepspeed_config_path = path
        logger.info(f"Created DeepSpeed config at {path}")

    def _prepare_dataset_for_training(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset for distributed training.

        This method extends the parent class's _prepare_dataset_for_training method
        to add distributed training capabilities.
        """
        # First, prepare the dataset using the parent class method
        formatted_dataset = super()._prepare_dataset_for_training(dataset)

        # Apply distributed training specific preparations
        if self.distributed_config.distributed_backend != DistributedBackend.NONE:
            # Create a distributed sampler
            sampler = torch.utils.data.distributed.DistributedSampler(
                formatted_dataset,
                num_replicas=self.distributed_config.world_size,
                rank=self.distributed_config.rank,
                shuffle=True,
            )

            # Create a DataLoader with the distributed sampler
            dataloader = DataLoader(
                formatted_dataset,
                batch_size=self.distributed_config.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

            # Return the dataset with the distributed sampler attached
            formatted_dataset.distributed_sampler = sampler
            formatted_dataset.distributed_dataloader = dataloader

        return formatted_dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        val_split: float = 0.1,
        output_dir: str = "./model_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.1,
        max_steps: Optional[int] = None,
        logging_steps: int = 10,
        save_steps: int = 100,
        save_strategy: str = "epoch",
        save_total_limit: int = 1,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        callbacks: Optional[List[Any]] = None,
        random_seed: int = 42,
        push_to_hub_strategy: str = "end",
        wandb_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        responses_only_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with distributed training capabilities.

        This method extends the parent class's train method to add distributed training capabilities.
        """
        # Update distributed config with training parameters
        self.distributed_config.batch_size = per_device_train_batch_size
        self.distributed_config.gradient_accumulation_steps = gradient_accumulation_steps

        # Call the parent class's train method
        return super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_split=val_split,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            callbacks=callbacks,
            random_seed=random_seed,
            push_to_hub_strategy=push_to_hub_strategy,
            wandb_config=wandb_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            responses_only_config=responses_only_config,
        )


def create_distributed_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_creator: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    distributed_config: Optional[DistributedTrainingConfig] = None,
    lora_config: Optional[Any] = None,
    destination_hub_config: Optional[Any] = None,
    debug_samples: int = 3,
) -> DistributedTrainer:
    """
    Factory function to create a distributed trainer based on configuration

    Args:
        model: The model to train
        tokenizer: The tokenizer for the model
        prompt_creator: PromptCreator instance for formatting prompts
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset (optional)
        distributed_config: The distributed training configuration
        lora_config: Optional LoRA configuration for parameter-efficient training
        destination_hub_config: Optional configuration for pushing to HuggingFace Hub
        debug_samples: Number of random samples to log during training for debugging

    Returns:
        An instance of the DistributedTrainer
    """
    if distributed_config is None:
        distributed_config = DistributedTrainingConfig()

    return DistributedTrainer(
        model=model,
        tokenizer=tokenizer,
        prompt_creator=prompt_creator,
        lora_config=lora_config,
        destination_hub_config=destination_hub_config,
        distributed_config=distributed_config,
        debug_samples=debug_samples,
    )
