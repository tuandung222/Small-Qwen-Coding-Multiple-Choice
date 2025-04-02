import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

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


class BaseDistributedTrainer(ABC):
    """Base class for distributed training"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[DistributedTrainingConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or DistributedTrainingConfig()

        # Initialize distributed environment
        self._setup_distributed()

        # Setup accelerator if using Accelerate
        if self.config.distributed_backend == DistributedBackend.ACCELERATE:
            self.accelerator = Accelerator(
                mixed_precision=self.config.accelerate_mixed_precision,
                gradient_accumulation_steps=self.config.accelerate_gradient_accumulation_steps,
                dispatch_batches=self.config.accelerate_dispatch_batches,
                split_batches=self.config.accelerate_split_batches,
                even_batches=self.config.accelerate_even_batches,
                project_dir=self.config.accelerate_project_dir,
                project_config=self.config.accelerate_project_config,
            )
        else:
            self.accelerator = None

    def _setup_distributed(self):
        """Setup distributed training environment"""
        if self.config.distributed_backend == DistributedBackend.NONE:
            logger.info("No distributed training backend selected, using single GPU/CPU")
            return

        if not dist.is_initialized():
            if self.config.world_size == -1:
                self.config.world_size = torch.cuda.device_count()

            if self.config.rank == -1:
                self.config.rank = int(os.environ.get("RANK", -1))

            if self.config.local_rank == -1:
                self.config.local_rank = int(os.environ.get("LOCAL_RANK", -1))

            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)

            logger.info(
                f"Initialized distributed training: world_size={self.config.world_size}, "
                f"rank={self.config.rank}, local_rank={self.config.local_rank}"
            )

    @abstractmethod
    def prepare_model(self) -> PreTrainedModel:
        """Prepare model for distributed training"""
        pass

    @abstractmethod
    def prepare_dataloader(self, dataset: Dataset, is_train: bool = True) -> DataLoader:
        """Prepare dataloader for distributed training"""
        pass

    def train(self, **kwargs):
        """Train the model"""
        # Prepare model and dataloaders
        self.model = self.prepare_model()
        train_dataloader = self.prepare_dataloader(self.train_dataset, is_train=True)
        eval_dataloader = (
            self.prepare_dataloader(self.eval_dataset, is_train=False)
            if self.eval_dataset
            else None
        )

        # Prepare optimizer and scheduler
        optimizer = self._prepare_optimizer()
        scheduler = self._prepare_scheduler(optimizer, len(train_dataloader))

        # Prepare everything with accelerator if using Accelerate
        if self.accelerator is not None:
            self.model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
                self.model, optimizer, train_dataloader, scheduler
            )
            if eval_dataloader:
                eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Training loop
        self._train_loop(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

    def _prepare_optimizer(self):
        """Prepare optimizer"""
        # Implementation depends on your specific needs
        pass

    def _prepare_scheduler(self, optimizer, num_training_steps):
        """Prepare learning rate scheduler"""
        # Implementation depends on your specific needs
        pass

    def _train_loop(self, **kwargs):
        """Training loop implementation"""
        # Implementation depends on your specific needs
        pass


class DDPTrainer(BaseDistributedTrainer):
    """Distributed Data Parallel trainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.config.distributed_backend != DistributedBackend.TORCH_RUN
            and not self.config.use_ddp
        ):
            raise ValueError("DDPTrainer requires use_ddp=True or distributed_backend=TORCH_RUN")

    def prepare_model(self) -> PreTrainedModel:
        """Prepare model for DDP training"""
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Wrap model with DDP
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )
        return self.model

    def prepare_dataloader(self, dataset: Dataset, is_train: bool = True) -> DataLoader:
        """Prepare dataloader for DDP training"""
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=is_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )


class FSDPTrainer(BaseDistributedTrainer):
    """Fully Sharded Data Parallel trainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.use_fsdp:
            raise ValueError("FSDPTrainer requires use_fsdp=True in config")

        # Import FSDP modules
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, StateDictType
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        self.FSDP = FSDP
        self.StateDictType = StateDictType
        self.MixedPrecision = MixedPrecision
        self.CPUOffload = CPUOffload

    def prepare_model(self) -> PreTrainedModel:
        """Prepare model for FSDP training"""
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Setup FSDP wrapping policy
        if self.config.fsdp_auto_wrap_policy == "transformer":
            auto_wrap_policy = transformer_auto_wrap_policy
        elif self.config.fsdp_auto_wrap_policy == "size":
            auto_wrap_policy = size_based_auto_wrap_policy(self.config.fsdp_min_num_params)
        else:
            auto_wrap_policy = None

        # Setup mixed precision
        mixed_precision_config = None
        if self.config.mixed_precision != "no":
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.bfloat16
                if self.config.mixed_precision == "bf16"
                else torch.float16,
                reduce_dtype=torch.bfloat16
                if self.config.mixed_precision == "bf16"
                else torch.float16,
                buffer_dtype=torch.bfloat16
                if self.config.mixed_precision == "bf16"
                else torch.float16,
            )

        # Setup CPU offload if enabled
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_offload_params else None

        # Wrap model with FSDP
        self.model = self.FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_config,
            cpu_offload=cpu_offload,
            device_id=self.config.local_rank,
        )

        return self.model

    def prepare_dataloader(self, dataset: Dataset, is_train: bool = True) -> DataLoader:
        """Prepare dataloader for FSDP training"""
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=is_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )


class DeepSpeedTrainer(BaseDistributedTrainer):
    """DeepSpeed trainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.distributed_backend != DistributedBackend.DEEPSPEED:
            raise ValueError("DeepSpeedTrainer requires distributed_backend=DEEPSPEED")

        # Import DeepSpeed
        try:
            import deepspeed

            self.deepspeed = deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed is not installed. Please install it with 'pip install deepspeed'"
            )

        # Create DeepSpeed config if not provided
        if not self.config.deepspeed_config_path:
            self._create_deepspeed_config()

    def _create_deepspeed_config(self):
        """Create DeepSpeed configuration file"""
        import json
        import tempfile

        deepspeed_config = {
            "train_batch_size": self.config.batch_size * self.config.world_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-4, "warmup_num_steps": 100},
            },
            "fp16": {"enabled": self.config.mixed_precision == "fp16"},
            "bf16": {"enabled": self.config.mixed_precision == "bf16"},
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True}
                if self.config.deepspeed_offload_optimizer
                else None,
                "offload_param": {"device": "cpu", "pin_memory": True}
                if self.config.deepspeed_offload_param
                else None,
                "reduce_bucket_size": self.config.deepspeed_zero_reduce_bucket_size,
                "reduce_scatter": self.config.deepspeed_zero_reduce_scatter,
                "contiguous_gradients": self.config.deepspeed_zero_contiguous_gradients,
            },
            "gradient_checkpointing": self.config.gradient_checkpointing,
        }

        # Create temporary file for DeepSpeed config
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(deepspeed_config, f, indent=2)

        self.config.deepspeed_config_path = path
        logger.info(f"Created DeepSpeed config at {path}")

    def prepare_model(self) -> PreTrainedModel:
        """Prepare model for DeepSpeed training"""
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Initialize DeepSpeed engine
        self.model_engine, optimizer, _, scheduler = self.deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=self.config.deepspeed_config_path,
        )

        return self.model_engine

    def prepare_dataloader(self, dataset: Dataset, is_train: bool = True) -> DataLoader:
        """Prepare dataloader for DeepSpeed training"""
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=is_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )


class AccelerateTrainer(BaseDistributedTrainer):
    """Accelerate trainer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.distributed_backend != DistributedBackend.ACCELERATE:
            raise ValueError("AccelerateTrainer requires distributed_backend=ACCELERATE")

    def prepare_model(self) -> PreTrainedModel:
        """Prepare model for Accelerate training"""
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Model will be prepared by accelerator.prepare() in the train method
        return self.model

    def prepare_dataloader(self, dataset: Dataset, is_train: bool = True) -> DataLoader:
        """Prepare dataloader for Accelerate training"""
        # For Accelerate, we use a regular DataLoader with DistributedSampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=is_train,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )


def create_distributed_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DistributedTrainingConfig] = None,
) -> BaseDistributedTrainer:
    """
    Factory function to create the appropriate distributed trainer based on configuration

    Args:
        model: The model to train
        tokenizer: The tokenizer for the model
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset (optional)
        config: The distributed training configuration

    Returns:
        An instance of the appropriate distributed trainer
    """
    if config is None:
        config = DistributedTrainingConfig()

    # Determine which trainer to use based on configuration
    if config.distributed_backend == DistributedBackend.DEEPSPEED:
        return DeepSpeedTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
    elif config.distributed_backend == DistributedBackend.ACCELERATE:
        return AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
    elif config.use_fsdp:
        return FSDPTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
    elif config.use_ddp or config.distributed_backend == DistributedBackend.TORCH_RUN:
        return DDPTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )
    else:
        # No distributed training
        return QwenTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
