from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ModelConfig:
    """Configuration for model-related parameters"""

    source_model: str = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
    destination_repo: str = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v2"
    max_seq_length: int = 2048
    private: bool = False
    save_method: str = "lora"
    push_strategy: str = "end"
    push_to_hub: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA-specific parameters"""

    r: int = 64
    alpha: int = 16
    dropout: float = 0.1
    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer parameters"""

    optimizer_type: str = "lion_8bit"
    weight_decay: float = 0.1
    beta1: float = 0.95
    beta2: float = 0.98
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    optim_bits: int = 8


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduler"""

    lr_scheduler_type: str = "cosine"
    num_cycles: int = 1
    power: float = 1.0
    last_epoch: int = -1


@dataclass
class TrainingConfig:
    """Configuration for general training parameters"""

    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_steps: Optional[int] = None
    logging_steps: int = 10
    save_steps: int = 30
    save_strategy: str = "steps"
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    random_seed: int = 42


@dataclass
class ValidationConfig:
    """Configuration for validation parameters"""

    val_split: float = 0.1
    validation_steps: int = 50
    minimal_validating: bool = True
    max_validation_samples: int = 60
    validate_at_start: bool = True
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.01


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters"""

    dataset_id: str = "tuandunghcmut/coding-mcq-reasoning"
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False


@dataclass
class PromptConfig:
    """Configuration for prompt-related parameters"""

    prompt_template: str = "teacher_reasoned"
    prompt_track_diversity: bool = True
    prompt_track_quality: bool = True
    prompt_categorize: bool = True
    prompt_comparison: bool = True
    max_prompts_to_save: int = 100


@dataclass
class ResponseOnlyConfig:
    """Configuration for response-only training"""

    enabled: bool = False
    instruction_token: str = "<|im_start|>user\n"
    response_token: str = "<|im_start|>assistant\n"
    instruction_token_id: Optional[int] = None
    response_token_id: Optional[int] = None


@dataclass
class AttentionConfig:
    """Configuration for attention implementation"""

    implementation: str = "flash_attention_2"
    force_implementation: bool = False


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging"""

    project: str = "Qwen2.5-Coder-1.5B-Instruct-Coding-Multiple-Choice"
    report_to: str = "wandb"
    tags: List[str] = field(default_factory=lambda: ["qwen", "multiple-choice", "coding", "lora"])


@dataclass
class EnvironmentConfig:
    """Configuration for environment settings"""

    full_determinism: bool = False
    torch_compile: bool = False
    use_cpu: bool = False
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True


@dataclass
class CompleteConfig:
    """Complete configuration for training"""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    response_only: ResponseOnlyConfig = field(default_factory=ResponseOnlyConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def from_args(cls, args):
        """Create a CompleteConfig from parsed arguments"""
        # This method will be implemented to convert argparse arguments to config objects
        # For now, it's a placeholder
        return cls()
