import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for model configuration
    """

    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-7B-Chat",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`"},
    )
    use_bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 16-bit (mixed) precision training"},
    )
    use_fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training"},
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"},
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"},
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 2"},
    )
    use_xformers: bool = field(
        default=False,
        metadata={"help": "Whether to use xFormers"},
    )
    use_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing"},
    )


@dataclass
class LoRAArguments:
    """
    Arguments for LoRA configuration
    """

    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha scaling"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "LoRA target modules"},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type (none, all, or lora_only)"},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "LoRA task type"},
    )


@dataclass
class TrainingArguments:
    """
    Arguments for training configuration
    """

    output_dir: str = field(
        default="./results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written"
        },
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform"},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass"
        },
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate for AdamW"},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay to apply"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for gradient clipping"},
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of steps for a linear warmup"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use"},
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps"},
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X updates steps"},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints"},
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model at the end of training"},
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "The metric to use to compare two different models"},
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"},
    )


@dataclass
class DataArguments:
    """
    Arguments for data configuration
    """

    dataset_name: str = field(
        default="Qwen/Qwen1.5-7B-Chat",
        metadata={"help": "The name of the dataset to use"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)"},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation data file (a text file)"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total sequence length for input text"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the dataset set to use for validation"},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total sequence length for input text after tokenization"},
    )


@dataclass
class WandbArguments:
    """
    Arguments for Weights & Biases configuration
    """

    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the W&B project"},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the W&B entity"},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the W&B run"},
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Tags for the W&B run"},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Group for the W&B run"},
    )
    wandb_job_type: Optional[str] = field(
        default=None,
        metadata={"help": "Job type for the W&B run"},
    )


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments

    Returns:
        Dict[str, Any]: Dictionary of parsed arguments
    """
    try:
        parser = argparse.ArgumentParser(description="Training script for language models")

        # Add argument groups
        model_args = ModelArguments()
        lora_args = LoRAArguments()
        training_args = TrainingArguments()
        data_args = DataArguments()
        wandb_args = WandbArguments()

        # Add arguments to parser
        for args in [model_args, lora_args, training_args, data_args, wandb_args]:
            for field_name, field in args.__dataclass_fields__.items():
                parser.add_argument(
                    f"--{field_name}",
                    type=field.type,
                    default=field.default,
                    help=field.metadata.get("help", ""),
                )

        # Parse arguments
        args = parser.parse_args()

        # Convert to dictionary
        args_dict = {
            "model": vars(model_args),
            "lora": vars(lora_args),
            "training": vars(training_args),
            "data": vars(data_args),
            "wandb": vars(wandb_args),
        }

        logger.info("Parsed arguments:")
        for group, group_args in args_dict.items():
            logger.info(f"{group}:")
            for key, value in group_args.items():
                logger.info(f"  {key}: {value}")

        return args_dict

    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise
