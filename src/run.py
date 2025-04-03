#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import unsloth  # Import unsloth first to apply all optimizations and avoid warnings
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb

try:
    import src
except:
    sys.path.append("../")

from src.callbacks.base_callback import BaseCallback
from src.callbacks.early_stopping_callback import EarlyStoppingCallback
from src.callbacks.lr_monitor_callback import LRMonitorCallback
from src.callbacks.model_loading_alert_callback import ModelLoadingAlertCallback
from src.callbacks.prompt_monitor_callback import PromptMonitorCallback
from src.callbacks.safety_checkpoint_callback import SafetyCheckpointCallback
from src.callbacks.validation_callback import ValidationCallback
from src.model.qwen_handler import HubConfig, ModelSource, QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.training.trainer import QwenTrainer
from src.utils.auth import setup_authentication
from src.utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("outputs/training.log")],
)
logger = logging.getLogger(__name__)


# Set reproducibility
def set_seed(seed):
    """Set random seed for reproducibility"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Set random seed to {seed} for reproducibility")


def setup_environment():
    """Setup training environment and configurations"""
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training will be slow!")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Setup authentication and get HuggingFace token
    try:
        setup_authentication()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set!")
    except Exception as e:
        logger.error(f"Authentication setup failed: {str(e)}")
        raise

    return hf_token


def setup_hub_configs(
    hf_token, source_model_id=None, destination_repo_id=None, private=False, save_method="lora"
):
    """
    Setup source and destination hub configurations

    Args:
        hf_token: Hugging Face token for authentication
        source_model_id: ID of the source model (defaults to Qwen2.5-Coder-1.5B-Instruct)
        destination_repo_id: ID for the destination repo (username/repo-name)
        private: Whether the destination repo should be private
        save_method: Method to use for saving the model

    Returns:
        Tuple of source and destination hub configurations
    """
    # Set default source model if not provided
    if not source_model_id:
        source_model_id = "unsloth/Qwen2.5-Coder-1.5B-Instruct"

    source_hub = HubConfig(model_id=source_model_id, token=hf_token)

    # Set default destination repo if not provided
    if not destination_repo_id:
        # Use the default repository name
        destination_repo_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice_v3"
        logger.info(f"Using default destination repository: {destination_repo_id}")

    # Check if the repository exists
    api = HfApi(token=hf_token)
    try:
        # Try to get the repo info to check if it exists
        api.repo_info(repo_id=destination_repo_id, repo_type="model")
        logger.info(f"Repository {destination_repo_id} already exists")
    except Exception as e:
        # If the repo doesn't exist, create it
        logger.info(f"Repository {destination_repo_id} not found, creating it...")
        try:
            create_repo(
                repo_id=destination_repo_id,
                token=hf_token,
                private=private,
                repo_type="model",
            )
            logger.info(f"Repository {destination_repo_id} created successfully")
            # Give HF a moment to register the new repo
            time.sleep(2)
        except Exception as create_error:
            logger.error(f"Failed to create repository: {str(create_error)}")
            raise

    destination_hub = HubConfig(
        model_id=destination_repo_id,
        token=hf_token,
        private=private,
        save_method=save_method,
    )

    logger.info(f"Source model: {source_hub.model_id}")
    logger.info(f"Destination model: {destination_hub.model_id}")

    return source_hub, destination_hub


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a Qwen model for multiple choice questions")

    # Model configuration
    parser.add_argument(
        "--source-model",
        type=str,
        default="unsloth/Qwen2.5-Coder-1.5B-Instruct",
        help="Source model ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--destination-repo",
        type=str,
        default="tuandunghcmut/Qwen25_Coder_MultipleChoice_v2",
        help="Destination repository ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length for the model"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization method for model loading (4bit, 8bit, or none)",
    )
    parser.add_argument(
        "--peft-type",
        type=str,
        default="lora",
        choices=["lora", "prefix_tuning", "prompt_tuning", "p_tuning"],
        help="Type of PEFT method to use",
    )

    # Training configuration
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=24, help="Per device batch size for training"
    )
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps (overrides warmup-ratio if provided)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Ratio of total steps to use for warmup (ignored if warmup-steps is provided)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_output",
        help="Directory to save model outputs (will be created inside the 'outputs' folder)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3, help="Patience for early stopping"
    )
    parser.add_argument(
        "--early-stopping-delta",
        type=float,
        default=0.01,
        help="Minimum change to qualify as improvement for early stopping",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="yaml_reasoning",
        choices=[
            "yaml_reasoning",
            "basic",
            "teacher_reasoned",
            "options",
            "socratic",
            "scientist",
            "lawyer",
            "debugger",
            "philosopher",
            "expert_novice",
            "pros_cons",
            "code_review",
            "math_proof",
        ],
        help="Prompt template to use for formatting",
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Use only 2 dataset instances for quick testing"
    )
    parser.add_argument(
        "--test-training-mode",
        action="store_true",
        help="Use only enough examples to fill one batch (batch_size) for minimal training testing",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Name for this experiment (used for WandB and checkpoint naming)",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Number of samples to log for debugging (0 to disable)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Number of steps between logging updates (non-test mode)",
    )
    parser.add_argument(
        "--test-logging-steps",
        type=int,
        default=10,
        help="Number of steps between logging updates (test modes)",
    )
    parser.add_argument(
        "--test-save-steps",
        type=int,
        default=20,
        help="Number of steps between model checkpoints (test modes)",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=50,
        help="Number of steps between validations",
    )
    parser.add_argument(
        "--metric-for-best",
        type=str,
        default="eval_loss",
        help="Metric to use for determining the best model",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        default=False,
        help="Whether greater values of the metric are better",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push the best model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--no-push-to-hub",
        action="store_false",
        dest="push_to_hub",
        help="Disable pushing the best model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--validate-at-start",
        action="store_true",
        default=True,
        help="Run validation before training starts",
    )
    parser.add_argument(
        "--no-validate-at-start",
        action="store_false",
        dest="validate_at_start",
        help="Disable validation before training starts",
    )

    # Prompt monitoring configuration
    # parser.add_argument(
    #     "--prompt-track-diversity",
    #     action="store_true",
    #     default=True,
    #     help="Track prompt diversity during training",
    # )
    # parser.add_argument(
    #     "--prompt-track-quality",
    #     action="store_true",
    #     default=True,
    #     help="Track prompt quality metrics during training",
    # )
    # parser.add_argument(
    #     "--prompt-interactive",
    #     action="store_true",
    #     default=False,
    #     help="Enable interactive prompt selection mode",
    # )
    # parser.add_argument(
    #     "--prompt-categorize",
    #     action="store_true",
    #     default=True,
    #     help="Automatically categorize prompts",
    # )
    # parser.add_argument(
    #     "--prompt-comparison",
    #     action="store_true",
    #     default=True,
    #     help="Enable prompt comparison features",
    # )
    # parser.add_argument(
    #     "--max-prompts-to-save",
    #     type=int,
    #     default=100,
    #     help="Maximum number of prompts to save for analysis",
    # )

    # Repository configuration
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the destination repository private",
    )
    parser.add_argument(
        "--save-method",
        type=str,
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf"],
        help="Method to use for saving the model",
    )
    parser.add_argument(
        "--push-strategy",
        type=str,
        default="best",
        choices=["best", "end", "all", "no"],
        help="When to push to hub: 'best'=best checkpoint, 'end'=end of training, 'all'=each save, 'no'=don't push",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="tuandunghcmut/coding-mcq-reasoning",
        help="Dataset ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction of data to use for validation"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer",
        type=str,
        default="lion_8bit",
        choices=[
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_torch_xla",
            "adamw_torch_npu_fused",
            "adamw_apex_fused",
            "adafactor",
            "adamw_anyprecision",
            "adamw_torch_4bit",
            "adamw_torch_8bit",
            "ademamix",
            "sgd",
            "adagrad",
            "adamw_bnb_8bit",
            "adamw_8bit",
            "ademamix_8bit",
            "lion_8bit",
            "lion_32bit",
            "paged_adamw_32bit",
            "paged_adamw_8bit",
            "paged_ademamix_32bit",
            "paged_ademamix_8bit",
            "paged_lion_32bit",
            "paged_lion_8bit",
            "rmsprop",
            "rmsprop_bnb",
            "rmsprop_bnb_8bit",
            "rmsprop_bnb_32bit",
            "galore_adamw",
            "galore_adamw_8bit",
            "galore_adafactor",
            "galore_adamw_layerwise",
            "galore_adamw_8bit_layerwise",
            "galore_adafactor_layerwise",
            "lomo",
            "adalomo",
            "grokadamw",
            "schedule_free_radam",
            "schedule_free_adamw",
            "schedule_free_sgd",
            "apollo_adamw",
            "apollo_adamw_layerwise",
        ],
        help="Optimizer to use for training",
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Beta1 for Adam-based optimizers"
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999, help="Beta2 for Adam-based optimizers"
    )
    parser.add_argument(
        "--adam-epsilon", type=float, default=1e-8, help="Epsilon for Adam-based optimizers"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--optim-bits",
        type=int,
        default=8,
        choices=[8, 32],
        help="Quantization bits for 8-bit optimizers",
    )

    # Learning Rate Scheduler
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine_with_warmup",
        choices=[
            "cosine_with_warmup",
            "cosine",
            "linear",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-scheduler-num-cycles",
        type=int,
        default=1,
        help="Number of cycles for cosine_with_restarts",
    )
    parser.add_argument(
        "--lr-scheduler-power",
        type=float,
        default=1.0,
        help="Power factor for polynomial scheduler",
    )
    parser.add_argument(
        "--lr-scheduler-last-epoch",
        type=int,
        default=-1,
        help="Index of last epoch when resuming training",
    )

    # Module targeting options
    parser.add_argument(
        "--legacy-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="[Deprecated] Use --target-modules instead",
    )
    parser.add_argument(
        "--fan-in-fan-out",
        type=bool,
        default=False,
        help="Set fan_in_fan_out for Conv1D",
    )
    parser.add_argument(
        "--legacy-gradient-checkpointing",
        type=bool,
        default=False,
        help="Use gradient checkpointing (legacy boolean parameter)",
    )
    parser.add_argument(
        "--modules-to-save",
        type=str,
        default=None,
        help="Modules to save in full precision",
    )

    # Response-only training options
    parser.add_argument(
        "--train-on-responses-only",
        action="store_true",
        help="Enable response-only training",
    )
    parser.add_argument(
        "--instruction-token",
        type=str,
        default="<|im_start|>user\n",
        help="Token/prefix indicating start of instruction",
    )
    parser.add_argument(
        "--response-token",
        type=str,
        default="<|im_start|>assistant\n",
        help="Token/prefix indicating start of response",
    )
    parser.add_argument(
        "--instruction-token-id",
        type=int,
        default=None,
        help="Token ID for instruction start (optional)",
    )
    parser.add_argument(
        "--response-token-id",
        type=int,
        default=None,
        help="Token ID for response start (optional)",
    )

    # Attention implementation
    parser.add_argument(
        "--attention-implementation",
        type=str,
        default="flash_attention_2",
        choices=["default", "flash_attention_2", "sdpa", "eager", "xformers"],
        help="Type of attention implementation to use",
    )
    parser.add_argument(
        "--use-flash-attention",
        action="store_true",
        help="Use Flash Attention 2 if available (shortcut for setting attention-implementation=flash_attention_2)",
    )
    parser.add_argument(
        "--force-attn-implementation",
        action="store_true",
        help="Force the attention implementation even if not optimal for the hardware",
    )
    # Lion optimizer parameters
    parser.add_argument(
        "--lion-beta1",
        type=float,
        default=0.9,
        help="Beta1 parameter for Lion optimizer",
    )
    parser.add_argument(
        "--lion-beta2",
        type=float,
        default=0.99,
        help="Beta2 parameter for Lion optimizer",
    )

    # Validation options
    parser.add_argument(
        "--minimal-validating",
        action="store_true",
        help="Enable minimal validation to speed up training",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=100,
        help="Maximum number of samples to use for validation when minimal-validating is enabled",
    )

    # LoRA configuration
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA attention dimension",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--gradient-checkpointing-ratio",
        type=float,
        default=1.0,
        help="Ratio of layers to use gradient checkpointing",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--save-safetensors",
        action="store_true",
        default=True,
        help="Save model in safetensors format",
    )

    # Training configuration
    training = parser.add_argument_group("training")
    training.add_argument(
        "--save-steps",
        type=int,
        default=30,
        help="Number of steps between safety checkpoints (default: 30)",
    )
    training.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Maximum number of safety checkpoints to keep (default: 5)",
    )
    training.add_argument(
        "--save-strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch", "no"],
        help="Strategy to save checkpoints (default: steps)",
    )

    # New command line arguments
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for data loading",
    )
    parser.add_argument(
        "--full-determinism",
        action="store_true",
        default=False,
        help="Enable full determinism in training",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for training",
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        default=False,
        help="Use CPU for training",
    )
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy for training",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--eval-delay",
        type=int,
        default=0,
        help="Number of steps to delay evaluation",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Where to report training metrics",
    )
    parser.add_argument(
        "--remove-unused-columns",
        action="store_true",
        default=False,
        help="Remove unused columns from dataset",
    )

    # Add options for load_best_model_at_end
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        default=True,
        help="Load the best model at the end of training",
    )
    parser.add_argument(
        "--no-load-best-model-at-end",
        action="store_false",
        dest="load_best_model_at_end",
        help="Disable loading the best model at the end of training",
    )

    return parser.parse_args()


def setup_model_and_trainer(source_hub, destination_hub, args):
    """Initialize model handler and trainer"""
    try:
        # Initialize model handler
        logger.info("Initializing model handler...")

        # Configure quantization
        quantization_config = None
        if args.quantization != "none":
            from transformers import BitsAndBytesConfig

            if args.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif args.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

        model_handler = QwenModelHandler(
            model_name=source_hub.model_id,
            max_seq_length=args.max_seq_length,
            quantization=args.quantization,  # Pass the quantization string
            model_source=ModelSource.UNSLOTH,
            device_map="auto",
            source_hub_config=source_hub,
        )

        # Configure PEFT method
        logger.info(f"Setting up {args.peft_type} configuration...")

        # Get target modules string using either new or legacy parameter
        target_modules_str = getattr(args, "target_modules", None) or getattr(
            args, "legacy_target_modules", ""
        )
        target_modules = (
            target_modules_str.split(",")
            if target_modules_str
            else ["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        # Set up PEFT configuration based on type
        if args.peft_type == "lora":
            from peft import LoraConfig

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
        else:
            raise ValueError(f"Only LoRA is currently supported. Got {args.peft_type}")

        # Get prompt template based on arg
        prompt_type = get_prompt_template(args.prompt_template)

        # Set response-only training configuration if enabled
        response_only_config = None
        if args.train_on_responses_only:
            response_only_config = {
                "instruction_token": args.instruction_token,
                "response_token": args.response_token,
                "instruction_token_id": args.instruction_token_id,
                "response_token_id": args.response_token_id,
            }

        # Configure attention implementation if specified
        attention_config = {
            "implementation": args.attention_implementation,
            "force_implementation": args.force_attn_implementation,
        }

        # If use_flash_attention is set, override the implementation choice
        if args.use_flash_attention:
            attention_config["implementation"] = "flash_attention_2"

        # Initialize trainer
        logger.info("Initializing QwenTrainer instance...")
        trainer = QwenTrainer(
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            prompt_creator=PromptCreator(prompt_type),
            lora_config=lora_config,
            destination_hub_config=destination_hub,
            debug_samples=args.debug_samples,
            responses_only_config=response_only_config,
            attention_config=attention_config,
        )

        return trainer

    except Exception as e:
        logger.error(f"Error in setup: {str(e)}")
        raise


def get_prompt_template(template_name):
    """Get prompt template constant from name"""
    templates = {
        "yaml_reasoning": PromptCreator.YAML_REASONING,
        "basic": PromptCreator.BASIC,
        "teacher_reasoned": PromptCreator.TEACHER_REASONED,
        "options": PromptCreator.OPTIONS,
    }
    return templates.get(template_name, PromptCreator.YAML_REASONING)


def load_datasets(hf_token, dataset_id, test_mode=False, test_training_mode=False, batch_size=32):
    """
    Load datasets from HuggingFace Hub

    Args:
        hf_token: HuggingFace token for authentication
        dataset_id: ID of the dataset on HuggingFace Hub
        test_mode: If True, use only 2 dataset instances for quick testing
        test_training_mode: If True, use only enough examples to fill one batch
        batch_size: Batch size for training (used when test_training_mode is True)

    Returns:
        Dataset: Training dataset
    """
    try:
        logger.info(f"Loading dataset {dataset_id} from HuggingFace Hub...")
        dataset = load_dataset(dataset_id, token=hf_token, split="train")
        logger.info(f"Loaded {len(dataset)} training examples")

        # Apply test mode if enabled
        if test_mode:
            logger.info("TEST MODE ENABLED: Using only 2 dataset instances")
            dataset = dataset.select(range(2))
            logger.info(f"Dataset reduced to {len(dataset)} examples")
        elif test_training_mode:
            # Use one full batch + a few extra examples for validation if needed
            num_examples = batch_size + max(
                2, int(batch_size * 0.2)
            )  # batch_size + 20% for validation
            logger.info(
                f"TEST TRAINING MODE ENABLED: Using only {num_examples} dataset instances (one batch + validation)"
            )
            dataset = dataset.select(range(min(num_examples, len(dataset))))
            logger.info(f"Dataset reduced to {len(dataset)} examples")

        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info(f"Features: {list(dataset.features.keys())}")
        logger.info(f"Example:\n{dataset[0]}")

        return dataset

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def main():
    try:
        # Parse command line arguments
        args = parse_args()

        # Setup environment
        hf_token = setup_environment()

        # Generate experiment name if not provided
        if not args.experiment_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if args.test_mode:
                args.experiment_name = f"test_lora_{timestamp}"
            elif args.test_training_mode:
                args.experiment_name = f"test_train_lora_{timestamp}"
            else:
                args.experiment_name = f"lora_experiment_{timestamp}"

        logger.info(f"Using experiment name: {args.experiment_name}")

        # Setup hub configurations
        source_hub, destination_hub = setup_hub_configs(
            hf_token=hf_token,
            source_model_id=args.source_model,
            destination_repo_id=args.destination_repo,
            private=args.private,
            save_method=args.save_method,
        )

        # Initialize model and trainer
        logger.info("Initializing LoRA model and trainer...")

        # Configure model loading
        model_kwargs = {
            "device_map": "auto",
        }

        # Add memory constraints if specified
        if hasattr(args, "max_memory") and args.max_memory:
            import json

            model_kwargs["max_memory"] = json.loads(args.max_memory)
        if hasattr(args, "max_memory_per_gpu") and args.max_memory_per_gpu:
            if "max_memory" not in model_kwargs:
                model_kwargs["max_memory"] = {}
            for i in range(torch.cuda.device_count()):
                model_kwargs["max_memory"][i] = args.max_memory_per_gpu

        trainer = setup_model_and_trainer(
            source_hub=source_hub, destination_hub=destination_hub, args=args
        )

        # Load dataset from HuggingFace Hub
        train_dataset = load_datasets(
            hf_token,
            args.dataset,
            test_mode=args.test_mode,
            test_training_mode=args.test_training_mode,
            batch_size=args.batch_size,
        )

        # Ensure output directory is inside the 'outputs' folder
        outputs_root = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_root, exist_ok=True)

        # Create full output path with experiment name
        output_dir = os.path.join(outputs_root, args.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Training outputs will be saved to: {output_dir}")

        # Create a symlink to the latest output
        latest_link = os.path.join(outputs_root, "latest")
        if os.path.exists(latest_link) and os.path.islink(latest_link):
            os.remove(latest_link)

        try:
            os.symlink(output_dir, latest_link, target_is_directory=True)
            logger.info(f"Created symlink: {latest_link} -> {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create symlink: {e}")

        # Initialize callbacks
        callbacks = []

        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            patience=args.early_stopping_patience, min_delta=args.early_stopping_delta
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LRMonitorCallback(trainer=trainer)
        callbacks.append(lr_monitor)

        # Model loading alert
        model_loading_alert = ModelLoadingAlertCallback(use_unsloth=True)
        callbacks.append(model_loading_alert)

        # Safety checkpoint callback
        safety_checkpoint = SafetyCheckpointCallback(
            save_steps=args.save_steps, save_total_limit=args.save_total_limit
        )
        # Set trainer attribute on safety checkpoint callback
        safety_checkpoint.trainer = trainer
        callbacks.append(safety_checkpoint)

        # Validation callback - create this last so it has access to the validation dataset
        # that will be created during the call to trainer.train()
        validation_callback = ValidationCallback(
            trainer_instance=trainer,
            validation_steps=args.validation_steps,
            push_to_hub=args.push_to_hub,
            metric_for_best=args.metric_for_best,
            greater_is_better=args.greater_is_better,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_delta,
            validate_at_start=args.validate_at_start,
            minimal_validating=args.minimal_validating,
            max_validation_samples=args.max_validation_samples,
        )

        callbacks.append(validation_callback)

        # Setup WandB logging
        try:
            import wandb

            # Create WandB configuration
            model_name = args.source_model.split("/")[-1]
            project_name = f"{model_name}-LoRA-Training"

            run_prefix = (
                "TEST_" if args.test_mode else "TEST-TRAIN_" if args.test_training_mode else ""
            )
            run_name = f"{run_prefix}{args.experiment_name}_b{args.batch_size}_lr{args.learning_rate}_e{args.epochs}"

            tags = ["qwen", "lora", "coding", "multiple-choice"]
            if args.test_mode:
                tags.append("test_mode")
            elif args.test_training_mode:
                tags.append("test_training_mode")

            notes = f"{run_prefix}LoRA training"

            wandb_config = {
                "project": project_name,
                "name": run_name,
                "tags": tags,
                "notes": notes,
                "config": {
                    "model": {
                        "name": args.source_model,
                    },
                    "lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "target_modules": getattr(args, "target_modules", None)
                        or getattr(args, "legacy_target_modules", ""),
                    },
                    "training": {
                        "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "warmup_ratio": args.warmup_ratio,
                        "max_steps": args.max_train_steps,
                    },
                },
            }

            # Initialize WandB
            if wandb.run is None:
                logger.info(f"Initializing WandB run: {run_name}")
                wandb.init(**wandb_config)
                logger.info(f"WandB run initialized: {wandb.run.name}")
                logger.info(f"WandB run URL: {wandb.run.get_url()}")

            # Add WandB callback
            from src.utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

            # Create a WandBConfig with the same settings
            wandb_logger_config = WandBConfig(
                project_name=project_name,
                run_name=run_name,
                tags=tags,
                notes=notes,
                config=wandb_config.get("config"),
            )
            wandb_logger = WandBLogger(config=wandb_logger_config)
            wandb_callback = WandBCallback(logger=wandb_logger)
            callbacks.append(wandb_callback)

        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            logger.warning("Continuing without WandB logging")

        # Start training
        logger.info("Starting LoRA training...")

        results = trainer.train(
            train_dataset=train_dataset,
            val_split=args.val_split,
            output_dir=output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_train_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best,
            greater_is_better=args.greater_is_better,
            callbacks=callbacks,
            random_seed=args.random_seed,
            push_to_hub_strategy=args.push_strategy,
            optimizer_config={
                "optimizer_type": args.optimizer,
                "weight_decay": args.weight_decay,
                "beta1": args.lion_beta1 if args.optimizer.startswith("lion") else args.adam_beta1,
                "beta2": args.lion_beta2 if args.optimizer.startswith("lion") else args.adam_beta2,
                "epsilon": args.adam_epsilon,
                "max_grad_norm": args.max_grad_norm,
            },
            lr_scheduler_config={
                "lr_scheduler_type": "cosine",
                "num_cycles": 1,
                "power": 1.0,
            },
        )

        # Log results
        logger.info("LoRA training completed!")
        if isinstance(results, dict):
            logger.info("Training metrics:")
            for key, value in results.items():
                logger.info(f"{key}: {value}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback

        logger.error(f"Error details: {traceback.format_exc()}")
        return 1
    finally:
        # Cleanup wandb
        try:
            import wandb

            if "wandb" in globals() and wandb.run is not None:
                wandb.finish()
        except Exception as e:
            logger.warning(f"Failed to clean up wandb: {e}")
            pass


if __name__ == "__main__":
    sys.exit(main())
