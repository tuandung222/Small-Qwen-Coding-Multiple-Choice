import argparse
from typing import Dict, Any
from src.config.training_config import (
    ModelConfig,
    LoRAConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    TrainingConfig,
    ValidationConfig,
    DatasetConfig,
    PromptConfig,
    ResponseOnlyConfig,
    AttentionConfig,
    WandBConfig,
    EnvironmentConfig,
    CompleteConfig,
)

def parse_args() -> CompleteConfig:
    """
    Parse command line arguments
    
    Returns:
        CompleteConfig: Complete configuration object
    """
    parser = argparse.ArgumentParser(description="Train a Qwen model with LoRA")
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-Coder-1.5B-Instruct",
                           help="Model ID from HuggingFace Hub")
    model_group.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    
    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    lora_group.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    lora_group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    lora_group.add_argument("--lora_target_modules", type=str, nargs="+", 
                           default=["q_proj", "k_proj", "v_proj", "o_proj"],
                           help="LoRA target modules")
    
    # Optimizer arguments
    optimizer_group = parser.add_argument_group("Optimizer")
    optimizer_group.add_argument("--optimizer_type", type=str, default="adamw",
                               choices=["adamw", "torch_adamw"], help="Optimizer type")
    optimizer_group.add_argument("--learning_rate", type=float, default=2e-5,
                               help="Learning rate")
    optimizer_group.add_argument("--weight_decay", type=float, default=0.01,
                               help="Weight decay")
    
    # LR scheduler arguments
    scheduler_group = parser.add_argument_group("LR Scheduler")
    scheduler_group.add_argument("--scheduler_type", type=str, default="cosine",
                               choices=["cosine", "linear", "constant"],
                               help="Learning rate scheduler type")
    scheduler_group.add_argument("--warmup_ratio", type=float, default=0.1,
                               help="Warmup ratio")
    
    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--num_train_epochs", type=int, default=3,
                              help="Number of training epochs")
    training_group.add_argument("--per_device_train_batch_size", type=int, default=4,
                              help="Per device training batch size")
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=4,
                              help="Gradient accumulation steps")
    training_group.add_argument("--logging_steps", type=int, default=10,
                              help="Logging steps")
    training_group.add_argument("--save_steps", type=int, default=100,
                              help="Save steps")
    training_group.add_argument("--save_strategy", type=str, default="steps",
                              choices=["steps", "epoch", "no"],
                              help="Save strategy")
    training_group.add_argument("--save_total_limit", type=int, default=3,
                              help="Maximum number of checkpoints to keep")
    training_group.add_argument("--fp16", action="store_true", help="Use fp16 precision")
    training_group.add_argument("--gradient_checkpointing", action="store_true",
                              help="Use gradient checkpointing")
    
    # Validation arguments
    validation_group = parser.add_argument_group("Validation")
    validation_group.add_argument("--per_device_eval_batch_size", type=int, default=4,
                                help="Per device evaluation batch size")
    validation_group.add_argument("--eval_steps", type=int, default=100,
                                help="Evaluation steps")
    validation_group.add_argument("--evaluation_strategy", type=str, default="steps",
                                choices=["steps", "epoch", "no"],
                                help="Evaluation strategy")
    validation_group.add_argument("--load_best_model_at_end", action="store_true",
                                help="Load best model at end of training")
    validation_group.add_argument("--metric_for_best_model", type=str, default="accuracy",
                                help="Metric for best model")
    validation_group.add_argument("--greater_is_better", action="store_true",
                                help="Greater is better for metric")
    
    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument("--dataset_id", type=str, required=True,
                             help="Dataset ID from HuggingFace Hub")
    dataset_group.add_argument("--val_split", type=float, default=0.1,
                             help="Validation split ratio")
    dataset_group.add_argument("--max_length", type=int, default=2048,
                             help="Maximum sequence length")
    dataset_group.add_argument("--response_only", action="store_true",
                             help="Use response-only training")
    
    # WandB arguments
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--wandb_enabled", action="store_true",
                           help="Enable Weights & Biases logging")
    wandb_group.add_argument("--wandb_project", type=str,
                           default="Qwen2.5-Coder-1.5B-Instruct-Coding-Multiple-Choice",
                           help="Weights & Biases project name")
    wandb_group.add_argument("--wandb_run_name", type=str, help="Weights & Biases run name")
    wandb_group.add_argument("--wandb_group", type=str, help="Weights & Biases group")
    wandb_group.add_argument("--wandb_tags", type=str, nargs="+",
                           help="Weights & Biases tags")
    
    # Environment arguments
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--seed", type=int, default=42, help="Random seed")
    env_group.add_argument("--hf_token", type=str, required=True,
                         help="HuggingFace token")
    env_group.add_argument("--output_dir", type=str, default="./outputs",
                         help="Output directory")
    env_group.add_argument("--experiment_name", type=str, help="Experiment name")
    
    args = parser.parse_args()
    
    # Create configuration objects
    model_config = ModelConfig(
        model_id=args.model_id,
        use_bf16=args.use_bf16,
        token=args.hf_token,
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    
    optimizer_config = OptimizerConfig(
        type=args.optimizer_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    scheduler_config = LRSchedulerConfig(
        type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
    )
    
    training_config = TrainingConfig(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    validation_config = ValidationConfig(
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )
    
    dataset_config = DatasetConfig(
        dataset_id=args.dataset_id,
        val_split=args.val_split,
        max_length=args.max_length,
    )
    
    prompt_config = PromptConfig(
        instruction_template="### Instruction:\n{instruction}\n\n### Response:\n",
        input_template="### Input:\n{input}\n\n",
    )
    
    response_only_config = ResponseOnlyConfig(
        enabled=args.response_only,
    )
    
    attention_config = AttentionConfig(
        attention_mode="flash_attention_2",
    )
    
    wandb_config = WandBConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
    )
    
    environment_config = EnvironmentConfig(
        seed=args.seed,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    
    return CompleteConfig(
        model=model_config,
        lora=lora_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        training=training_config,
        validation=validation_config,
        dataset=dataset_config,
        prompt=prompt_config,
        response_only=response_only_config,
        attention=attention_config,
        wandb=wandb_config,
        environment=environment_config,
    ) 