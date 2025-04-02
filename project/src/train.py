import os
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List
from peft import LoraConfig
from datasets import load_from_disk

from src.model.qwen_handler import QwenModelHandler
from src.data.prompt_creator import PromptCreator
from src.training.trainer import QwenTrainer
from src.training.callbacks import ValidationCallback, EarlyStoppingCallback
from src.utils.wandb_logger import WandBLogger, WandBConfig, WandBCallback

def setup_wandb_logging(
    model_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    project_name: str = "qwen-multiple-choice",
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> WandBLogger:
    """Setup W&B logging with comprehensive configuration"""
    wandb_config = WandBConfig(
        project_name=project_name,
        entity=entity,
        run_name=run_name,
        tags=tags,
        notes=notes,
        config=config,
    )
    
    logger = WandBLogger(wandb_config)
    logger.init_run(model_name)
    return logger

def setup_model_and_trainer(
    model_name: str = "Unsloth/Qwen2.5-7B",
    max_seq_length: int = 2048,
    quantization: Optional[str] = "4bit",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    prompt_type: str = PromptCreator.TEACHER_REASONED,
    hub_token: Optional[str] = None,
    hub_model_id: Optional[str] = None,
) -> tuple:
    """Setup model, tokenizer, and trainer"""
    # Initialize model handler
    model_handler = QwenModelHandler(
        model_name=model_name,
        max_seq_length=max_seq_length,
        quantization=quantization,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    # Initialize LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Initialize trainer
    trainer = QwenTrainer(
        model=model_handler.model,
        tokenizer=model_handler.tokenizer,
        prompt_creator=PromptCreator(prompt_type),
        lora_config=lora_config,
        hub_token=hub_token,
        hub_model_id=hub_model_id,
    )

    return model_handler, trainer

def train(
    # Model configuration
    model_name: str = "Unsloth/Qwen2.5-7B",
    max_seq_length: int = 2048,
    quantization: str = "4bit",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
    
    # Data paths
    train_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/dataset_ready_for_training",
    val_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_val_filtered",
    
    # Training parameters
    output_dir: str = "./model_output",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
    train_on_response_only: bool = True,
    packing: bool = False,
    
    # Logging and saving
    logging_steps: int = 10,
    save_steps: int = 100,
    run_name: Optional[str] = None,
    log_memory: bool = True,
    log_gradients: bool = True,
    
    # Validation
    validation_strategy: str = "steps",
    validation_steps: Optional[int] = 100,
    validation_epochs: int = 1,
    save_best_checkpoint: bool = True,
    early_stopping_patience: Optional[int] = 7,
    quality_val_size: int = 256,
    
    # Prompt and model type
    prompt_type: str = PromptCreator.TEACHER_REASONED,
    
    # HuggingFace Hub
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    
    # W&B configuration
    project_name: str = "qwen-multiple-choice",
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    
    # Other settings
    verbose: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Comprehensive training entrypoint with all features
    
    Args:
        model_name: Name or path of the model
        max_seq_length: Maximum sequence length
        quantization: Quantization type ("4bit", "8bit", or None)
        device_map: Device mapping strategy
        cache_dir: Cache directory for models
        train_path: Path to training dataset
        val_path: Path to validation dataset
        output_dir: Directory to save outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_steps: Maximum number of training steps
        train_on_response_only: Whether to use Unsloth's response-only training mode (ignores loss on user inputs)
        packing: Whether to use sequence packing
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        run_name: Name for the training run
        log_memory: Whether to log memory usage
        log_gradients: Whether to log gradient statistics
        validation_strategy: When to validate ("epoch", "steps", "no")
        validation_steps: Validate every N steps
        validation_epochs: Validate every N epochs
        save_best_checkpoint: Whether to save best checkpoint
        early_stopping_patience: Patience for early stopping
        quality_val_size: Size of quality validation set
        prompt_type: Type of prompt to use
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: Model ID for HuggingFace Hub
        hub_token: HuggingFace Hub token
        project_name: W&B project name
        entity: W&B entity name
        tags: W&B tags
        notes: W&B notes
        verbose: Whether to print verbose output
        seed: Random seed
        
    Returns:
        Dictionary containing training results
    """
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load datasets
    if verbose:
        print("Loading datasets...")
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path) if val_path else None

    if verbose:
        print(f"Loaded training dataset with {len(train_dataset)} examples")
        if val_dataset:
            print(f"Loaded validation dataset with {len(val_dataset)} examples")

    # Setup wandb config
    wandb_config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "quantization": quantization,
        "num_train_epochs": num_train_epochs,
        "batch_size": per_device_train_batch_size,
        "gradient_accumulation": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "train_on_response_only": train_on_response_only,
        "validation_strategy": validation_strategy,
        "prompt_type": prompt_type,
        "dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset) if val_dataset else 0,
    }

    # Setup wandb logging
    wandb_logger = setup_wandb_logging(
        model_name=model_name,
        run_name=run_name,
        config=wandb_config,
        project_name=project_name,
        entity=entity,
        tags=tags,
        notes=notes,
    )

    # Setup model and trainer
    model_handler, trainer = setup_model_and_trainer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        quantization=quantization,
        device_map=device_map,
        cache_dir=cache_dir,
        prompt_type=prompt_type,
        hub_token=hub_token,
        hub_model_id=hub_model_id,
    )

    # Log model information
    wandb_logger.log_model_info(model_handler.model)

    # Print training configuration
    if verbose:
        print("\nTraining Configuration:")
        print(f"Model: {model_name}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Batch size: {per_device_train_batch_size} x {gradient_accumulation_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Number of epochs: {num_train_epochs}")
        print(f"Validation strategy: {validation_strategy}")
        print(f"Prompt type: {prompt_type}")
        print(f"Push to Hub: {push_to_hub}")
        if push_to_hub:
            print(f"Hub model ID: {hub_model_id}")

    # Start training
    training_results = trainer.train(
        dataset=train_dataset,
        val_dataset=val_dataset,
        prompt_type=prompt_type,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        train_on_inputs=not train_on_response_only,
        packing=packing,
        logging_steps=logging_steps,
        save_steps=save_steps,
        verbose=verbose,
        push_to_hub=push_to_hub,
        validation_strategy=validation_strategy,
        validation_steps=validation_steps,
        validation_epochs=validation_epochs,
        save_best_checkpoint=save_best_checkpoint,
        early_stopping_patience=early_stopping_patience,
        run_name=run_name,
        log_memory=log_memory,
        log_gradients=log_gradients,
        quality_val_size=quality_val_size,
        callbacks=[WandBCallback(wandb_logger)],
    )

    # Apply Unsloth's train_on_responses_only if train_on_response_only is True
    if train_on_response_only and torch.cuda.device_count() == 1:  # Only for single GPU
        if verbose:
            print("Enabling Unsloth's response-only training mode (ignoring loss on user inputs)")
        
        # Import Unsloth's function for training only on responses
        from unsloth.chat_templates import train_on_responses_only
        
        # Apply the function to our trainer with Qwen's chat markers
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    # Print final results
    if verbose:
        print("\nTraining completed!")
        print(f"Best validation metric: {trainer.best_val_metric:.4f}")
        if trainer.best_checkpoint_path:
            print(f"Best checkpoint saved at: {trainer.best_checkpoint_path}")

    # Finish W&B run
    wandb_logger.finish()

    return {
        "training_stats": training_results,
        "best_val_metric": trainer.best_val_metric,
        "best_checkpoint_path": trainer.best_checkpoint_path,
    }

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen model for multiple choice questions")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Unsloth/Qwen2.5-7B")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--quantization", type=str, default="4bit")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./model_output")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # W&B arguments
    parser.add_argument("--project_name", type=str, default="qwen-multiple-choice")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--notes", type=str)
    
    # HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    
    args = parser.parse_args()

    # Run training
    results = train(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        quantization=args.quantization,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        project_name=args.project_name,
        entity=args.entity,
        tags=args.tags,
        notes=args.notes,
    ) 