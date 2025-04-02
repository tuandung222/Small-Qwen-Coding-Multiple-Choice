import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from .model.qwen_handler import QwenModelHandler, ModelSource
from .data.prompt_creator import PromptCreator
from .training.trainer import QwenTrainer
from .training.callbacks import ValidationCallback, EarlyStoppingCallback
from .utils.wandb_logger import WandBLogger, WandBConfig, WandBCallback
from .utils.auth import setup_authentication

def setup_wandb_logging(
    project_name: str,
    run_name: str = None,
    log_memory: bool = True,
    log_gradients: bool = True,
) -> WandBLogger:
    """Setup Weights & Biases logging"""
    config = WandBConfig(
        project_name=project_name,
        run_name=run_name,
        log_memory=log_memory,
        log_gradients=log_gradients,
    )
    logger = WandBLogger(config)
    logger.setup()
    return logger

def setup_model_and_trainer(
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 16,
    verbose: bool = False,
    model_source: str = ModelSource.HUGGINGFACE,
) -> tuple[QwenModelHandler, QwenTrainer]:
    """Setup model and trainer"""
    # Create model handler
    model_handler = QwenModelHandler(
        model_name=model_name,
        model_source=model_source,
        max_seq_length=2048,  # Default max sequence length
        quantization="4bit",  # Default to 4-bit quantization
        device_map="auto",
    )
    
    # Create prompt creator and trainer
    prompt_creator = PromptCreator(prompt_type=PromptCreator.TEACHER_REASONED)
    trainer = QwenTrainer(
        model=model_handler.model,
        tokenizer=model_handler.tokenizer,
        prompt_creator=prompt_creator,
        lora_config=LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    
    return model_handler, trainer

def train(
    model_name: str,
    train_path: str,
    val_path: str = None,
    output_dir: str = "./model_output",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 16,
    validation_strategy: str = "epoch",
    checkpoint_strategy: str = "best",
    train_on_response_only: bool = False,
    verbose: bool = False,
    model_source: str = ModelSource.HUGGINGFACE,
) -> dict:
    """
    Train a model on the given dataset
    
    Args:
        model_name: Name of the model to train
        train_path: Path to training dataset
        val_path: Path to validation dataset
        output_dir: Directory to save model outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        validation_strategy: Validation strategy ("epoch", "steps", "no")
        checkpoint_strategy: Checkpoint strategy ("best", "last", "all")
        train_on_response_only: Whether to train only on responses
        verbose: Whether to print verbose output
        model_source: Source of the model ("huggingface" or "unsloth")
        
    Returns:
        Dictionary containing training results
    """
    # Validate inputs
    if not os.path.exists(train_path):
        raise ValueError("Train path does not exist")
    if val_path and not os.path.exists(val_path):
        raise ValueError("Validation path does not exist")
    if validation_strategy not in ["epoch", "steps", "no"]:
        raise ValueError("Invalid validation strategy")
    if checkpoint_strategy not in ["best", "last", "all"]:
        raise ValueError("Invalid checkpoint strategy")
    if model_source not in [ModelSource.HUGGINGFACE, ModelSource.UNSLOTH]:
        raise ValueError("Invalid model source")
        
    # Setup authentication
    setup_authentication()
    
    # Load datasets
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path) if val_path else None
    
    # Setup model and trainer
    model_handler, trainer = setup_model_and_trainer(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        verbose=verbose,
        model_source=model_source,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Add validation callback if needed
    if validation_strategy != "no" and val_dataset:
        validation_callback = ValidationCallback(trainer_instance=trainer)
        callbacks.append(validation_callback)
        
    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(patience=3)
    callbacks.append(early_stopping_callback)
    
    # Add wandb callback
    wandb_logger = setup_wandb_logging(
        project_name=f"train_{model_name}",
        log_memory=True,
        log_gradients=True,
    )
    wandb_callback = WandBCallback(logger=wandb_logger)
    callbacks.append(wandb_callback)
    
    # Train model
    if train_on_response_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        callbacks=callbacks,
    )
    
    # Save checkpoints based on strategy
    if checkpoint_strategy in ["best", "all"]:
        trainer.save_checkpoint(
            os.path.join(output_dir, "checkpoints"),
            val_metric=results["best_val_metric"],
            is_best=True,
        )
    if checkpoint_strategy in ["last", "all"]:
        trainer.save_checkpoint(
            os.path.join(output_dir, "checkpoints"),
            is_best=False,
        )
        
    return results

if __name__ == "__main__":
    # Example usage
    results = train(
        model_name="Qwen/Qwen1.5-7B",  # or "unsloth/Qwen2.5-7B" for Unsloth model
        train_path="./data/train_dataset",
        val_path="./data/val_dataset",
        output_dir="./model_output",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        verbose=True,
        model_source=ModelSource.HUGGINGFACE,  # or ModelSource.UNSLOTH
    )
    print(results) 