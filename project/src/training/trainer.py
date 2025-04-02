import torch
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Any, List, Union
import os
from ..model.qwen_handler import QwenModelHandler
from ..data.prompt_creator import PromptCreator
from ..data.response_parser import ResponseParser
from .callbacks import ValidationCallback, EarlyStoppingCallback
from datetime import datetime
from unsloth import FastLanguageModel
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from ..utils.auth import setup_authentication

class QwenTrainer:
    """
    Training handler for Qwen models with support for both Hugging Face and Unsloth optimizations.
    
    This class provides a unified interface for:
    1. Fine-tuning Qwen models with LoRA (Low-Rank Adaptation)
    2. Supporting both standard training and teacher-guided training
    3. Handling model checkpointing and validation
    4. Integration with Weights & Biases for experiment tracking
    5. Pushing models to Hugging Face Hub
    
    Key Features:
    - Parameter-efficient fine-tuning using LoRA
    - Support for teacher-guided training with YAML-formatted responses
    - Automatic mixed precision training (FP16/BF16)
    - Integration with Unsloth for optimized training
    - Flexible prompt formatting through PromptCreator
    - Comprehensive validation and checkpointing
    
    Example usage:
    ```python
    trainer = QwenTrainer(
        model=model,
        tokenizer=tokenizer,
        prompt_creator=PromptCreator(PromptCreator.TEACHER_REASONED),
        lora_config=lora_config,
        hub_token="your_token",  # Optional
        hub_model_id="your/model"  # Optional
    )
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="./model_output",
        num_train_epochs=3,
        per_device_train_batch_size=16
    )
    ```
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_creator: PromptCreator,
        lora_config: Optional[LoraConfig] = None,
        hub_token: Optional[str] = None,
        hub_model_id: Optional[str] = None,
    ):
        """
        Initialize the QwenTrainer with model, tokenizer, and configuration.
        
        Args:
            model: The base model to fine-tune (Qwen model from HF or Unsloth)
            tokenizer: The tokenizer associated with the model
            prompt_creator: PromptCreator instance for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient training
            hub_token: Optional HuggingFace Hub token for model pushing
            hub_model_id: Optional model ID for HuggingFace Hub
        
        The trainer will:
        1. Set up authentication if needed
        2. Configure the model for training
        3. Initialize tracking metrics
        4. Set up sequence length constraints
        """
        # Setup authentication for HF Hub access
        setup_authentication()
        
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_creator = prompt_creator
        self.lora_config = lora_config
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        
        # Initialize training state
        self.peft_model = None
        self.trainer = None
        self.train_dataset = None
        self.val_dataset = None
        self.best_val_metric = float('inf')
        self.best_checkpoint_path = None
        self.training_stats = {}
        self.validation_stats = {}
        
        # Set maximum sequence length based on model config
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.max_seq_length = min(2048, self.model.config.max_position_embeddings)
        else:
            self.max_seq_length = 2048  # Default fallback
    
    def validate(
        self,
        val_dataset,
        quality_val_dataset=None,
        prompt_type=None,
        batch_size=64,
        temperature=0.0,
    ):
        """Validate the model on validation dataset"""
        if prompt_type:
            self.prompt_creator.set_prompt_type(prompt_type)

        validation_callback = ValidationCallback(
            trainer_instance=self,
            val_dataset=val_dataset,
            quality_val_dataset=quality_val_dataset,
            validation_strategy="steps",
            validation_steps=100,
            save_best_checkpoint=True,
            early_stopper=EarlyStoppingCallback(patience=3),
        )

        return validation_callback

    def _create_temp_model_handler(self):
        """Create a temporary model handler for validation"""
        temp_handler = QwenModelHandler(
            model_name=self.model.config.name_or_path,
            device_map="auto",
        )
        temp_handler.model = self.model
        temp_handler.tokenizer = self.tokenizer
        return temp_handler

    def save_checkpoint(self, output_dir, val_metric=None, is_best=False):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        if is_best:
            output_dir = os.path.join(output_dir, "best_model")
        else:
            output_dir = os.path.join(output_dir, "checkpoint")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if val_metric is not None:
            with open(os.path.join(output_dir, "val_metric.txt"), "w") as f:
                f.write(str(val_metric))

    def prepare_model_for_training(self) -> Any:
        """
        Prepare model for training with LoRA configuration.
        
        This method:
        1. Applies LoRA configuration if provided
        2. Uses Unsloth's optimizations when possible
        3. Configures gradient checkpointing
        4. Sets up mixed precision training
        
        Returns:
            The prepared model ready for training
        """
        if self.lora_config:
            try:
                # Try using Unsloth's optimized LoRA implementation
                # Extract parameters from the PEFT LoraConfig
                r = self.lora_config.r
                lora_alpha = self.lora_config.lora_alpha
                lora_dropout = self.lora_config.lora_dropout
                target_modules = self.lora_config.target_modules
                
                # Use Unsloth's LoRA implementation
                self.peft_model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )
                
                # Print parameters info
                total_params = sum(p.numel() for p in self.peft_model.parameters())
                trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
                print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
                
                return self.peft_model
                
            except Exception as e:
                print(f"Failed to use Unsloth's LoRA implementation: {e}")
                print("Falling back to standard PEFT LoRA")
                
                # Fallback to standard PEFT LoRA
                self.peft_model = get_peft_model(self.model, self.lora_config)
                self.peft_model.print_trainable_parameters()
                return self.peft_model
        
        # If no LoRA config, return the base model
        self.model.train()
        return self.model

    def prepare_dataset(self, dataset: Any, prompt_type: Optional[str] = None, verbose: bool = False) -> Any:
        """Prepare dataset for SFTTrainer by returning text field"""
        # Temporarily set prompt type if provided
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.prompt_type = prompt_type
            
        if verbose:
            print(f"Preparing dataset with prompt type: {self.prompt_creator.prompt_type}")
            if self.prompt_creator.is_teacher_mode() and "yml_str" not in dataset.features:
                print("WARNING: Teacher mode enabled but 'yml_str' field not found in dataset")
            
        def format_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Transform function applied to each example during training"""
            questions = example["question"]
            choices = example["choices"] 
            answer = example["answer"]
            
            # For teacher mode, use the yml_str if available
            assistant_response = None
            if self.prompt_creator.is_teacher_mode() and "yml_str" in example:
                assistant_response = example["yml_str"]
            
            # Call create_training_prompt without task_id parameter
            user_prompt = self.prompt_creator.create_training_prompt(
                question=questions,
                choices=choices
            )
            
            if assistant_response is None:
                # Default to simple answer format if no teacher completion available
                assistant_response = answer
            
            # Apply chat template for training
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Return text field for SFTTrainer
            return {"text": text}
        
        # Use map instead of with_transform for reliable transformation
        columns_to_remove = [col for col in ['task_id', 'question', 'choices', 'answer', 'yml_str'] 
                            if col in dataset.features]
        
        transformed_dataset = dataset.map(
            format_example, 
            remove_columns=columns_to_remove,
            batched=False
        )
        
        # Preview the transformed data
        if verbose:
            print("Preview of transformed data:")
            print(f"Keys: {list(transformed_dataset[0].keys())}")
            sample_text = transformed_dataset[0]['text']
            sample_length = len(self.tokenizer.encode(sample_text))
            print(f"Text sample: {sample_text[:100]}...")
            print(f"Encoded length of first sample: {sample_length} tokens")
            
            # Check for potential length issues
            if sample_length > self.max_seq_length:
                print(f"WARNING: Sample exceeds max sequence length ({sample_length} > {self.max_seq_length})")
            
        # Restore original prompt type if changed
        if original_prompt_type is not None:
            self.prompt_creator.prompt_type = original_prompt_type
            
        return transformed_dataset
    
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
        warmup_steps: int = 100,
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
    ) -> Dict[str, Any]:
        """
        Train the model using optimized settings for Qwen models.
        
        This method:
        1. Prepares datasets with proper prompt formatting
        2. Automatically splits training data for validation if needed
        3. Configures training arguments for optimal performance
        4. Sets up LoRA if configured
        5. Handles mixed precision training automatically
        6. Manages checkpointing and validation
        
        Args:
            train_dataset: Dataset for training
            val_dataset: Optional dataset for validation
            val_split: Fraction of training data to use for validation if val_dataset is None
            output_dir: Directory to save model checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per GPU/CPU
            gradient_accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate for training
            warmup_steps: Number of warmup steps for learning rate scheduler
            max_steps: Maximum number of training steps (overrides num_train_epochs)
            logging_steps: Number of steps between logging updates
            save_steps: Number of steps between model saves
            save_strategy: When to save checkpoints ("steps", "epoch", or "no")
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Whether to load the best model after training
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            callbacks: Optional list of training callbacks
            random_seed: Random seed for dataset splitting and shuffling
            
        Returns:
            Dictionary containing training results and metrics
        """
        # If no validation dataset is provided, split the training dataset
        if val_dataset is None and load_best_model_at_end:
            print(f"No validation dataset provided. Splitting training data with {val_split:.1%} validation split...")
            # Shuffle and split the dataset
            shuffled_dataset = train_dataset.shuffle(seed=random_seed)
            split_datasets = shuffled_dataset.train_test_split(
                test_size=val_split,
                seed=random_seed
            )
            train_dataset = split_datasets["train"]
            val_dataset = split_datasets["test"]
            print(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation examples")
        
        # Prepare datasets with proper prompt formatting
        train_dataset = self.prepare_dataset(train_dataset, verbose=True)
        if val_dataset is not None:
            val_dataset = self.prepare_dataset(val_dataset, verbose=True)
        
        # Import required components
        from transformers import Trainer, TrainingArguments
        from unsloth import is_bfloat16_supported
        import os
        
        # Prepare model with LoRA if configured
        model_to_train = self.prepare_model_for_training()
        
        if model_to_train is None:
            raise RuntimeError("Model preparation failed")
        
        # Setup training arguments
        training_args_dict = {
            # Basic training configuration
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            
            # Logging and saving configuration
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_strategy": save_strategy,
            "save_total_limit": save_total_limit,
            
            # Model selection configuration
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
            
            # Mixed precision training
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            
            # Optimizer and scheduler configuration
            "optim": "paged_adamw_8bit",  # Memory-efficient optimizer
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            
            # Integration configuration
            "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
            "push_to_hub": bool(self.hub_model_id),
            "hub_model_id": self.hub_model_id,
            "hub_token": self.hub_token,
            
            # Set random seed
            "seed": random_seed,
        }
        
        # Handle max_steps vs num_train_epochs
        if max_steps is not None and max_steps > 0:
            training_args_dict["max_steps"] = max_steps
            training_args_dict.pop("num_train_epochs", None)
            
        # Set evaluation strategy to match save strategy if using load_best_model_at_end
        if load_best_model_at_end:
            training_args_dict["evaluation_strategy"] = save_strategy
        else:
            # Set evaluation strategy based on whether we have a validation dataset
            training_args_dict["evaluation_strategy"] = save_strategy if val_dataset is not None else "no"
        
        # Create training arguments
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize HuggingFace Trainer
        self.trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
        )
        
        # Run training
        train_result = self.trainer.train()
        
        # Update model reference
        self.model = model_to_train
        
        return train_result

    def push_to_hub(self):
        """Push model to HuggingFace Hub"""
        if not self.hub_model_id or not self.hub_token:
            raise ValueError("hub_model_id and hub_token must be provided to push to hub")
            
        self.trainer.push_to_hub()

    def save_results(self, results: Dict[str, Any], output_dir: str = "./results") -> str:
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results with max sequence length included
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
            "max_sequence_length": self.max_seq_length,  # Track max sequence length
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast
                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    # Keep as-is if conversion fails
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        self.model = self.model.from_pretrained(checkpoint_path)
        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_path)

    def setup(self):
        """Setup trainer with LoRA if configured"""
        if self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config)
            
    def prepare_model_for_training(self):
        """Prepare model for training"""
        from peft import get_peft_model
        if self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config)
        self.model.train()
