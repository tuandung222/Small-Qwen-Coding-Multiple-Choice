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

class QwenTrainer:
    """Training handler for Qwen models with optional HuggingFace Hub integration"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_creator: Optional[PromptCreator] = None,
        lora_config: Optional[Any] = None,
        hub_token: Optional[str] = None,
        hub_model_id: Optional[str] = None,
    ):
        """
        Initialize the trainer with model, tokenizer and optional LoRA config
        
        Args:
            model: The model to fine-tune
            tokenizer: The tokenizer for the model
            prompt_creator: Optional PromptCreator for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient fine-tuning
            hub_token: Optional HuggingFace Hub token for pushing models
            hub_model_id: Optional model ID for pushing to HuggingFace Hub
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_creator = prompt_creator or PromptCreator(PromptCreator.BASIC)
        self.lora_config = lora_config
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        self.peft_model = None
        
        # Ensure we have a proper max sequence length
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
        """Apply Unsloth's LoRA configuration instead of PEFT"""
        if self.lora_config:
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
        dataset: Any,
        prompt_type: Optional[str] = None,
        output_dir: str = "./model_output",
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        num_train_epochs: int = 3,
        max_steps: Optional[int] = None,
        learning_rate: float = 2e-4,
        train_on_inputs: bool = True,
        packing: bool = False,
        logging_steps: int = 10,
        save_steps: int = 100,
        verbose: bool = True,
        push_to_hub: bool = False,
    ) -> Dict[str, Any]:
        """Train the model using Unsloth's optimized training"""
        # Prepare dataset with on-the-fly transformation  
        prepared_dataset = self.prepare_dataset(dataset, prompt_type, verbose)
        
        # Import Unsloth's optimized trainer and utilities
        from unsloth import is_bfloat16_supported
        
        # Prepare the model with Unsloth's LoRA
        model_to_train = self.prepare_model_for_training()
        
        # Setup training arguments with proper handling of max_steps
        training_args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "optim": "paged_adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
            "push_to_hub": push_to_hub,
            "hub_model_id": self.hub_model_id if push_to_hub else None,
            "hub_token": self.hub_token if push_to_hub else None,
        }
        
        # Only add max_steps if it's not None
        if max_steps is not None and max_steps > 0:
            training_args_dict["max_steps"] = max_steps
        
        # Create TrainingArguments with the prepared dictionary
        training_args = TrainingArguments(**training_args_dict)
        
        # Use SFTTrainer with Unsloth-specific settings
        trainer = SFTTrainer(
            model=model_to_train,
            tokenizer=self.tokenizer,
            train_dataset=prepared_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,  # Use our tracked max sequence length
            args=training_args,
            packing=False,  # Set packing to False to avoid shape issues
        )
        
        if verbose:
            # Log dataset and training info
            train_size = len(prepared_dataset)
            steps_per_epoch = train_size // (per_device_train_batch_size * gradient_accumulation_steps)
            total_steps = steps_per_epoch * num_train_epochs if max_steps is None else max_steps
            
            print(f"Training with dataset size: {train_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total training steps: {total_steps}")
            print(f"Max sequence length: {self.max_seq_length}")
            print(f"Push to HuggingFace Hub: {push_to_hub}")
            if push_to_hub:
                print(f"Hub model ID: {self.hub_model_id}")
        
        # Start training
        trainer_stats = trainer.train()
        
        # Update self.model to the fine-tuned version
        self.model = model_to_train
        
        return trainer_stats
        
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

    def prepare_model_for_training(self):
        """Prepare model for training"""
        from peft import get_peft_model
        if self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config)
        self.model.train()
