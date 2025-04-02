from typing import Optional, Union, Dict, Any
import torch
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import threading
import os
from huggingface_hub import HfApi

class QwenModelHandler:
    """Handler for Qwen models with inference and saving capabilities using Unsloth"""
    
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-7B",
        max_seq_length: int = 768,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize model and tokenizer using Unsloth
        
        Args:
            model_name: Name or path of the model (preferably an unsloth model)
            max_seq_length: Maximum sequence length for the model
            quantization: Quantization type (None, '4bit', '8bit') - for compatibility
            device_map: Device mapping strategy
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self.quantization = quantization
        self.cache_dir = cache_dir
        
        # Convert quantization parameter to load_in_4bit parameter for Unsloth
        self.load_in_4bit = quantization == "4bit"
        
        # Load tokenizer and model
        self.tokenizer, self.model = self._load_model()
        
    def _load_model(self) -> tuple:
        """Load model and tokenizer with Unsloth for optimization"""
        print(f"Loading {self.model_name} with Unsloth, max_seq_length={self.max_seq_length}")
        
        # Set dtype based on hardware
        dtype = None  # None for auto detection
        
        # Load model and tokenizer with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir=self.cache_dir,
        )
        
        return tokenizer, model
      
    def generate_with_streaming(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = True
    ) -> Union[str, TextIteratorStreamer]:
        """
        Generate completion with optional streaming using Unsloth's optimized inference
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the output
            
        Returns:
            Either the generated text or a streamer object
        """
        # Enable faster inference
        FastLanguageModel.for_inference(self.model)
        
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)
        
        # Generate with streaming if requested
        if stream:
            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Start generation in a thread
            generation_kwargs = {
                "input_ids": model_inputs.input_ids,
                "attention_mask": model_inputs.attention_mask,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
                "do_sample": temperature > 0.0,
                "use_cache": True,  # Important for Unsloth performance
                "min_p": 0.1 if temperature > 0.0 else None, # Optional: Unsloth recommends this for better quality
            }
            
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return the streamer that yields text chunks
            return streamer
        else:
            # Generate without streaming
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                use_cache=True,  # Important for Unsloth performance
                min_p=0.1 if temperature > 0.0 else None, # Optional: Unsloth recommends this
            )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0][model_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text
            
    def calculate_perplexity(self, prompt: str, answer: str, temperature: float = 0.0) -> float:
        """
        Calculate perplexity for a prompt and answer pair
        
        Args:
            prompt: The input prompt
            answer: The expected answer
            temperature: Sampling temperature
            
        Returns:
            Perplexity score
        """
        # Format chat for perplexity calculation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False
        )
        
        # Tokenize the text
        encodings = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)
            
        # Get loss and calculate perplexity
        neg_log_likelihood = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
        
        return perplexity
  
    def save_model(self, output_dir: str, save_method: str = "lora") -> str:
        """
        Save model to disk using Unsloth's optimized methods
        
        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_16bit", "merged_4bit", "gguf")
            
        Returns:
            Path to saved model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Use Unsloth's saving methods
        if save_method == "lora":
            # Save LoRA weights
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            # Save merged model in float16
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_16bit")
        elif save_method == "merged_4bit":
            # Save merged model in 4bit
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_4bit")
        elif save_method == "gguf":
            # Save in GGUF format for llama.cpp
            self.model.save_pretrained_gguf(output_dir, self.tokenizer, quantization_method="q4_k_m")
        else:
            raise ValueError(f"Unknown save method: {save_method}")
            
        print(f"Model saved to {output_dir} using method {save_method}")
        return output_dir
        
    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        save_method: str = "lora",
        private: bool = False
    ) -> str:
        """
        Push model to Hugging Face Hub using Unsloth's optimized methods
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            token: Optional Hugging Face token
            save_method: Method to use for saving
            private: Whether to make the repository private
            
        Returns:
            URL of the pushed model
        """
        # Use Unsloth's hub methods directly
        if save_method == "lora":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="lora", token=token)
        elif save_method == "merged_16bit":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="merged_16bit", token=token)
        elif save_method == "merged_4bit":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="merged_4bit", token=token)
        elif save_method == "gguf":
            # Push multiple GGUF variants
            self.model.push_to_hub_gguf(
                repo_id, 
                self.tokenizer, 
                quantization_method=["q4_k_m", "q5_k_m"], 
                token=token
            )
        else:
            raise ValueError(f"Unknown save method: {save_method}")
        
        hub_url = f"https://huggingface.co/{repo_id}"
        print(f"Model successfully pushed to: {hub_url}")
        return hub_url
