"""
Optimized inference module for Qwen models.
This module provides optimized inference utilities for faster model inference.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class OptimizedInference:
    """
    Optimized inference utilities for faster model inference.

    Features:
    - torch.inference_mode() for faster inference
    - Batched inference for better throughput
    - Half-precision inference (fp16/bf16)
    - Flash Attention for faster attention computation
    - KV cache optimization
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        precision: str = "bf16",  # Options: "fp32", "fp16", "bf16"
        batch_size: int = 4,
    ):
        """
        Initialize optimized inference utilities.

        Args:
            model: The model to optimize for inference
            tokenizer: The tokenizer for the model
            device: Device to use for inference (None for auto-detection)
            precision: Precision to use for inference (fp32, fp16, bf16)
            batch_size: Batch size for batched inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Convert model to desired precision
        self.precision = precision
        if self.precision == "fp16":
            self.model = self.model.half()
            logger.info("Model converted to fp16 precision")
        elif (
            self.precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            self.model = self.model.to(dtype=torch.bfloat16)
            logger.info("Model converted to bf16 precision")
        else:
            logger.info("Using fp32 precision")

        # Move model to device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")

        # Set model to eval mode
        self.model.eval()

        # Try to optimize with unsloth if available
        try:
            from unsloth import FastLanguageModel

            logger.info("Optimizing model with Unsloth...")
            self.model = FastLanguageModel.for_inference(self.model)
            logger.info("Model optimized with Unsloth")
        except ImportError:
            logger.info("Unsloth not available, using standard model")

        # Enable flash attention if available
        self._enable_flash_attention()

    def _enable_flash_attention(self):
        """Enable flash attention if available"""
        try:
            # Check if model config has flash attention attributes
            if hasattr(self.model.config, "use_flash_attention_2"):
                self.model.config.use_flash_attention_2 = True
                logger.info("Flash Attention 2 enabled")
            elif (
                hasattr(self.model.config, "attn_implementation")
                and "flash_attention" in self.model.config.attn_implementation
            ):
                logger.info(
                    f"Using attention implementation: {self.model.config.attn_implementation}"
                )
            else:
                logger.info("Flash Attention not available for this model")
        except Exception as e:
            logger.warning(f"Error enabling Flash Attention: {e}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate text with optimized inference.

        Args:
            prompts: One or more prompts to generate from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            List of generated texts
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        all_results = []

        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_results = self._generate_batch(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs,
            )
            all_results.extend(batch_results)

        return all_results

    def _generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for a batch of prompts"""
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate with inference mode for faster inference
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)

        # Decode outputs
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Return only the generated text (without the prompt)
        generated_texts = []
        for prompt, output in zip(prompts, decoded_outputs):
            # Remove the prompt from the output (may need adjustment for different models)
            if output.startswith(prompt):
                generated_text = output[len(prompt) :].strip()
            else:
                generated_text = output.strip()
            generated_texts.append(generated_text)

        return generated_texts

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Create a chat completion with optimized inference.

        Args:
            messages: Chat messages in the format [{role: "user", content: "Hello"}, ...]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        generated = self.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs
        )[0]

        return generated

    def benchmark(
        self, prompt: str, max_new_tokens: int = 100, num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            prompt: Prompt to use for benchmarking
            max_new_tokens: Number of tokens to generate
            num_runs: Number of runs for benchmarking

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Warmup
        logger.info("Warming up...")
        with torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.model.generate(**inputs, max_new_tokens=10)

        # Benchmark
        logger.info(f"Running benchmark with {num_runs} runs...")
        latencies = []
        tokens_per_second = []

        for _ in range(num_runs):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Time generation
            start_time = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            end_time = time.time()

            # Calculate metrics
            latency = end_time - start_time
            num_generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            tokens_per_sec = num_generated_tokens / latency

            latencies.append(latency)
            tokens_per_second.append(tokens_per_sec)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens_per_sec = sum(tokens_per_second) / len(tokens_per_second)

        return {
            "avg_latency": avg_latency,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "num_runs": num_runs,
            "max_new_tokens": max_new_tokens,
            "precision": self.precision,
            "device": self.device,
        }


# Example usage:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-Coder-1.5B-Instruct")

# Create optimized inference
optimizer = OptimizedInference(
    model=model,
    tokenizer=tokenizer,
    precision="bf16",
    batch_size=4
)

# Generate text
prompts = ["Write a Python function to calculate Fibonacci numbers."]
generated_texts = optimizer.generate(prompts, max_new_tokens=256)

# Chat completion
messages = [
    {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
]
response = optimizer.create_chat_completion(messages)

# Benchmark
benchmark_results = optimizer.benchmark(prompts[0])
print(f"Average latency: {benchmark_results['avg_latency']:.2f} seconds")
print(f"Average tokens per second: {benchmark_results['avg_tokens_per_second']:.2f}")
"""
