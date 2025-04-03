#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unsloth optimizer for Qwen models.
This module provides utilities for optimizing Qwen models with Unsloth for faster inference.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError("Unsloth is not installed. Please install it with `pip install unsloth`.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnslothOptimizer:
    """
    Unsloth optimizer for Qwen models.

    This class provides utilities for optimizing Qwen models with Unsloth for faster inference.
    It uses FastLanguageModel.from_pretrained to load the model and FastLanguageModel.for_inference
    to optimize it for inference.

    Args:
        model_name_or_path (str): The name or path of the model to load.
        adapter_path (Optional[str], optional): The path to the LoRA adapter. Defaults to None.
        max_seq_length (int, optional): The maximum sequence length. Defaults to 4096.
        dtype (torch.dtype, optional): The dtype to use. Defaults to torch.bfloat16.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        device_map (str, optional): The device map to use. Defaults to "auto".
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        max_seq_length: int = 4096,
        dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ):
        """Initialize the Unsloth optimizer."""
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map

        logger.info(f"Loading model {model_name_or_path} with Unsloth...")
        start_time = time.time()

        # Load the model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Optimize the model for inference
        self.model = self._optimize_for_inference()

        logger.info(f"Model loaded and optimized in {time.time() - start_time:.2f} seconds.")

    def _load_model(self) -> Tuple[Any, Any]:
        """Load the model and tokenizer using Unsloth."""
        # Load the model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name_or_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            device_map=self.device_map,
        )

        # Load the adapter if provided
        if self.adapter_path:
            logger.info(f"Loading adapter from {self.adapter_path}")
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,  # LoRA attention dimension
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=42,
                use_rslora=False,  # We use normal LoRA
                loftq_config=None,
            )
            # Load the adapter weights
            model.load_adapter(self.adapter_path)

        return model, tokenizer

    def _optimize_for_inference(self) -> Any:
        """Optimize the model for inference using Unsloth."""
        model = FastLanguageModel.for_inference(self.model)
        return model

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from prompts.

        Args:
            prompts (Union[str, List[str]]): The prompt or list of prompts.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 256.
            temperature (float, optional): The temperature for sampling. Defaults to 0.7.
            top_p (float, optional): The top-p for nucleus sampling. Defaults to 0.9.
            top_k (int, optional): The top-k for top-k sampling. Defaults to 50.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            **kwargs: Additional arguments to pass to the model's generate method.

        Returns:
            List[str]: The generated texts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

            # Decode the outputs
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(decoded_output)

        return results

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Create a chat completion from messages.

        Args:
            messages (List[Dict[str, str]]): List of messages with "role" and "content" keys.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 256.
            temperature (float, optional): The temperature for sampling. Defaults to 0.7.
            top_p (float, optional): The top-p for nucleus sampling. Defaults to 0.9.
            top_k (int, optional): The top-k for top-k sampling. Defaults to 50.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            **kwargs: Additional arguments to pass to the model's generate method.

        Returns:
            str: The generated response.
        """
        # Format the messages for chat
        chat_input = self._format_chat_messages(messages)

        # Generate the response
        response = self.generate(
            chat_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )[0]

        # Extract the assistant's response
        assistant_response = self._extract_assistant_response(response, chat_input)

        return assistant_response

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages for the model.

        Args:
            messages (List[Dict[str, str]]): List of messages with "role" and "content" keys.

        Returns:
            str: Formatted chat input for the model.
        """
        # Qwen chat format may vary, this is a common format
        chat_input = ""

        # For Qwen-based models, we use their specific chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Use the tokenizer's built-in chat template
            chat_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback to a generic template
            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    chat_input += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    chat_input += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    chat_input += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            # Add the final assistant prompt
            chat_input += "<|im_start|>assistant\n"

        return chat_input

    def _extract_assistant_response(self, full_response: str, chat_input: str) -> str:
        """
        Extract the assistant's response from the full response.

        Args:
            full_response (str): The full response from the model.
            chat_input (str): The chat input provided to the model.

        Returns:
            str: The assistant's response.
        """
        # Remove the chat input from the beginning of the response
        if full_response.startswith(chat_input):
            assistant_response = full_response[len(chat_input) :]
        else:
            assistant_response = full_response

        # If the response contains end markers, extract only the assistant part
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]

        return assistant_response.strip()

    def benchmark(
        self,
        prompt: str = "Write a Python function to calculate Fibonacci numbers.",
        max_new_tokens: int = 100,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """
        Benchmark the model's inference speed.

        Args:
            prompt (str, optional): The prompt to use for benchmarking.
                Defaults to "Write a Python function to calculate Fibonacci numbers.".
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
            num_runs (int, optional): The number of runs to average over. Defaults to 5.

        Returns:
            Dict[str, float]: A dictionary with the benchmark results:
                - avg_latency: Average latency in seconds
                - avg_tokens_per_second: Average tokens per second
        """
        total_time = 0
        total_tokens = 0

        logger.info(f"Running benchmark with {num_runs} runs...")

        for i in range(num_runs):
            start_time = time.time()

            # Generate text
            outputs = self.generate(prompt, max_new_tokens=max_new_tokens)[0]

            # Calculate time and tokens
            run_time = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(outputs)) - len(
                self.tokenizer.encode(prompt)
            )

            logger.info(
                f"Run {i+1}: {run_time:.2f}s, {tokens_generated} tokens, "
                f"{tokens_generated/run_time:.2f} tokens/s"
            )

            total_time += run_time
            total_tokens += tokens_generated

        # Calculate averages
        avg_latency = total_time / num_runs
        avg_tokens_per_second = total_tokens / total_time

        logger.info(
            f"Benchmark results: {avg_latency:.2f}s avg latency, "
            f"{avg_tokens_per_second:.2f} tokens/s"
        )

        return {
            "avg_latency": avg_latency,
            "avg_tokens_per_second": avg_tokens_per_second,
        }


# Example usage
if __name__ == "__main__":
    # Load and optimize a model
    optimizer = UnslothOptimizer(
        model_name_or_path="unsloth/Qwen2.5-Coder-1.5B-Instruct",
        max_seq_length=4096,
        dtype=torch.bfloat16,
    )

    # Generate text
    prompt = "Write a Python function to calculate Fibonacci numbers."
    output = optimizer.generate(prompt, max_new_tokens=256)[0]
    print(f"Generated text: {output}")

    # Chat completion
    messages = [
        {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
    ]
    response = optimizer.create_chat_completion(messages)
    print(f"Chat response: {response}")

    # Benchmark
    results = optimizer.benchmark(prompt, max_new_tokens=100, num_runs=3)
    print(f"Average latency: {results['avg_latency']:.2f}s")
    print(f"Average tokens per second: {results['avg_tokens_per_second']:.2f}")
