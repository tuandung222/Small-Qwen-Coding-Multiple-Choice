"""
Inference demo script for the Qwen model.

This script demonstrates how to use the optimized inference module for
faster inference with the Qwen model.

Usage:
    python src/inference_demo.py --model_path "path/to/model" --prompt "Your prompt here"
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import our optimized inference module
from model.optimized_inference import OptimizedInference
from model.qwen_handler import QwenModelHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Inference demo for Qwen model")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/Qwen2.5-Coder-1.5B-Instruct",
        help="Path to model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
    )

    # Inference arguments
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "4bit", "8bit"],
        help="Precision for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )

    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to calculate Fibonacci numbers.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to file containing prompt",
    )

    # Benchmark arguments
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for benchmark",
    )

    return parser.parse_args()


def load_model(args):
    """Load model and tokenizer"""
    logger.info(f"Loading model from {args.model_path}")

    # Setup quantization configuration if needed
    quantization_config = None
    if args.precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.precision == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Use QwenModelHandler for easy model loading
    model_handler = QwenModelHandler(
        model_name=args.model_path,
        device_map="auto",
        quantization=args.precision if args.precision in ["4bit", "8bit"] else "none",
    )

    # Load adapter if provided
    if args.adapter_path:
        logger.info(f"Loading adapter from {args.adapter_path}")
        model_handler.load_adapter(args.adapter_path)

    return model_handler.model, model_handler.tokenizer


def chat_completion_demo(args, model, tokenizer):
    """Demonstrate chat completion with the model"""
    logger.info("Chat completion demo")

    # Create optimized inference
    optimizer = OptimizedInference(
        model=model,
        tokenizer=tokenizer,
        precision=args.precision
        if args.precision not in ["4bit", "8bit"]
        else "fp32",  # Already quantized
        batch_size=args.batch_size,
    )

    # Create chat messages
    messages = [{"role": "user", "content": args.prompt}]

    # Generate response
    logger.info(f"Generating response for prompt: {args.prompt}")
    start_time = time.time()
    response = optimizer.create_chat_completion(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    end_time = time.time()

    # Print results
    logger.info(f"Generated response in {end_time - start_time:.2f} seconds:")
    print("\n" + "-" * 80)
    print("USER: " + args.prompt)
    print("\nASSISTANT: " + response)
    print("-" * 80 + "\n")


def run_benchmark(args, model, tokenizer):
    """Run inference benchmark"""
    logger.info("Running inference benchmark")

    # Create optimized inference
    optimizer = OptimizedInference(
        model=model,
        tokenizer=tokenizer,
        precision=args.precision
        if args.precision not in ["4bit", "8bit"]
        else "fp32",  # Already quantized
        batch_size=args.batch_size,
    )

    # Run benchmark
    results = optimizer.benchmark(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
    )

    # Print results
    print("\n" + "=" * 50)
    print(" BENCHMARK RESULTS ")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Precision: {args.precision}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Average latency: {results['avg_latency']:.2f} seconds")
    print(f"Average tokens per second: {results['avg_tokens_per_second']:.2f}")
    print("=" * 50 + "\n")


def main():
    """Main function"""
    args = parse_args()

    # Load prompt from file if provided
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            args.prompt = f.read()

    # Load model and tokenizer
    model, tokenizer = load_model(args)

    # Run chat completion demo
    chat_completion_demo(args, model, tokenizer)

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(args, model, tokenizer)


if __name__ == "__main__":
    main()
