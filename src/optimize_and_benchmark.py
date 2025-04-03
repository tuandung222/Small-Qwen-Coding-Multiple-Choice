#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimization and benchmarking script for Qwen models.

This script combines various optimization techniques for Qwen models and
provides a CLI for benchmarking and comparing different optimization strategies.

Example usage:
    
    # Benchmark different optimization techniques
    python optimize_and_benchmark.py --model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" --benchmark all
    
    # Run optimized inference with a prompt
    python optimize_and_benchmark.py --model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" --prompt "Write a Python function to calculate Fibonacci numbers." --optimization torch_inference_mode
    
    # Run with LoRA adapter
    python optimize_and_benchmark.py --model_path "unsloth/Qwen2.5-Coder-1.5B-Instruct" --adapter_path "./adapter" --prompt "Write a Python function to calculate Fibonacci numbers."
"""

import os
import time
import json
import torch
import argparse
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import local optimization modules
try:
    from src.model.optimized_inference import OptimizedInference
    from src.model.unsloth_optimizer import UnslothOptimizer
    from src.model.model_compiler import ModelCompiler
except ImportError:
    # When running from the script's directory
    try:
        from model.optimized_inference import OptimizedInference
        from model.unsloth_optimizer import UnslothOptimizer
        from model.model_compiler import ModelCompiler
    except ImportError:
        # Fall back to assuming we're in the repo root
        import sys
        sys.path.append(".")
        from src.model.optimized_inference import OptimizedInference
        from src.model.unsloth_optimizer import UnslothOptimizer
        from src.model.model_compiler import ModelCompiler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize and benchmark Qwen models")
    
    # Model and data arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="unsloth/Qwen2.5-Coder-1.5B-Instruct",
        help="Path or name of the pre-trained model",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to calculate Fibonacci numbers.",
        help="Prompt for generation or benchmarking",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a file containing prompts (one per line)",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--optimization",
        type=str,
        default="all",
        choices=["none", "torch_inference_mode", "torch_compile", "unsloth", "all"],
        help="Optimization technique to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "4bit", "8bit"],
        help="Precision to use for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Mode for torch.compile",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="Backend for torch.compile",
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k for top-k sampling",
    )
    
    # Benchmarking arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=[None, "single", "all", "comparative"],
        help="Whether to run benchmarks",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs for benchmarking",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save benchmark results",
    )
    parser.add_argument(
        "--save_optimized_model",
        type=str,
        default=None,
        help="Path to save the optimized model",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    precision: str = "bf16",
) -> tuple:
    """
    Load model and tokenizer with the specified precision.
    
    Args:
        model_path: Path or name of the pre-trained model
        adapter_path: Path to the LoRA adapter
        precision: Precision to use (fp32, fp16, bf16, 4bit, 8bit)
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path} with precision {precision}")
    
    # Configure quantization if needed
    quantization_config = None
    torch_dtype = None
    
    if precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif precision == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as e:
            logger.error(f"Failed to load adapter: {str(e)}")
            raise
    
    return model, tokenizer


def optimize_model(
    model: Any,
    tokenizer: Any,
    optimization: str,
    precision: str,
    batch_size: int,
    compile_mode: str,
    compile_backend: str,
) -> Any:
    """
    Apply the specified optimization technique to the model.
    
    Args:
        model: The model to optimize
        tokenizer: The tokenizer
        optimization: Optimization technique to use
        precision: Precision to use
        batch_size: Batch size for inference
        compile_mode: Mode for torch.compile
        compile_backend: Backend for torch.compile
        
    Returns:
        Any: Optimized model or inference object
    """
    logger.info(f"Applying optimization: {optimization}")
    
    if optimization == "none":
        return model
    
    if optimization == "torch_inference_mode":
        return OptimizedInference(
            model=model,
            tokenizer=tokenizer,
            precision=precision if precision in ["fp32", "fp16", "bf16"] else "fp32",
            batch_size=batch_size,
        )
    
    if optimization == "torch_compile":
        compiler = ModelCompiler(
            model=model,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            enable_sdpa=True,
        )
        compiler.optimize_attention()
        return compiler.get_model()
    
    if optimization == "unsloth":
        try:
            # Return the original model and tokenizer if unsloth is not properly installed
            from unsloth import FastLanguageModel
            
            # We need to reload the model with unsloth
            optimizer = UnslothOptimizer(
                model_name_or_path=model.config._name_or_path,
                max_seq_length=4096,
                dtype=torch.bfloat16 if precision == "bf16" else (
                    torch.float16 if precision == "fp16" else torch.float32
                ),
                load_in_4bit=True if precision == "4bit" else False,
            )
            return optimizer
        except ImportError:
            logger.warning("Unsloth is not installed. Falling back to torch_inference_mode.")
            return OptimizedInference(
                model=model,
                tokenizer=tokenizer,
                precision=precision if precision in ["fp32", "fp16", "bf16"] else "fp32",
                batch_size=batch_size,
            )
    
    if optimization == "all":
        # Apply all optimizations that make sense together
        try:
            # Try unsloth first, which is typically the most optimized approach
            from unsloth import FastLanguageModel
            
            # We need to reload the model with unsloth
            optimizer = UnslothOptimizer(
                model_name_or_path=model.config._name_or_path,
                max_seq_length=4096,
                dtype=torch.bfloat16 if precision == "bf16" else (
                    torch.float16 if precision == "fp16" else torch.float32
                ),
                load_in_4bit=True if precision == "4bit" else False,
            )
            return optimizer
        except ImportError:
            # Fall back to torch.compile + OptimizedInference
            logger.warning("Unsloth is not installed. Falling back to torch_compile + torch_inference_mode.")
            
            # First apply torch.compile
            compiler = ModelCompiler(
                model=model,
                compile_mode=compile_mode,
                compile_backend=compile_backend,
                enable_sdpa=True,
            )
            compiler.optimize_attention()
            compiled_model = compiler.get_model()
            
            # Then wrap with OptimizedInference
            return OptimizedInference(
                model=compiled_model,
                tokenizer=tokenizer,
                precision=precision if precision in ["fp32", "fp16", "bf16"] else "fp32",
                batch_size=batch_size,
            )
    
    # Default fallback
    return model


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    optimization: str,
) -> str:
    """
    Generate text using the optimized model.
    
    Args:
        model: The optimized model or inference object
        tokenizer: The tokenizer
        prompt: The prompt for generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        top_k: Top-k for top-k sampling
        optimization: The optimization technique being used
        
    Returns:
        str: Generated text
    """
    logger.info(f"Generating text for prompt: {prompt[:50]}...")
    
    # Different handling based on optimization type
    if optimization in ["torch_inference_mode", "all"]:
        if hasattr(model, "generate") and hasattr(model, "tokenizer"):
            # OptimizedInference or UnslothOptimizer
            output = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            return output[0] if isinstance(output, list) else output
    
    if optimization in ["none", "torch_compile"]:
        # Standard model or torch.compile model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if optimization == "unsloth":
        # Unsloth optimizer already has a generate method
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return output[0] if isinstance(output, list) else output
    
    # Fallback for unexpected cases
    raise ValueError(f"Unsupported optimization technique: {optimization}")


def benchmark_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_runs: int,
    optimization: str,
) -> Dict[str, float]:
    """
    Benchmark the model's inference speed.
    
    Args:
        model: The optimized model or inference object
        tokenizer: The tokenizer
        prompt: The prompt for benchmarking
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        top_k: Top-k for top-k sampling
        num_runs: Number of runs for benchmarking
        optimization: The optimization technique being used
        
    Returns:
        Dict[str, float]: Benchmark results
    """
    logger.info(f"Benchmarking with {num_runs} runs...")
    
    # Different handling based on optimization type
    if optimization in ["torch_inference_mode", "all"]:
        if hasattr(model, "benchmark"):
            # Use built-in benchmark if available
            results = model.benchmark(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
            )
            return results
    
    if optimization == "unsloth":
        if hasattr(model, "benchmark"):
            # Use built-in benchmark if available
            results = model.benchmark(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
            )
            return results
    
    # Manual benchmarking for other cases
    total_time = 0
    total_tokens = 0
    
    # Warm-up run
    _ = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        optimization=optimization,
    )
    
    # Actual benchmark runs
    for i in range(num_runs):
        start_time = time.time()
        
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            optimization=optimization,
        )
        
        end_time = time.time()
        run_time = end_time - start_time
        
        # Calculate tokens based on the optimization type
        if optimization in ["none", "torch_compile"]:
            prompt_tokens = len(tokenizer.encode(prompt))
            output_tokens = len(tokenizer.encode(output))
            tokens_generated = output_tokens - prompt_tokens
        else:
            # Estimate tokens for other optimization types
            tokens_generated = max_new_tokens
        
        logger.info(f"Run {i+1}: {run_time:.2f}s, ~{tokens_generated} tokens, "
                   f"{tokens_generated/run_time:.2f} tokens/s")
        
        total_time += run_time
        total_tokens += tokens_generated
    
    avg_latency = total_time / num_runs
    avg_tokens_per_second = total_tokens / total_time
    
    logger.info(f"Benchmark results: {avg_latency:.2f}s avg latency, "
               f"{avg_tokens_per_second:.2f} tokens/s")
    
    return {
        "total_time": total_time,
        "avg_latency": avg_latency,
        "avg_tokens_per_second": avg_tokens_per_second,
        "num_runs": num_runs,
    }


def run_comparative_benchmark(
    model_path: str,
    adapter_path: Optional[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    precision: str,
    batch_size: int,
    compile_mode: str,
    compile_backend: str,
    num_runs: int,
) -> Dict[str, Dict[str, float]]:
    """
    Run benchmark for different optimization techniques and compare results.
    
    Args:
        model_path: Path or name of the pre-trained model
        adapter_path: Path to the LoRA adapter
        prompt: The prompt for benchmarking
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        top_k: Top-k for top-k sampling
        precision: Precision to use
        batch_size: Batch size for inference
        compile_mode: Mode for torch.compile
        compile_backend: Backend for torch.compile
        num_runs: Number of runs for benchmarking
        
    Returns:
        Dict[str, Dict[str, float]]: Benchmark results for each optimization technique
    """
    # Optimization techniques to compare
    optimizations = ["none", "torch_inference_mode", "torch_compile"]
    
    # Try to add unsloth if available
    try:
        from unsloth import FastLanguageModel
        optimizations.append("unsloth")
    except ImportError:
        logger.warning("Unsloth is not installed. Skipping unsloth benchmark.")
    
    results = {}
    
    for opt in optimizations:
        logger.info(f"Benchmarking optimization: {opt}")
        
        # Load model fresh each time to ensure fair comparison
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            adapter_path=adapter_path,
            precision=precision,
        )
        
        # Apply optimization
        optimized_model = optimize_model(
            model=model,
            tokenizer=tokenizer,
            optimization=opt,
            precision=precision,
            batch_size=batch_size,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )
        
        # Run benchmark
        benchmark_results = benchmark_model(
            model=optimized_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_runs=num_runs,
            optimization=opt,
        )
        
        results[opt] = benchmark_results
    
    # Compare results
    logger.info("Comparative benchmark results:")
    for opt, res in results.items():
        logger.info(f"{opt}: {res['avg_latency']:.2f}s avg latency, "
                   f"{res['avg_tokens_per_second']:.2f} tokens/s")
    
    # Identify the fastest
    fastest_opt = min(results.items(), key=lambda x: x[1]['avg_latency'])[0]
    logger.info(f"Fastest optimization: {fastest_opt}")
    
    return results


def save_benchmark_results(results: Dict, output_file: str):
    """Save benchmark results to a file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_file}")


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a file, one per line."""
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load prompts
    prompts = [args.prompt]
    if args.prompt_file:
        prompts = load_prompts_from_file(args.prompt_file)
    
    # Handle benchmarking
    if args.benchmark == "comparative":
        results = run_comparative_benchmark(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            prompt=prompts[0],  # Use the first prompt for comparative benchmark
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            precision=args.precision,
            batch_size=args.batch_size,
            compile_mode=args.compile_mode,
            compile_backend=args.compile_backend,
            num_runs=args.num_runs,
        )
        
        if args.output_file:
            save_benchmark_results(results, args.output_file)
        
        return
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        precision=args.precision,
    )
    
    # Apply optimization
    optimized_model = optimize_model(
        model=model,
        tokenizer=tokenizer,
        optimization=args.optimization,
        precision=args.precision,
        batch_size=args.batch_size,
        compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
    )
    
    # Run benchmark if requested
    if args.benchmark in ["single", "all"]:
        benchmark_results = benchmark_model(
            model=optimized_model,
            tokenizer=tokenizer,
            prompt=prompts[0],  # Use the first prompt for benchmarking
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_runs=args.num_runs,
            optimization=args.optimization,
        )
        
        if args.output_file:
            save_benchmark_results(benchmark_results, args.output_file)
    
    # Generate text for each prompt
    if args.benchmark != "all":  # Skip generation for benchmark-only mode
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating for prompt {i+1}/{len(prompts)}")
            
            output = generate_text(
                model=optimized_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                optimization=args.optimization,
            )
            
            print(f"\nPrompt: {prompt}\n")
            print(f"Generated: {output}\n")
            print("-" * 80)
    
    # Save optimized model if requested
    if args.save_optimized_model and hasattr(optimized_model, "save_pretrained"):
        optimized_model.save_pretrained(args.save_optimized_model)
        logger.info(f"Optimized model saved to {args.save_optimized_model}")


if __name__ == "__main__":
    main() 