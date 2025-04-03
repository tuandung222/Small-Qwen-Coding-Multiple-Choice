import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for model benchmarking
    """

    batch_size: int = 32
    num_samples: int = 1000
    max_length: int = 100
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    device: str = "cuda"
    precision: str = "fp16"
    seed: int = 42
    num_warmup: int = 10
    num_iterations: int = 100


def benchmark_generation_speed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """
    Benchmark model generation speed

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer to use
        config: Benchmark configuration

    Returns:
        Dict[str, float]: Speed metrics
    """
    try:
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Set device
        device = torch.device(config.device)
        model = model.to(device)

        # Generate test prompts
        test_prompts = [
            "Write a function to sort a list in Python.",
            "Explain how neural networks work.",
            "Write a SQL query to join two tables.",
            "What is the difference between Python and JavaScript?",
            "How does garbage collection work in Java?",
        ]

        # Speed metrics
        metrics = {
            "generation_time": [],
            "tokens_per_second": [],
            "throughput": [],
        }

        # Warmup
        logger.info("Warming up...")
        for _ in range(config.num_warmup):
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )

        # Benchmark
        logger.info("Running benchmark...")
        for _ in range(config.num_iterations):
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )
                end_time.record()

                # Wait for GPU to finish
                torch.cuda.synchronize()

                # Record metrics
                generation_time = start_time.elapsed_time(end_time)
                metrics["generation_time"].append(generation_time)

                # Compute tokens per second
                num_tokens = len(outputs[0])
                tokens_per_second = num_tokens / (generation_time / 1000)  # Convert to seconds
                metrics["tokens_per_second"].append(tokens_per_second)

                # Compute throughput
                throughput = 1 / (generation_time / 1000)  # Generations per second
                metrics["throughput"].append(throughput)

        # Compute average metrics
        metrics = {
            "avg_generation_time": np.mean(metrics["generation_time"]),
            "avg_tokens_per_second": np.mean(metrics["tokens_per_second"]),
            "avg_throughput": np.mean(metrics["throughput"]),
            "std_generation_time": np.std(metrics["generation_time"]),
            "std_tokens_per_second": np.std(metrics["tokens_per_second"]),
            "std_throughput": np.std(metrics["throughput"]),
        }

        logger.info("Completed generation speed benchmarking")
        return metrics

    except Exception as e:
        logger.error(f"Error benchmarking generation speed: {str(e)}")
        raise


def benchmark_inference_speed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataloader: torch.utils.data.DataLoader,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """
    Benchmark model inference speed

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer to use
        test_dataloader: Test data loader
        config: Benchmark configuration

    Returns:
        Dict[str, float]: Speed metrics
    """
    try:
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Set device
        device = torch.device(config.device)
        model = model.to(device)

        # Speed metrics
        metrics = {
            "inference_time": [],
            "samples_per_second": [],
            "throughput": [],
        }

        # Warmup
        logger.info("Warming up...")
        for _ in range(config.num_warmup):
            batch = next(iter(test_dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                _ = model(**batch)

        # Benchmark
        logger.info("Running benchmark...")
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                _ = model(**batch)
                end_time.record()

                # Wait for GPU to finish
                torch.cuda.synchronize()

                # Record metrics
                inference_time = start_time.elapsed_time(end_time)
                metrics["inference_time"].append(inference_time)

                # Compute samples per second
                batch_size = batch["input_ids"].size(0)
                samples_per_second = batch_size / (inference_time / 1000)  # Convert to seconds
                metrics["samples_per_second"].append(samples_per_second)

                # Compute throughput
                throughput = batch_size / (inference_time / 1000)  # Samples per second
                metrics["throughput"].append(throughput)

        # Compute average metrics
        metrics = {
            "avg_inference_time": np.mean(metrics["inference_time"]),
            "avg_samples_per_second": np.mean(metrics["samples_per_second"]),
            "avg_throughput": np.mean(metrics["throughput"]),
            "std_inference_time": np.std(metrics["inference_time"]),
            "std_samples_per_second": np.std(metrics["samples_per_second"]),
            "std_throughput": np.std(metrics["throughput"]),
        }

        logger.info("Completed inference speed benchmarking")
        return metrics

    except Exception as e:
        logger.error(f"Error benchmarking inference speed: {str(e)}")
        raise


def benchmark_memory_usage(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """
    Benchmark model memory usage

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer to use
        config: Benchmark configuration

    Returns:
        Dict[str, float]: Memory metrics
    """
    try:
        # Set device
        device = torch.device(config.device)
        model = model.to(device)

        # Memory metrics
        metrics = {
            "model_size": 0,
            "peak_memory": [],
            "allocated_memory": [],
            "cached_memory": [],
        }

        # Measure model size
        for param in model.parameters():
            metrics["model_size"] += param.nelement() * param.element_size()

        # Generate test prompts
        test_prompts = [
            "Write a function to sort a list in Python.",
            "Explain how neural networks work.",
            "Write a SQL query to join two tables.",
            "What is the difference between Python and JavaScript?",
            "How does garbage collection work in Java?",
        ]

        # Warmup
        logger.info("Warming up...")
        for _ in range(config.num_warmup):
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )

        # Benchmark
        logger.info("Running benchmark...")
        for _ in range(config.num_iterations):
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Generate
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )

            # Record memory stats
            metrics["peak_memory"].append(torch.cuda.max_memory_allocated())
            metrics["allocated_memory"].append(torch.cuda.memory_allocated())
            metrics["cached_memory"].append(torch.cuda.memory_reserved())

        # Convert to MB and compute statistics
        metrics = {
            "model_size_mb": metrics["model_size"] / (1024 * 1024),
            "avg_peak_memory_mb": np.mean(metrics["peak_memory"]) / (1024 * 1024),
            "avg_allocated_memory_mb": np.mean(metrics["allocated_memory"]) / (1024 * 1024),
            "avg_cached_memory_mb": np.mean(metrics["cached_memory"]) / (1024 * 1024),
            "std_peak_memory_mb": np.std(metrics["peak_memory"]) / (1024 * 1024),
            "std_allocated_memory_mb": np.std(metrics["allocated_memory"]) / (1024 * 1024),
            "std_cached_memory_mb": np.std(metrics["cached_memory"]) / (1024 * 1024),
        }

        logger.info("Completed memory usage benchmarking")
        return metrics

    except Exception as e:
        logger.error(f"Error benchmarking memory usage: {str(e)}")
        raise


def benchmark_gpu_utilization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """
    Benchmark GPU utilization

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer to use
        config: Benchmark configuration

    Returns:
        Dict[str, float]: GPU metrics
    """
    try:
        # Set device
        device = torch.device(config.device)
        model = model.to(device)

        # GPU metrics
        metrics = {
            "gpu_utilization": [],
            "gpu_memory_utilization": [],
            "gpu_power_usage": [],
            "gpu_temperature": [],
        }

        # Generate test prompts
        test_prompts = [
            "Write a function to sort a list in Python.",
            "Explain how neural networks work.",
            "Write a SQL query to join two tables.",
            "What is the difference between Python and JavaScript?",
            "How does garbage collection work in Java?",
        ]

        # Warmup
        logger.info("Warming up...")
        for _ in range(config.num_warmup):
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )

        # Benchmark
        logger.info("Running benchmark...")
        for _ in range(config.num_iterations):
            # Generate
            prompt = np.random.choice(test_prompts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=True,
                )

            # Record GPU stats
            metrics["gpu_utilization"].append(torch.cuda.utilization())
            metrics["gpu_memory_utilization"].append(torch.cuda.memory_utilization())
            metrics["gpu_power_usage"].append(torch.cuda.power_usage())
            metrics["gpu_temperature"].append(torch.cuda.temperature())

        # Compute average metrics
        metrics = {
            "avg_gpu_utilization": np.mean(metrics["gpu_utilization"]),
            "avg_gpu_memory_utilization": np.mean(metrics["gpu_memory_utilization"]),
            "avg_gpu_power_usage": np.mean(metrics["gpu_power_usage"]),
            "avg_gpu_temperature": np.mean(metrics["gpu_temperature"]),
            "std_gpu_utilization": np.std(metrics["gpu_utilization"]),
            "std_gpu_memory_utilization": np.std(metrics["gpu_memory_utilization"]),
            "std_gpu_power_usage": np.std(metrics["gpu_power_usage"]),
            "std_gpu_temperature": np.std(metrics["gpu_temperature"]),
        }

        logger.info("Completed GPU utilization benchmarking")
        return metrics

    except Exception as e:
        logger.error(f"Error benchmarking GPU utilization: {str(e)}")
        raise


def run_all_benchmarks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataloader: torch.utils.data.DataLoader,
    config: BenchmarkConfig,
) -> Dict[str, Dict[str, float]]:
    """
    Run all model benchmarks

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer to use
        test_dataloader: Test data loader
        config: Benchmark configuration

    Returns:
        Dict[str, Dict[str, float]]: All benchmark metrics
    """
    try:
        # Run benchmarks
        generation_metrics = benchmark_generation_speed(model, tokenizer, config)
        inference_metrics = benchmark_inference_speed(model, tokenizer, test_dataloader, config)
        memory_metrics = benchmark_memory_usage(model, tokenizer, config)
        gpu_metrics = benchmark_gpu_utilization(model, tokenizer, config)

        # Combine metrics
        all_metrics = {
            "generation": generation_metrics,
            "inference": inference_metrics,
            "memory": memory_metrics,
            "gpu": gpu_metrics,
        }

        logger.info("Completed all benchmarks")
        return all_metrics

    except Exception as e:
        logger.error(f"Error running all benchmarks: {str(e)}")
        raise
