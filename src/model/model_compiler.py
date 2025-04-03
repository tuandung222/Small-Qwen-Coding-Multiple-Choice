#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Torch model compiler for Qwen models.
This module provides utilities for compiling Qwen models with torch.compile and other optimizations.
"""

import os
import torch
import time
import logging
from typing import Optional, Dict, Any, List, Union
from transformers import PreTrainedModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelCompiler:
    """
    Model compiler for Qwen models.
    
    This class provides utilities for compiling Qwen models with torch.compile and other optimizations.
    It supports various compilation backends and modes to optimize model performance.
    
    Args:
        model (PreTrainedModel): The model to compile.
        compile_mode (str, optional): The compile mode to use. Defaults to "reduce-overhead".
        compile_backend (str, optional): The backend to use. Defaults to "inductor".
        fullgraph (bool, optional): Whether to use fullgraph. Defaults to False.
        dynamic (bool, optional): Whether to use dynamic shapes. Defaults to False.
        use_cache (bool, optional): Whether to use the cache. Defaults to True.
        enable_vae_slicing (bool, optional): Enable VAE slicing (for diffusion models). Defaults to False.
        enable_xformers (bool, optional): Enable xFormers memory efficient attention. Defaults to False.
        enable_sdpa (bool, optional): Enable SDPA for attention computation. Defaults to True.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        compile_mode: str = "reduce-overhead",
        compile_backend: str = "inductor",
        fullgraph: bool = False,
        dynamic: bool = False,
        use_cache: bool = True,
        enable_vae_slicing: bool = False,
        enable_xformers: bool = False,
        enable_sdpa: bool = True,
    ):
        """Initialize the model compiler."""
        self.original_model = model
        self.compiled_model = None
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.use_cache = use_cache
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_xformers = enable_xformers
        self.enable_sdpa = enable_sdpa
        
        # Check if torch.compile is available
        if not hasattr(torch, "compile"):
            raise ImportError(
                "torch.compile is not available. Please upgrade to PyTorch 2.0 or later."
            )
        
        # Apply optimizations
        self._setup_optimizations()
    
    def _setup_optimizations(self) -> None:
        """Apply optimizations to the model."""
        logger.info("Setting up optimizations...")
        
        # Set CUDA graph capture mode if available
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        
        # Enable VAE slicing for diffusion models (if applicable)
        if self.enable_vae_slicing and hasattr(self.original_model, "enable_vae_slicing"):
            logger.info("Enabling VAE slicing")
            self.original_model.enable_vae_slicing()
        
        # Enable xFormers memory efficient attention (if applicable)
        if self.enable_xformers:
            try:
                if hasattr(self.original_model, "enable_xformers_memory_efficient_attention"):
                    logger.info("Enabling xFormers memory efficient attention")
                    self.original_model.enable_xformers_memory_efficient_attention()
            except ImportError:
                logger.warning("xFormers is not installed. Skipping xFormers optimization.")
        
        # Enable scaled dot product attention (SDPA)
        if self.enable_sdpa and torch.__version__ >= "2.0.0":
            os.environ["PYTORCH_ENABLE_SCALED_DOT_PRODUCT_ATTENTION"] = "1"
            logger.info("Enabling scaled dot product attention (SDPA)")
        
        # Compile the model
        self._compile_model()
    
    def _compile_model(self) -> None:
        """Compile the model using torch.compile."""
        logger.info(f"Compiling model with mode={self.compile_mode}, backend={self.compile_backend}...")
        start_time = time.time()
        
        try:
            # Process kwargs for torch.compile
            compile_kwargs = {
                "mode": self.compile_mode,
                "backend": self.compile_backend,
                "fullgraph": self.fullgraph,
                "dynamic": self.dynamic,
            }
            
            # Filter out None values
            compile_kwargs = {k: v for k, v in compile_kwargs.items() if v is not None}
            
            # Compile the model's forward function
            self.compiled_model = torch.compile(
                self.original_model,
                **compile_kwargs
            )
            
            logger.info(f"Model compiled in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Failed to compile model: {str(e)}")
            logger.warning("Falling back to the original model")
            self.compiled_model = self.original_model
    
    def get_model(self) -> PreTrainedModel:
        """Get the compiled model."""
        return self.compiled_model if self.compiled_model is not None else self.original_model
    
    @staticmethod
    def available_backends() -> List[str]:
        """Get the available compilation backends."""
        backends = []
        try:
            # For PyTorch 2.0+
            if hasattr(torch._dynamo, "list_backends"):
                backends = torch._dynamo.list_backends()
            # For PyTorch 2.1+
            elif hasattr(torch._inductor, "list_backends"):
                backends = torch._inductor.list_backends()
        except:
            backends = ["inductor", "aot_eager", "eager", "aot_ts_nvfuser"]
        
        return backends
    
    @staticmethod
    def available_modes() -> List[str]:
        """Get the available compilation modes."""
        return ["default", "reduce-overhead", "max-autotune"]
    
    def optimize_attention(self) -> None:
        """Apply additional attention optimizations."""
        if not hasattr(self.original_model, "config"):
            logger.warning("Model doesn't have a config attribute. Skipping attention optimizations.")
            return
        
        # Enable attention optimizations in the model config if available
        if hasattr(self.original_model.config, "attn_implementation"):
            logger.info("Setting attention implementation to 'flash_attention_2'")
            self.original_model.config.attn_implementation = "flash_attention_2"
        
        # For language models, ensure we use KV cache
        if hasattr(self.original_model.config, "use_cache"):
            self.original_model.config.use_cache = self.use_cache
            logger.info(f"Setting use_cache={self.use_cache} in model config")

    def optimize_memory_usage(self) -> None:
        """Apply memory usage optimizations."""
        # Enable gradient checkpointing if available and supported
        if hasattr(self.original_model, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing")
            try:
                self.original_model.gradient_checkpointing_enable()
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {str(e)}")
    
    def benchmark(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        num_runs: int = 5,
        **generate_kwargs
    ) -> Dict[str, float]:
        """
        Benchmark the model's inference speed.
        
        Args:
            input_ids (torch.Tensor): The input ids.
            attention_mask (Optional[torch.Tensor], optional): The attention mask. Defaults to None.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
            num_runs (int, optional): The number of runs to average over. Defaults to 5.
            **generate_kwargs: Additional arguments to pass to the model's generate method.
            
        Returns:
            Dict[str, float]: A dictionary with the benchmark results:
                - total_time: Total time in seconds
                - avg_latency: Average latency in seconds
                - avg_tokens_per_second: Average tokens per second
        """
        # Get the model to use
        model = self.get_model()
        
        # Ensure the model is in the right mode
        model.eval()
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Prepare the generate kwargs
        generate_kwargs = generate_kwargs or {}
        generate_kwargs["max_new_tokens"] = max_new_tokens
        
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        
        # Warmup
        logger.info("Warming up...")
        with torch.inference_mode():
            model.generate(input_ids, **generate_kwargs)
        
        # Benchmark
        logger.info(f"Running benchmark with {num_runs} runs...")
        total_time = 0
        total_tokens = 0
        
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.inference_mode():
                outputs = model.generate(input_ids, **generate_kwargs)
            
            run_time = time.time() - start_time
            tokens_generated = outputs.shape[1] - input_ids.shape[1]
            
            logger.info(f"Run {i+1}: {run_time:.2f}s, {tokens_generated} tokens, "
                       f"{tokens_generated/run_time:.2f} tokens/s")
            
            total_time += run_time
            total_tokens += tokens_generated
        
        # Calculate averages
        avg_latency = total_time / num_runs
        avg_tokens_per_second = total_tokens / total_time
        
        logger.info(f"Benchmark results: {avg_latency:.2f}s avg latency, "
                   f"{avg_tokens_per_second:.2f} tokens/s")
        
        return {
            "total_time": total_time,
            "avg_latency": avg_latency,
            "avg_tokens_per_second": avg_tokens_per_second,
        }


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a model compiler
    compiler = ModelCompiler(
        model=model,
        compile_mode="reduce-overhead",
        compile_backend="inductor",
        enable_sdpa=True,
    )
    
    # Apply optimizations
    compiler.optimize_attention()
    
    # Get the compiled model
    compiled_model = compiler.get_model()
    
    # Benchmark
    prompt = "Write a Python function to calculate Fibonacci numbers:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    results = compiler.benchmark(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        num_runs=3,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    print(f"Average latency: {results['avg_latency']:.2f}s")
    print(f"Average tokens per second: {results['avg_tokens_per_second']:.2f}")
    
    # Generate with the compiled model
    with torch.inference_mode():
        outputs = compiled_model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}") 