"""
Memory profiling callback for monitoring CUDA memory usage during training.
"""

import gc
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class MemoryProfilingCallback(BaseCallback):
    """
    Callback for comprehensive memory profiling during training.

    Features:
    1. CUDA memory tracking (allocated, cached, reserved)
    2. Memory profiling for forward/backward passes
    3. Peak memory usage monitoring
    4. Memory fragmentation detection
    5. GPU utilization tracking
    6. Automatic memory logging and visualization
    7. Memory leak detection
    8. OOM prevention warnings
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        detailed_profiling: bool = True,
        warning_threshold: float = 0.90,  # 90% memory usage warning
        track_fragmentation: bool = True,
        log_to_file: bool = True,
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.detailed_profiling = detailed_profiling
        self.warning_threshold = warning_threshold
        self.track_fragmentation = track_fragmentation
        self.log_to_file = log_to_file
        self.output_dir = output_dir

        # Initialize tracking
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.memory_stats = []
        self.step_memory = {}
        self.phase_memory = {}
        self.last_gc_count = 0

        # Create output directory if needed
        if self.log_to_file and self.output_dir:
            self.memory_log_dir = os.path.join(self.output_dir, "memory_logs")
            os.makedirs(self.memory_log_dir, exist_ok=True)
            self.memory_log_file = os.path.join(self.memory_log_dir, "memory_profile.log")
            self.memory_stats_file = os.path.join(self.memory_log_dir, "memory_stats.json")

        # Initialize CUDA events for timing
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "cpu_memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.now().isoformat(),
        }

        if torch.cuda.is_available():
            # CUDA memory stats
            stats.update(
                {
                    "cuda_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                    "cuda_reserved": torch.cuda.memory_reserved() / (1024**3),  # GB
                    "cuda_max_allocated": torch.cuda.max_memory_allocated() / (1024**3),
                    "cuda_max_reserved": torch.cuda.max_memory_reserved() / (1024**3),
                }
            )

            # Get per-device memory for multi-GPU setups
            for i in range(torch.cuda.device_count()):
                stats[f"cuda_{i}_allocated"] = torch.cuda.memory_allocated(i) / (1024**3)
                stats[f"cuda_{i}_reserved"] = torch.cuda.memory_reserved(i) / (1024**3)

            # Calculate memory fragmentation
            if self.track_fragmentation:
                allocated = stats["cuda_allocated"]
                reserved = stats["cuda_reserved"]
                if reserved > 0:
                    stats["fragmentation_ratio"] = 1 - (allocated / reserved)
                else:
                    stats["fragmentation_ratio"] = 0.0

            # Get GPU utilization if available
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_utilization"] = utilization.gpu
                stats["gpu_memory_utilization"] = utilization.memory
            except:
                pass

        return stats

    def _check_memory_status(self, stats: Dict[str, float]) -> None:
        """Check memory status and log warnings if needed."""
        if not torch.cuda.is_available():
            return

        # Check if memory usage exceeds warning threshold
        if "cuda_allocated" in stats:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_usage = stats["cuda_allocated"] / total_memory

            if memory_usage > self.warning_threshold:
                warning_msg = (
                    f"⚠️ High memory usage detected: {memory_usage*100:.1f}% "
                    f"({stats['cuda_allocated']:.2f}GB / {total_memory:.2f}GB)"
                )
                logger.warning(warning_msg)

                # Log to wandb if available
                try:
                    import wandb

                    if wandb.run:
                        wandb.alert(
                            title="High Memory Usage Warning",
                            text=warning_msg,
                            level=wandb.AlertLevel.WARNING,
                        )
                except ImportError:
                    pass

        # Check for memory fragmentation
        if "fragmentation_ratio" in stats and stats["fragmentation_ratio"] > 0.3:
            logger.warning(
                f"High memory fragmentation detected: "
                f"{stats['fragmentation_ratio']*100:.1f}% of reserved memory is unused"
            )

    def _log_memory_stats(self, stats: Dict[str, float], step: int, phase: str = ""):
        """Log memory statistics to various outputs."""
        # Add step and phase information
        stats["step"] = step
        if phase:
            stats["phase"] = phase

        # Log to console
        logger.info(f"\nMemory stats at step {step} {phase}:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Log to file if enabled
        if self.log_to_file and hasattr(self, "memory_log_file"):
            with open(self.memory_log_file, "a") as f:
                f.write(f"\nStep {step} {phase}:\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

        # Log to wandb if available
        try:
            import wandb

            if wandb.run:
                # Create a more organized wandb log structure
                log_data = {
                    "memory/step": step,
                    "memory/timestamp": stats["timestamp"],
                }

                # CUDA memory metrics
                if "cuda_allocated" in stats:
                    log_data.update(
                        {
                            "memory/cuda/allocated_gb": stats["cuda_allocated"],
                            "memory/cuda/reserved_gb": stats["cuda_reserved"],
                            "memory/cuda/max_allocated_gb": stats["cuda_max_allocated"],
                            "memory/cuda/max_reserved_gb": stats["cuda_max_reserved"],
                        }
                    )

                # CPU metrics
                log_data.update(
                    {
                        "memory/cpu/percent": stats["cpu_percent"],
                        "memory/cpu/memory_percent": stats["cpu_memory_percent"],
                    }
                )

                # GPU utilization if available
                if "gpu_utilization" in stats:
                    log_data.update(
                        {
                            "memory/gpu/utilization": stats["gpu_utilization"],
                            "memory/gpu/memory_utilization": stats["gpu_memory_utilization"],
                        }
                    )

                # Fragmentation metrics
                if "fragmentation_ratio" in stats:
                    log_data["memory/fragmentation_ratio"] = stats["fragmentation_ratio"]

                # Phase-specific metrics
                if phase:
                    log_data["memory/phase"] = phase

                wandb.log(log_data, step=step)
        except ImportError:
            pass

        # Store in history
        self.memory_stats.append(stats)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Record memory usage at the start of each step."""
        if state.global_step % self.log_every_n_steps == 0:
            stats = self._get_memory_stats()
            self.step_memory["begin"] = stats

            if self.detailed_profiling and torch.cuda.is_available():
                self.start_event.record()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Record and analyze memory usage at the end of each step."""
        if state.global_step % self.log_every_n_steps == 0:
            # Get current stats
            stats = self._get_memory_stats()
            self.step_memory["end"] = stats

            # Calculate memory change during step
            if "begin" in self.step_memory:
                begin_stats = self.step_memory["begin"]
                memory_change = {}

                for key in stats:
                    if key in begin_stats and isinstance(stats[key], (int, float)):
                        memory_change[f"{key}_change"] = stats[key] - begin_stats[key]

                stats.update(memory_change)

            # Check memory status
            self._check_memory_status(stats)

            # Log statistics
            self._log_memory_stats(stats, state.global_step, "step_end")

            # Measure CUDA event timing if enabled
            if self.detailed_profiling and torch.cuda.is_available():
                self.end_event.record()
                torch.cuda.synchronize()
                step_time = (
                    self.start_event.elapsed_time(self.end_event) / 1000
                )  # Convert to seconds
                logger.info(f"Step time: {step_time:.3f}s")

            # Check for memory leaks
            current_gc_count = gc.get_count()[0]
            if current_gc_count > self.last_gc_count + 2:
                logger.warning(
                    f"Potential memory leak detected: "
                    f"Garbage collector runs increased by {current_gc_count - self.last_gc_count}"
                )
            self.last_gc_count = current_gc_count

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize memory tracking at the start of training."""
        logger.info("Initializing memory profiling...")

        # Clear CUDA cache at the start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get initial memory stats
        stats = self._get_memory_stats()
        self._log_memory_stats(stats, 0, "train_begin")

        # Log initial GPU info
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            logger.info(f"\nGPU Information:")
            logger.info(f"Device: {properties.name}")
            logger.info(f"Total memory: {properties.total_memory/1024**3:.2f}GB")
            logger.info(f"CUDA capability: {properties.major}.{properties.minor}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Summarize memory usage at the end of training."""
        # Get final memory stats
        stats = self._get_memory_stats()
        self._log_memory_stats(stats, state.global_step, "train_end")

        # Calculate and log memory usage summary
        if self.memory_stats:
            peak_allocated = max(s.get("cuda_allocated", 0) for s in self.memory_stats)
            peak_reserved = max(s.get("cuda_reserved", 0) for s in self.memory_stats)
            avg_allocated = sum(s.get("cuda_allocated", 0) for s in self.memory_stats) / len(
                self.memory_stats
            )
            avg_fragmentation = sum(
                s.get("fragmentation_ratio", 0) for s in self.memory_stats
            ) / len(self.memory_stats)

            summary = {
                "peak_allocated_gb": peak_allocated,
                "peak_reserved_gb": peak_reserved,
                "avg_allocated_gb": avg_allocated,
                "avg_fragmentation_ratio": avg_fragmentation,
                "total_steps_profiled": len(self.memory_stats),
            }

            logger.info("\nMemory Usage Summary:")
            logger.info(f"Peak allocated: {peak_allocated:.2f}GB")
            logger.info(f"Peak reserved: {peak_reserved:.2f}GB")
            logger.info(f"Average allocated: {avg_allocated:.2f}GB")
            logger.info(f"Average fragmentation: {avg_fragmentation*100:.1f}%")

            # Save summary to file
            if self.log_to_file and hasattr(self, "memory_stats_file"):
                import json

                with open(self.memory_stats_file, "w") as f:
                    json.dump(summary, f, indent=2)

            # Log to wandb
            try:
                import wandb

                if wandb.run:
                    wandb.run.summary.update(
                        {
                            "memory/summary/peak_allocated_gb": peak_allocated,
                            "memory/summary/peak_reserved_gb": peak_reserved,
                            "memory/summary/avg_allocated_gb": avg_allocated,
                            "memory/summary/avg_fragmentation": avg_fragmentation,
                        }
                    )
            except ImportError:
                pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
