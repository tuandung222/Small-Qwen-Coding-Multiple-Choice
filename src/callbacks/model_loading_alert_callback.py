"""
Model loading alert callback for QLoRA and Unsloth optimization warnings.
"""

from typing import Dict

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from .base_callback import BaseCallback, logger


class ModelLoadingAlertCallback(BaseCallback):
    """
    Callback to alert when model loading method changes and warn about optimizations.

    Features:
    - Monitors QLoRA configuration
    - Checks if Unsloth was requested but not used
    - Provides detailed memory usage analysis
    - Alerts about missing optimizations
    """

    def __init__(self, use_unsloth: bool = True):
        super().__init__()
        self.use_unsloth = use_unsloth
        self.alert_shown = False
        self.trainer = None
        self.warning_count = 0
        self.max_warnings = 3  # Show warning multiple times during training

    def _calculate_memory_usage(self, model) -> Dict[str, float]:
        """Calculate approximate memory usage for QLoRA."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Estimate memory usage for QLoRA
            base_model_memory = total_params * 0.5 / (1024**3)  # GB for 4-bit model
            lora_memory = trainable_params * 4 / (1024**3)  # GB for LoRA weights (FP32)
            optimizer_memory = trainable_params * 8 / (1024**3)  # GB for Adam states
            activation_memory = 2  # GB (approximate)

            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "base_model_memory_gb": base_model_memory,
                "lora_memory_gb": lora_memory,
                "optimizer_memory_gb": optimizer_memory,
                "activation_memory_gb": activation_memory,
                "total_memory_gb": base_model_memory
                + lora_memory
                + optimizer_memory
                + activation_memory,
            }
        except Exception as e:
            logger.warning(f"Failed to calculate memory usage: {e}")
            return None

    def _show_unsloth_warning(self, model):
        """Show warning when Unsloth was requested but not used."""
        warning_msg = "\n" + "!" * 80 + "\n"
        warning_msg += "⚠️  UNSLOTH OPTIMIZATION WARNING  ⚠️\n"
        warning_msg += "!" * 80 + "\n\n"

        warning_msg += "Training is using HuggingFace's transformers instead of Unsloth!\n"
        warning_msg += "This means you're missing out on significant speed improvements:\n"
        warning_msg += "1. 🚀 SPEED IMPROVEMENTS:\n"
        warning_msg += "   - Up to 2x faster training\n"
        warning_msg += "   - Optimized attention patterns\n"
        warning_msg += "   - Better memory utilization\n"

        warning_msg += "\n2. 💾 MEMORY BENEFITS:\n"
        warning_msg += "   - More efficient memory usage\n"
        warning_msg += "   - Better GPU memory management\n"
        warning_msg += "   - Reduced CPU-GPU transfers\n"

        warning_msg += "\nRECOMMENDED ACTIONS:\n"
        warning_msg += "1. Stop training\n"
        warning_msg += "2. Ensure Unsloth is properly installed:\n"
        warning_msg += "   pip install unsloth\n"
        warning_msg += "3. Check model initialization code\n"
        warning_msg += "4. Verify CUDA compatibility\n"
        warning_msg += "5. Restart training with Unsloth enabled\n"

        warning_msg += "\n" + "!" * 80 + "\n"

        print(warning_msg)

        # Log to wandb
        try:
            import wandb

            if wandb.run:
                wandb.alert(
                    title="⚠️ Training Without Unsloth Optimizations",
                    text=warning_msg,
                    level=wandb.AlertLevel.WARNING,
                )
        except ImportError:
            pass

    def _show_qlora_warning(self, model):
        """Show warning about missing QLoRA optimizations."""
        memory_info = self._calculate_memory_usage(model)

        warning_msg = "\n" + "!" * 80 + "\n"
        warning_msg += "⚠️  CRITICAL PERFORMANCE WARNING  ⚠️\n"
        warning_msg += "!" * 80 + "\n\n"

        warning_msg += "Training is proceeding WITHOUT proper QLoRA optimizations! This means:\n"
        warning_msg += "1. 💾 SIGNIFICANTLY HIGHER MEMORY USAGE:\n"
        if memory_info:
            warning_msg += (
                f"   - Base model memory (4-bit): {memory_info['base_model_memory_gb']:.2f} GB\n"
            )
            warning_msg += f"   - LoRA weights memory: {memory_info['lora_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Optimizer states: {memory_info['optimizer_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Activation memory: {memory_info['activation_memory_gb']:.2f} GB\n"
            warning_msg += f"   - Total estimated memory: {memory_info['total_memory_gb']:.2f} GB\n"
            warning_msg += f"   - With proper QLoRA, this could be reduced by up to 80%!\n"

        warning_msg += "\n2. ⚡ TRAINING INEFFICIENCIES:\n"
        warning_msg += "   - Missing 4-bit quantization optimizations\n"
        warning_msg += "   - Suboptimal memory access patterns\n"
        warning_msg += "   - Higher CPU-GPU transfer overhead\n"

        warning_msg += "\n3. 🔧 MISSING QLORA FEATURES:\n"
        warning_msg += "   - No double quantization\n"
        warning_msg += "   - No NF4 data type benefits\n"
        warning_msg += "   - No paged optimizer states\n"
        warning_msg += "   - No efficient k-bit training\n"

        warning_msg += "\nRECOMMENDED ACTIONS:\n"
        warning_msg += "1. Stop training (Ctrl+C)\n"
        warning_msg += "2. Ensure proper QLoRA setup:\n"
        warning_msg += "   - Install bitsandbytes>=0.39.0\n"
        warning_msg += "   - Install transformers>=4.31.0\n"
        warning_msg += "   - Install accelerate>=0.20.3\n"
        warning_msg += "   - Install peft>=0.4.0\n"
        warning_msg += "3. Check BitsAndBytes configuration\n"
        warning_msg += "4. Verify k-bit training preparation\n"
        warning_msg += "5. Restart training with proper QLoRA settings\n"

        warning_msg += "\n" + "!" * 80 + "\n"

        print(warning_msg)

        # Log to wandb
        if memory_info:
            metrics = {
                "memory/base_model_gb": memory_info["base_model_memory_gb"],
                "memory/lora_weights_gb": memory_info["lora_memory_gb"],
                "memory/optimizer_gb": memory_info["optimizer_memory_gb"],
                "memory/activation_gb": memory_info["activation_memory_gb"],
                "memory/total_gb": memory_info["total_memory_gb"],
            }
            self._log_to_wandb(metrics, state.global_step if state else 0)

            # Also send alert to wandb
            try:
                import wandb

                if wandb.run:
                    wandb.alert(
                        title="⚠️ Training Without QLoRA Optimizations",
                        text=warning_msg,
                        level=wandb.AlertLevel.ERROR,
                    )
            except ImportError:
                pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Show alerts at the beginning of training."""
        if not self.alert_shown:
            try:
                if self.trainer is None:
                    logger.warning("Trainer not available in ModelLoadingAlertCallback")
                    return control

                if not hasattr(self.trainer, "model") or self.trainer.model is None:
                    logger.warning("Model not available in ModelLoadingAlertCallback")
                    return control

                model = self.trainer.model

                # Check for Unsloth usage
                if self.use_unsloth and not hasattr(model, "is_unsloth_model"):
                    self._show_unsloth_warning(model)
                    self.alert_shown = True
                    self.warning_count += 1

                # Check for QLoRA optimizations
                if not hasattr(model, "is_qlora_model") or not model.is_qlora_model:
                    self._show_qlora_warning(model)
                    self.alert_shown = True
                    self.warning_count += 1

            except Exception as e:
                logger.warning(f"Error checking model loading method: {e}")

        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        """Show periodic warnings during training."""
        if self.warning_count < self.max_warnings:
            if state.global_step % 100 == 0:  # Show warning every 100 steps
                try:
                    model = self.trainer.model

                    # Check for Unsloth usage
                    if self.use_unsloth and not hasattr(model, "is_unsloth_model"):
                        self._show_unsloth_warning(model)
                        self.warning_count += 1

                    # Check for QLoRA optimizations
                    if not hasattr(model, "is_qlora_model") or not model.is_qlora_model:
                        self._show_qlora_warning(model)
                        self.warning_count += 1

                except Exception as e:
                    logger.warning(f"Error checking model during training: {e}")

        return control
