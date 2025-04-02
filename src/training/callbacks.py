import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import wandb
from src.model.qwen_handler import QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser

# Setup logger
logger = logging.getLogger(__name__)


class ValidationCallback(TrainerCallback):
    """
    Enhanced callback for validation, metric tracking, and model pushing.

    Features:
    1. Regular validation every N steps
    2. Comprehensive metric calculation
    3. Automatic model pushing on improvement
    4. Detailed logging to WandB
    5. Early stopping support
    6. Initial validation before training starts
    """

    def __init__(
        self,
        trainer_instance,
        validation_steps: int = 50,
        push_to_hub: bool = True,
        metric_for_best: str = "eval_loss",
        greater_is_better: bool = False,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0,
        validate_at_start: bool = True,
    ):
        self.trainer = trainer_instance
        self.validation_steps = validation_steps
        self.push_to_hub = push_to_hub
        self.metric_for_best = metric_for_best
        self.greater_is_better = greater_is_better
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.validate_at_start = validate_at_start

        # Initialize tracking
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_checkpoint = None
        self.validation_history = []
        self.no_improvement_count = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run initial validation before training starts if validate_at_start is True."""
        if self.validate_at_start:
            logger.info("Running initial validation before training starts...")
            metrics = self._run_validation()

            # Log metrics
            self._log_metrics(metrics, 0)  # Use step 0 for initial validation

            # Check for improvement (this will be the first validation)
            current_metric = metrics.get(self.metric_for_best)
            if current_metric is not None:
                improved = self._check_improvement(current_metric)
                if improved:
                    self._handle_improvement(metrics, 0)
                else:
                    # If no improvement, still update best metric
                    self.best_metric = current_metric
                    logger.info(f"Initial {self.metric_for_best}: {self.best_metric:.4f}")

            logger.info("Initial validation completed.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run validation every validation_steps."""
        if state.global_step % self.validation_steps == 0:
            # Run validation
            metrics = self._run_validation()

            # Log metrics
            self._log_metrics(metrics, state.global_step)

            # Check for improvement
            current_metric = metrics.get(self.metric_for_best)
            if current_metric is not None:
                improved = self._check_improvement(current_metric)
                if improved:
                    self._handle_improvement(metrics, state.global_step)
                else:
                    self.no_improvement_count += 1
                    # Check for early stopping
                    if self.no_improvement_count >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {self.no_improvement_count} validations without improvement"
                        )
                        control.should_training_stop = True
                if improved:
                    self.no_improvement_count = 0

    def _run_validation(self) -> Dict[str, float]:
        """Run comprehensive validation."""
        try:
            # Get validation dataset
            val_dataset = self.trainer.val_dataset
            if val_dataset is None:
                return {}

            # Run evaluation with response-only loss calculation
            metrics = self.trainer.evaluate(
                val_dataset,
                metric_key_prefix="eval",
                compute_loss_on_response_only=True,  # New parameter to indicate response-only loss
            )

            # Calculate perplexity from response-only loss
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Add validation step information
            metrics["validation_step"] = self.trainer.state.global_step
            metrics["validation_epoch"] = self.trainer.state.epoch

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {}

    def _calculate_additional_metrics(self, val_dataset) -> Dict[str, float]:
        """Calculate additional validation metrics."""
        metrics = {}
        try:
            # Get validation dataset
            val_dataset = self.trainer.val_dataset
            if val_dataset is None:
                return {}

            # Run evaluation with response-only loss calculation
            metrics = self.trainer.evaluate(
                val_dataset,
                metric_key_prefix="eval",
                compute_loss_on_response_only=True,  # New parameter to indicate response-only loss
            )

            # Calculate perplexity from response-only loss
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"]))

            # Add validation step information
            metrics["validation_step"] = self.trainer.state.global_step
            metrics["validation_epoch"] = self.trainer.state.epoch

        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")

        return metrics

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to various destinations."""
        # Log to WandB
        try:
            if wandb.run is not None:
                # Log basic metrics
                wandb.log(metrics, step=step)

                # Create validation trend plot
                if self.validation_history:
                    trend_data = [
                        [h["step"], h["metrics"][self.metric_for_best]]
                        for h in self.validation_history
                    ]
                    trend_data.append([step, metrics[self.metric_for_best]])

                    wandb.log(
                        {
                            "validation_trend": wandb.plot.line(
                                table_data=trend_data,
                                columns=["step", self.metric_for_best],
                                title=f"{self.metric_for_best} Trend",
                            )
                        }
                    )
        except ImportError:
            pass

        # Log to console
        logger.info(f"\nValidation metrics at step {step}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Store in history
        self.validation_history.append({"step": step, "metrics": metrics})

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best so far."""
        if self.greater_is_better:
            return current_metric > (self.best_metric + self.early_stopping_min_delta)
        return current_metric < (self.best_metric - self.early_stopping_min_delta)

    def _handle_improvement(self, metrics: Dict[str, float], step: int):
        """Handle model improvement."""
        # Update best metric
        self.best_metric = metrics[self.metric_for_best]

        # Save checkpoint
        checkpoint_dir = os.path.join(self.trainer.args.output_dir, f"checkpoint-{step}")
        self.trainer.save_checkpoint(checkpoint_dir)
        self.best_checkpoint = checkpoint_dir

        # Push to hub if configured
        if self.push_to_hub and hasattr(self.trainer, "push_to_hub"):
            try:
                logger.info("Pushing best model to hub...")
                self.trainer.push_to_hub(
                    commit_message=f"Best model at step {step} with {self.metric_for_best}={self.best_metric:.4f}"
                )
                logger.info("Successfully pushed to hub")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")

        # Log improvement
        logger.info(
            f"New best model at step {step} with {self.metric_for_best}={self.best_metric:.4f}"
        )

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Summarize validation history at end of training."""
        # Create validation summary
        summary = {
            "best_metric": self.best_metric,
            "best_step": next(
                (
                    h["step"]
                    for h in self.validation_history
                    if h["metrics"][self.metric_for_best] == self.best_metric
                ),
                None,
            ),
            "total_validations": len(self.validation_history),
            "validation_trend": [
                h["metrics"][self.metric_for_best] for h in self.validation_history
            ],
        }

        # Log summary
        logger.info("\nValidation Summary:")
        logger.info(f"Best {self.metric_for_best}: {summary['best_metric']:.4f}")
        logger.info(f"Best step: {summary['best_step']}")
        logger.info(f"Total validations: {summary['total_validations']}")

        # Log to WandB
        try:
            if wandb.run is not None:
                wandb.run.summary.update(summary)
        except ImportError:
            pass


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for implementing early stopping during training.

    This callback monitors validation metrics and stops training when no improvement
    is seen for a specified number of validation rounds (patience).

    Key Features:
    1. Configurable Patience:
       - Set how many validation rounds to wait for improvement
       - Default patience is 3 rounds

    2. Minimum Improvement Delta:
       - Define minimum change required to consider as improvement
       - Helps avoid stopping due to minor fluctuations

    3. Flexible Metric Monitoring:
       - Works with any validation metric
       - Supports both minimization and maximization objectives

    4. Automatic Training Control:
       - Automatically signals training to stop when criteria are met
       - Integrates seamlessly with the training loop

    Args:
        patience (int, optional): Number of validation rounds to wait for improvement.
            Default is 3.
        min_delta (float, optional): Minimum change in metric to qualify as improvement.
            Default is 0.0.

    Attributes:
        patience (int): Number of rounds to wait
        min_delta (float): Minimum improvement required
        best_metric (float): Best metric value seen so far
        no_improvement_count (int): Number of rounds without improvement

    Example:
        ```python
        # Create callback with custom patience
        early_stopping = EarlyStoppingCallback(
            patience=5,      # Wait for 5 rounds
            min_delta=0.01  # Require 1% improvement
        )

        # Add to trainer's callback list
        trainer.train(
            callbacks=[early_stopping],
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        ```

    Note:
        - The callback automatically adapts to whether the metric should be
          maximized or minimized based on the trainer's configuration.
        - Training will stop when no_improvement_count >= patience
        - Set min_delta higher for metrics with high variance to avoid premature stopping
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.no_improvement_count = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """
        Called after each validation step to check for early stopping conditions.

        This method:
        1. Retrieves the current validation metric
        2. Compares it with the best metric considering min_delta
        3. Updates the no-improvement counter
        4. Signals training to stop if patience is exceeded

        Args:
            args: Training arguments containing configuration
            state: Current training state
            control: Training control object
            metrics: Dictionary of current metrics
            **kwargs: Additional arguments
        """
        # Get validation metric
        metric_to_check = args.metric_for_best_model
        metric_value = metrics.get(metric_to_check)

        if metric_value is not None:
            # Check if this is the best model
            if args.greater_is_better:
                is_improvement = metric_value > (self.best_metric + self.min_delta)
            else:
                is_improvement = metric_value < (self.best_metric - self.min_delta)

            if is_improvement:
                self.best_metric = metric_value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            # Stop training if no improvement for patience epochs
            if self.no_improvement_count >= self.patience:
                control.should_training_stop = True


class LRMonitorCallback(TrainerCallback):
    """Custom callback to track learning rates during training."""

    def __init__(self, trainer=None):
        self.trainer = trainer

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        if state.global_step % args.logging_steps == 0:
            try:
                # Get learning rate scheduler
                lr_scheduler = self.trainer.lr_scheduler
                optimizer = self.trainer.optimizer

                # Get current learning rate
                if hasattr(lr_scheduler, "get_last_lr"):
                    lrs = lr_scheduler.get_last_lr()
                    current_lr = lrs[0] if lrs else None
                else:
                    # Fallback - try to get from optimizer
                    current_lr = optimizer.param_groups[0]["lr"]

                # Log to wandb
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "trainer/learning_rate": current_lr,
                                "trainer/global_step": state.global_step,
                                "trainer/epoch": state.epoch,
                                "trainer/total_steps": state.max_steps,
                                "trainer/percent_complete": state.global_step
                                / state.max_steps
                                * 100
                                if state.max_steps
                                else 0,
                            }
                        )

                        # Also log optimizer parameters
                        if optimizer and hasattr(optimizer, "param_groups"):
                            for i, param_group in enumerate(optimizer.param_groups):
                                # Log parameters like weight decay, momentum, etc.
                                for key, value in param_group.items():
                                    if key != "params" and not isinstance(value, (list, tuple)):
                                        wandb.log({f"optimizer/group{i}_{key}": value})
                except ImportError:
                    logger.warning("wandb not installed, skipping logging")
            except Exception as e:
                logger.warning(f"Error logging learning rate: {e}")
        return control


class PromptMonitorCallback(TrainerCallback):
    """Custom callback to show random prompts during training with enhanced visualization."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        logging_steps: int = 10,
        save_to_file: bool = True,
        log_to_wandb: bool = True,
        max_prompts_to_save: int = 100,
        analyze_tokens: bool = True,
        show_token_stats: bool = True,
        output_dir: Optional[str] = None,
        track_diversity: bool = True,
        track_quality: bool = True,
        enable_interactive: bool = False,
        categorize_prompts: bool = True,
        enable_comparison: bool = False,
    ):
        """
        Initialize the prompt monitor callback with enhanced features.

        Args:
            dataset: The training dataset
            tokenizer: The tokenizer used for the model
            logging_steps: Number of steps between showing prompts
            save_to_file: Whether to save prompts to a file in the experiment folder
            log_to_wandb: Whether to log prompts to wandb
            max_prompts_to_save: Maximum number of prompts to save to file
            analyze_tokens: Whether to analyze token distribution
            show_token_stats: Whether to show token statistics in terminal
            output_dir: Directory to save prompt files (defaults to trainer output_dir)
            track_diversity: Whether to track prompt diversity over time
            track_quality: Whether to track prompt quality metrics
            enable_interactive: Whether to enable interactive prompt selection
            categorize_prompts: Whether to categorize prompts automatically
            enable_comparison: Whether to enable prompt comparison features
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.logging_steps = logging_steps
        self.last_prompt = None
        self.last_prompt_idx = None
        self.save_to_file = save_to_file
        self.log_to_wandb = log_to_wandb
        self.max_prompts_to_save = max_prompts_to_save
        self.analyze_tokens = analyze_tokens
        self.show_token_stats = show_token_stats
        self.output_dir = output_dir
        self.prompt_history = []
        self.trainer = None
        self.prompt_file = None
        self.prompt_file_path = None

        # New features
        self.track_diversity = track_diversity
        self.track_quality = track_quality
        self.enable_interactive = enable_interactive
        self.categorize_prompts = categorize_prompts
        self.enable_comparison = enable_comparison

        # Initialize diversity tracking
        self.diversity_scores = []
        self.last_prompt_embedding = None

        # Initialize quality tracking
        self.quality_scores = []

        # Initialize categorization
        self.prompt_categories = {}
        self.category_counts = {}

        # Initialize comparison data
        self.comparison_prompts = []

        # For interactive mode
        self.interactive_mode = False
        self.marked_prompts = set()

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Initialize prompt file at the beginning of training."""
        if self.save_to_file and self.output_dir is None:
            self.output_dir = args.output_dir

        if self.save_to_file and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.prompt_file_path = os.path.join(self.output_dir, "prompt_history.json")
            self.prompt_file = open(self.prompt_file_path, "w")
            self.prompt_file.write("[\n")  # Start JSON array
            logger.info(f"Saving prompts to {self.prompt_file_path}")

            # Create additional files for enhanced features
            if self.track_diversity:
                diversity_file = os.path.join(self.output_dir, "prompt_diversity.json")
                with open(diversity_file, "w") as f:
                    f.write("[]")
                logger.info(f"Created diversity tracking file: {diversity_file}")

            if self.track_quality:
                quality_file = os.path.join(self.output_dir, "prompt_quality.json")
                with open(quality_file, "w") as f:
                    f.write("[]")
                logger.info(f"Created quality tracking file: {quality_file}")

            if self.categorize_prompts:
                categories_file = os.path.join(self.output_dir, "prompt_categories.json")
                with open(categories_file, "w") as f:
                    f.write("{}")
                logger.info(f"Created categories file: {categories_file}")

        return control

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Close prompt file at the end of training and save additional data."""
        if self.save_to_file and self.prompt_file:
            self.prompt_file.write("\n]")  # End JSON array
            self.prompt_file.close()
            logger.info(f"Saved {len(self.prompt_history)} prompts to {self.prompt_file_path}")

            # Save additional data
            if self.track_diversity and self.output_dir:
                diversity_file = os.path.join(self.output_dir, "prompt_diversity.json")
                with open(diversity_file, "w") as f:
                    import json

                    json.dump(self.diversity_scores, f, indent=2)
                logger.info(f"Saved diversity scores to {diversity_file}")

            if self.track_quality and self.output_dir:
                quality_file = os.path.join(self.output_dir, "prompt_quality.json")
                with open(quality_file, "w") as f:
                    import json

                    json.dump(self.quality_scores, f, indent=2)
                logger.info(f"Saved quality scores to {quality_file}")

            if self.categorize_prompts and self.output_dir:
                categories_file = os.path.join(self.output_dir, "prompt_categories.json")
                with open(categories_file, "w") as f:
                    import json

                    json.dump(self.category_counts, f, indent=2)
                logger.info(f"Saved category counts to {categories_file}")

            if self.enable_comparison and self.output_dir and self.comparison_prompts:
                comparison_file = os.path.join(self.output_dir, "prompt_comparisons.json")
                with open(comparison_file, "w") as f:
                    import json

                    json.dump(self.comparison_prompts, f, indent=2)
                logger.info(f"Saved comparison data to {comparison_file}")

        return control

    def _analyze_tokens(self, text: str) -> Dict[str, Any]:
        """Analyze token distribution in the prompt."""
        tokens = self.tokenizer.encode(text)
        token_ids = tokens
        token_texts = [self.tokenizer.decode([t]) for t in tokens]

        # Count token frequencies
        token_freq = {}
        for token_id in token_ids:
            token_freq[token_id] = token_freq.get(token_id, 0) + 1

        # Sort by frequency
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)

        # Get top tokens
        top_tokens = sorted_tokens[:10]
        top_token_texts = [(self.tokenizer.decode([t[0]]), t[1]) for t in top_tokens]

        return {
            "token_count": len(tokens),
            "unique_tokens": len(token_freq),
            "top_tokens": top_token_texts,
            "token_ids": token_ids,
            "token_texts": token_texts,
        }

    def _calculate_prompt_quality(self, text: str) -> Dict[str, float]:
        """Calculate quality metrics for the prompt."""
        # Basic quality metrics
        metrics = {
            "length": len(text),
            "word_count": len(text.split()),
            "avg_word_length": sum(len(word) for word in text.split()) / max(1, len(text.split())),
            "sentence_count": text.count(".") + text.count("!") + text.count("?"),
            "question_count": text.count("?"),
            "code_block_count": text.count("```"),
        }

        # Calculate complexity (simple approximation)
        words = text.split()
        unique_words = set(words)
        metrics["vocabulary_richness"] = len(unique_words) / max(1, len(words))

        # Calculate readability (Flesch-Kincaid approximation)
        if metrics["sentence_count"] > 0:
            metrics["readability"] = (
                0.39 * (len(words) / metrics["sentence_count"])
                + 11.8 * (len(unique_words) / len(words))
                - 15.59
            )
        else:
            metrics["readability"] = 0

        return metrics

    def _calculate_prompt_diversity(self, text: str) -> float:
        """Calculate diversity score compared to previous prompts."""
        if not self.prompt_history:
            return 1.0  # First prompt is considered maximally diverse

        # Simple token-based diversity
        current_tokens = set(self.tokenizer.encode(text))

        # Calculate Jaccard similarity with previous prompts
        similarities = []
        for prev_prompt in self.prompt_history[-5:]:  # Compare with last 5 prompts
            prev_tokens = set(prev_prompt.get("token_analysis", {}).get("token_ids", []))
            if prev_tokens:
                intersection = len(current_tokens.intersection(prev_tokens))
                union = len(current_tokens.union(prev_tokens))
                similarity = intersection / max(1, union)
                similarities.append(similarity)

        # Diversity is inverse of average similarity
        avg_similarity = sum(similarities) / max(1, len(similarities))
        diversity = 1.0 - avg_similarity

        return diversity

    def _categorize_prompt(self, text: str) -> str:
        """Categorize the prompt based on content."""
        # Simple keyword-based categorization
        text_lower = text.lower()

        if "function" in text_lower or "def " in text_lower:
            return "function_definition"
        elif "class" in text_lower or "object" in text_lower:
            return "class_definition"
        elif "error" in text_lower or "exception" in text_lower:
            return "error_handling"
        elif "loop" in text_lower or "for " in text_lower or "while " in text_lower:
            return "loops"
        elif "if " in text_lower or "else" in text_lower:
            return "conditionals"
        elif "import" in text_lower or "from " in text_lower:
            return "imports"
        elif "return" in text_lower:
            return "return_statements"
        elif "?" in text_lower:
            return "questions"
        elif "```" in text_lower:
            return "code_blocks"
        else:
            return "general"

    def _save_prompt_to_file(self, prompt_data: Dict[str, Any]) -> None:
        """Save prompt data to file."""
        if not self.save_to_file or not self.prompt_file:
            return

        # Add comma if not the first entry
        if self.prompt_history:
            self.prompt_file.write(",\n")

        # Write prompt data as JSON
        import json

        json.dump(prompt_data, self.prompt_file, indent=2)

        # Add to history
        self.prompt_history.append(prompt_data)

        # Trim history if needed
        if len(self.prompt_history) > self.max_prompts_to_save:
            logger.warning(f"Reached maximum prompts to save ({self.max_prompts_to_save})")

        # Update category counts
        if self.categorize_prompts and "category" in prompt_data:
            category = prompt_data["category"]
            self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def _log_to_wandb(self, prompt_data: Dict[str, Any]) -> None:
        """Log prompt data to wandb."""
        if not self.log_to_wandb:
            return

        try:
            import wandb

            if wandb.run is not None:
                # Create a table for the prompt
                prompt_table = wandb.Table(
                    columns=["step", "prompt", "token_count", "unique_tokens", "category"]
                )
                prompt_table.add_data(
                    prompt_data["step"],
                    prompt_data["prompt"],
                    prompt_data["token_analysis"]["token_count"],
                    prompt_data["token_analysis"]["unique_tokens"],
                    prompt_data.get("category", "unknown"),
                )

                # Log the table
                wandb.log(
                    {
                        "prompts/current": prompt_table,
                        "prompts/token_count": prompt_data["token_analysis"]["token_count"],
                        "prompts/unique_tokens": prompt_data["token_analysis"]["unique_tokens"],
                    }
                )

                # Log top tokens as a bar chart
                if "top_tokens" in prompt_data["token_analysis"]:
                    top_tokens_data = {
                        token: count for token, count in prompt_data["token_analysis"]["top_tokens"]
                    }
                    wandb.log(
                        {
                            "prompts/top_tokens": wandb.plot.bar(
                                wandb.Table(
                                    data=[[k, v] for k, v in top_tokens_data.items()],
                                    columns=["token", "count"],
                                ),
                                "token",
                                "count",
                                title="Top Tokens in Prompt",
                            )
                        }
                    )

                # Log quality metrics if available
                if "quality_metrics" in prompt_data:
                    for metric, value in prompt_data["quality_metrics"].items():
                        wandb.log({f"prompts/quality/{metric}": value})

                # Log diversity score if available
                if "diversity_score" in prompt_data:
                    wandb.log({"prompts/diversity_score": prompt_data["diversity_score"]})

                # Log category distribution
                if self.category_counts:
                    category_table = wandb.Table(
                        data=[[k, v] for k, v in self.category_counts.items()],
                        columns=["category", "count"],
                    )
                    wandb.log(
                        {
                            "prompts/category_distribution": wandb.plot.bar(
                                category_table, "category", "count", title="Prompt Categories"
                            )
                        }
                    )
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning(f"Error logging to wandb: {e}")

    def _handle_interactive_mode(self, prompt_data: Dict[str, Any]) -> None:
        """Handle interactive prompt selection if enabled."""
        if not self.enable_interactive:
            return

        try:
            import sys

            print("\n" + "=" * 80)
            print("INTERACTIVE PROMPT MODE")
            print("=" * 80)
            print(f"Step: {prompt_data['step']}")
            print(f"Category: {prompt_data.get('category', 'unknown')}")
            print(f"Token Count: {prompt_data['token_analysis']['token_count']}")
            print("-" * 80)
            print(prompt_data["prompt"])
            print("-" * 80)
            print("Options:")
            print("  [s] Save this prompt for later comparison")
            print("  [m] Mark this prompt as interesting")
            print("  [c] Continue to next prompt")
            print("  [q] Quit interactive mode")

            choice = input("Enter your choice: ").strip().lower()

            if choice == "s":
                self.comparison_prompts.append(prompt_data)
                print(f"Saved prompt for comparison. Total saved: {len(self.comparison_prompts)}")
            elif choice == "m":
                self.marked_prompts.add(prompt_data["step"])
                print("Marked prompt as interesting")
            elif choice == "q":
                self.interactive_mode = False
                print("Exiting interactive mode")

        except Exception as e:
            logger.warning(f"Error in interactive mode: {e}")

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Show a random prompt at each logging step with enhanced visualization."""
        if state.global_step % self.logging_steps == 0:
            try:
                # Sample a random example that's different from the last one
                max_attempts = 5  # Limit attempts to avoid infinite loop
                for _ in range(max_attempts):
                    idx = random.randint(0, len(self.dataset) - 1)
                    if idx != self.last_prompt_idx:
                        break

                example = self.dataset[idx]
                self.last_prompt_idx = idx

                # Get the prompt text
                prompt = example["text"]

                # Only show if it's different from the last one
                if prompt != self.last_prompt:
                    # Analyze tokens if enabled
                    token_analysis = {}
                    if self.analyze_tokens:
                        token_analysis = self._analyze_tokens(prompt)

                    # Calculate quality metrics if enabled
                    quality_metrics = {}
                    if self.track_quality:
                        quality_metrics = self._calculate_prompt_quality(prompt)
                        self.quality_scores.append(
                            {"step": state.global_step, "metrics": quality_metrics}
                        )

                    # Calculate diversity score if enabled
                    diversity_score = 0.0
                    if self.track_diversity:
                        diversity_score = self._calculate_prompt_diversity(prompt)
                        self.diversity_scores.append(
                            {"step": state.global_step, "score": diversity_score}
                        )

                    # Categorize prompt if enabled
                    category = "unknown"
                    if self.categorize_prompts:
                        category = self._categorize_prompt(prompt)

                    # Create prompt data
                    prompt_data = {
                        "step": state.global_step,
                        "epoch": state.epoch,
                        "prompt": prompt,
                        "example_idx": idx,
                        "token_analysis": token_analysis,
                        "quality_metrics": quality_metrics,
                        "diversity_score": diversity_score,
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Save to file
                    self._save_prompt_to_file(prompt_data)

                    # Log to wandb
                    self._log_to_wandb(prompt_data)

                    # Handle interactive mode if enabled
                    if self.enable_interactive and self.interactive_mode:
                        self._handle_interactive_mode(prompt_data)

                    # Print to terminal
                    print("\n" + "=" * 80)
                    print(f"Random Training Prompt (Step {state.global_step}):")
                    print("-" * 80)
                    print(prompt)

                    # Show token statistics if enabled
                    if self.show_token_stats and token_analysis:
                        print("-" * 80)
                        print(f"Token Count: {token_analysis['token_count']}")
                        print(f"Unique Tokens: {token_analysis['unique_tokens']}")
                        print("Top Tokens:")
                        for token, count in token_analysis.get("top_tokens", [])[:5]:
                            print(f"  {token}: {count}")

                    # Show quality metrics if enabled
                    if self.track_quality and quality_metrics:
                        print("-" * 80)
                        print("Quality Metrics:")
                        for metric, value in quality_metrics.items():
                            print(f"  {metric}: {value:.2f}")

                    # Show diversity score if enabled
                    if self.track_diversity:
                        print("-" * 80)
                        print(f"Diversity Score: {diversity_score:.4f}")

                    # Show category if enabled
                    if self.categorize_prompts:
                        print("-" * 80)
                        print(f"Category: {category}")

                    print("=" * 80 + "\n")
                    self.last_prompt = prompt

            except Exception as e:
                logger.warning(f"Error showing random prompt: {e}")
        return control


class ModelLoadingAlertCallback(TrainerCallback):
    """Callback to alert when model loading method changes."""

    def __init__(self, use_unsloth: bool = True):
        """
        Initialize the model loading alert callback.

        Args:
            use_unsloth: Whether Unsloth was attempted for model loading
        """
        self.use_unsloth = use_unsloth
        self.alert_shown = False
        self.trainer = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Show alert at the beginning of training if Unsloth was attempted but not used."""
        if self.use_unsloth and not self.alert_shown:
            try:
                # Check if trainer is available
                if self.trainer is None:
                    logger.warning("Trainer not available in ModelLoadingAlertCallback")
                    return control

                # Check if model is available
                if not hasattr(self.trainer, "model") or self.trainer.model is None:
                    logger.warning("Model not available in ModelLoadingAlertCallback")
                    return control

                # Check if the model is using Unsloth
                model = self.trainer.model
                if not hasattr(model, "is_unsloth_model") or not model.is_unsloth_model:
                    print("\n" + "=" * 80)
                    print(
                        "\033[91mWARNING: Using standard Transformers loading instead of Unsloth optimization\033[0m"
                    )
                    print(
                        "\033[91mThis may result in slower training and higher memory usage\033[0m"
                    )
                    print("=" * 80 + "\n")
                    self.alert_shown = True
            except Exception as e:
                logger.warning(f"Error checking model loading method: {e}")
        return control
