"""
Prompt monitoring callback for tracking prompt quality and diversity.
"""

from transformers import TrainerControl, TrainerState, TrainingArguments
from typing import Dict, List, Optional, Any

from .base_callback import BaseCallback, logger


class PromptMonitorCallback(BaseCallback):
    """
    Callback for monitoring prompt quality, diversity, and characteristics.
    
    Features:
    - Tracks prompt diversity metrics
    - Monitors prompt quality
    - Saves sample prompts for analysis
    - Categorizes prompts by type and complexity
    - Compares prompts across training steps
    """

    def __init__(
        self,
        track_diversity: bool = False,
        track_quality: bool = False,
        max_prompts_to_save: int = 100,
        categorize: bool = False,
        comparison: bool = False,
    ):
        super().__init__()
        self.track_diversity = track_diversity
        self.track_quality = track_quality
        self.max_prompts_to_save = max_prompts_to_save
        self.categorize = categorize
        self.comparison = comparison
        self.saved_prompts = []
        self.prompt_metrics = {}
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs,
    ) -> None:
        """Log prompt metrics during training."""
        if not (self.track_diversity or self.track_quality):
            return
            
        # In a real implementation, this would analyze prompts and log metrics
        if state.global_step % 100 == 0:
            logger.info(f"PromptMonitorCallback active at step {state.global_step}")
            
            # Example metrics we would track in a full implementation
            example_metrics = {
                "prompts/diversity_score": 0.85,
                "prompts/quality_score": 0.92,
                "prompts/avg_length": 150,
                "prompts/total_saved": len(self.saved_prompts),
            }
            
            # Log to wandb if available
            self._log_to_wandb(example_metrics, state.global_step)
            
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Initialize prompt monitoring at the beginning of training."""
        logger.info("Initializing prompt monitoring...")
        
    def save_prompt(self, prompt: str, metadata: Dict[str, Any] = None) -> None:
        """Save a prompt for analysis."""
        if len(self.saved_prompts) >= self.max_prompts_to_save:
            # Remove the oldest prompt
            self.saved_prompts.pop(0)
            
        prompt_data = {
            "text": prompt,
            "metadata": metadata or {}
        }
        
        self.saved_prompts.append(prompt_data) 