from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Optional, Dict, Any, List
import torch
import wandb
from ..model.qwen_handler import QwenModelHandler
from ..data.prompt_creator import PromptCreator
from ..data.response_parser import ResponseParser
import numpy as np

class ValidationCallback(TrainerCallback):
    """Callback for validation during training"""
    
    def __init__(self, trainer_instance):
        self.trainer = trainer_instance
        self.best_metric = float('inf')
        self.best_checkpoint = None
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """Called after evaluation"""
        # Get validation metric
        metric_to_check = args.metric_for_best_model
        metric_value = metrics.get(metric_to_check)
        
        if metric_value is not None:
            # Check if this is the best model
            if args.greater_is_better:
                is_best = metric_value > self.best_metric
            else:
                is_best = metric_value < self.best_metric
                
            if is_best:
                self.best_metric = metric_value
                self.best_checkpoint = state.best_model_checkpoint
                
                # Log best metric
                metrics["best_" + metric_to_check] = self.best_metric

class EarlyStoppingCallback(TrainerCallback):
    """Callback for early stopping"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float('inf')
        self.no_improvement_count = 0
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """Called after evaluation"""
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
