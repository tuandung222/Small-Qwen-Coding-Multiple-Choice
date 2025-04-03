import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """
    Configuration for metric computation
    """
    metric_names: List[str]
    average: str = "binary"
    sample_weight: Optional[np.ndarray] = None
    zero_division: int = 0
    labels: Optional[List[int]] = None
    pos_label: int = 1
    beta: float = 1.0


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    config: MetricConfig,
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        config: Metric configuration
        
    Returns:
        Dict[str, float]: Computed metrics
    """
    try:
        metrics = {}
        
        for metric_name in config.metric_names:
            if metric_name == "accuracy":
                metrics["accuracy"] = accuracy_score(
                    labels,
                    predictions,
                    sample_weight=config.sample_weight,
                )
            elif metric_name in ["precision", "recall", "f1"]:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels,
                    predictions,
                    average=config.average,
                    sample_weight=config.sample_weight,
                    zero_division=config.zero_division,
                    labels=config.labels,
                    pos_label=config.pos_label,
                    beta=config.beta,
                )
                metrics["precision"] = precision
                metrics["recall"] = recall
                metrics["f1"] = f1
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def compute_rouge_metrics(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE metrics
    
    Args:
        predictions: Generated texts
        labels: Reference texts
        
    Returns:
        Dict[str, float]: ROUGE metrics
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        
        metrics = {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
        
        for pred, label in zip(predictions, labels):
            scores = scorer.score(label, pred)
            metrics["rouge1"] += scores["rouge1"].fmeasure
            metrics["rouge2"] += scores["rouge2"].fmeasure
            metrics["rougeL"] += scores["rougeL"].fmeasure
            
        # Average scores
        num_samples = len(predictions)
        metrics = {k: v / num_samples for k, v in metrics.items()}
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing ROUGE metrics: {str(e)}")
        raise


def compute_bleu_metrics(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute BLEU metrics
    
    Args:
        predictions: Generated texts
        labels: Reference texts
        
    Returns:
        Dict[str, float]: BLEU metrics
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smooth = SmoothingFunction()
        metrics = {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
        }
        
        for pred, label in zip(predictions, labels):
            # Tokenize
            pred_tokens = pred.split()
            label_tokens = label.split()
            
            # Compute BLEU scores
            metrics["bleu1"] += sentence_bleu(
                [label_tokens],
                pred_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=smooth.method1,
            )
            metrics["bleu2"] += sentence_bleu(
                [label_tokens],
                pred_tokens,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smooth.method1,
            )
            metrics["bleu3"] += sentence_bleu(
                [label_tokens],
                pred_tokens,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smooth.method1,
            )
            metrics["bleu4"] += sentence_bleu(
                [label_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth.method1,
            )
            
        # Average scores
        num_samples = len(predictions)
        metrics = {k: v / num_samples for k, v in metrics.items()}
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing BLEU metrics: {str(e)}")
        raise


def compute_bert_score(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, float]:
    """
    Compute BERTScore
    
    Args:
        predictions: Generated texts
        labels: Reference texts
        
    Returns:
        Dict[str, float]: BERTScore metrics
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            predictions,
            labels,
            lang="en",
            verbose=False,
        )
        
        metrics = {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing BERTScore: {str(e)}")
        raise


def compute_metrics_for_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metric_config: MetricConfig,
) -> Dict[str, float]:
    """
    Compute metrics for text generation
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        eval_dataloader: Evaluation data loader
        device: Device to evaluate on
        metric_config: Metric configuration
        
    Returns:
        Dict[str, float]: Computed metrics
    """
    try:
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Generate
                outputs = model.generate(
                    **batch,
                    max_length=100,
                    num_beams=4,
                    early_stopping=True,
                )
                
                # Decode
                predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
        # Compute metrics
        metrics = {}
        
        # Classification metrics
        if any(metric in metric_config.metric_names for metric in ["accuracy", "precision", "recall", "f1"]):
            # Convert to numpy arrays
            predictions = np.array(all_predictions)
            labels = np.array(all_labels)
            
            # Compute metrics
            classification_metrics = compute_metrics(
                predictions=predictions,
                labels=labels,
                config=metric_config,
            )
            metrics.update(classification_metrics)
            
        # ROUGE metrics
        if "rouge" in metric_config.metric_names:
            rouge_metrics = compute_rouge_metrics(
                predictions=all_predictions,
                labels=all_labels,
            )
            metrics.update(rouge_metrics)
            
        # BLEU metrics
        if "bleu" in metric_config.metric_names:
            bleu_metrics = compute_bleu_metrics(
                predictions=all_predictions,
                labels=all_labels,
            )
            metrics.update(bleu_metrics)
            
        # BERTScore
        if "bert_score" in metric_config.metric_names:
            bert_score_metrics = compute_bert_score(
                predictions=all_predictions,
                labels=all_labels,
            )
            metrics.update(bert_score_metrics)
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing metrics for generation: {str(e)}")
        raise 