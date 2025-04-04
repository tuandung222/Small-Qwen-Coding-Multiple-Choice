import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    metrics: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        eval_dataset: The dataset to evaluate on
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        metrics: List of metrics to compute
        max_length: Maximum sequence length
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        length_penalty: Length penalty for beam search
        no_repeat_ngram_size: Size of n-grams that can't be repeated
        early_stopping: Whether to use early stopping in beam search

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        # Default metrics if none provided
        if metrics is None:
            metrics = ["accuracy", "perplexity"]

        results = {}

        # Create data loader
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False
        )

        # Initialize metric accumulators
        total_loss = 0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", input_ids.clone()).to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

                loss = outputs.loss
                logits = outputs.logits

                # Accumulate metrics
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += attention_mask.sum().item()

                if "accuracy" in metrics:
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.ne(tokenizer.pad_token_id).sum().item()

        # Calculate metrics
        if "perplexity" in metrics:
            results["perplexity"] = np.exp(total_loss / total_tokens)

        if "accuracy" in metrics:
            results["accuracy"] = (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            )

        logger.info(f"Evaluation results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: Optional[int] = None,
    num_beams: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate text from a prompt

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        max_length: Maximum sequence length
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        length_penalty: Length penalty for beam search
        no_repeat_ngram_size: Size of n-grams that can't be repeated
        early_stopping: Whether to use early stopping in beam search
        device: Device to use for generation

    Returns:
        str: Generated text
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise


def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray],
    metric_fn: Optional[callable] = None,
    metric_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for evaluation predictions

    Args:
        eval_pred: Tuple of predictions and labels
        metric_fn: Optional metric function to use
        metric_fn_kwargs: Optional keyword arguments for metric function

    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    try:
        predictions, labels = eval_pred

        if metric_fn is not None:
            metric_fn_kwargs = metric_fn_kwargs or {}
            return metric_fn(predictions, labels, **metric_fn_kwargs)

        # Default metrics
        results = {}

        # Accuracy
        results["accuracy"] = (predictions == labels).mean()

        # Perplexity (if logits are provided)
        if len(predictions.shape) > 1:
            log_probs = -np.log(predictions + 1e-10)
            results["perplexity"] = np.exp(log_probs.mean())

        return results

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise


def save_evaluation_results(
    results: Dict[str, float],
    output_dir: str,
    prefix: str = "eval",
) -> str:
    """
    Save evaluation results to a file

    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save results
        prefix: Prefix for the results file

    Returns:
        str: Path to the saved results file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"{prefix}_results_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved evaluation results to {results_file}")
        return results_file

    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        raise


def load_evaluation_results(
    results_file: str,
) -> Dict[str, float]:
    """
    Load evaluation results from a file

    Args:
        results_file: Path to the results file

    Returns:
        Dict[str, float]: Dictionary of evaluation results
    """
    try:
        with open(results_file, "r") as f:
            results = json.load(f)

        logger.info(f"Loaded evaluation results from {results_file}")
        return results

    except Exception as e:
        logger.error(f"Error loading evaluation results: {str(e)}")
        raise
