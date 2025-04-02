import torch
import numpy as np
from typing import Dict, Any, List
import re
import yaml

def evaluate_reasoning_quality(predicted_reasoning: str, teacher_reasoning: str, model=None, tokenizer=None) -> float:
    """
    Evaluate the quality of predicted reasoning compared to teacher reasoning

    Args:
        predicted_reasoning: Model's predicted reasoning
        teacher_reasoning: Teacher's reasoning (ground truth)
        model: Optional model for embeddings
        tokenizer: Optional tokenizer for the model

    Returns:
        Score between 0 and 1 indicating reasoning quality
    """
    if not predicted_reasoning or not teacher_reasoning:
        return 0.0

    # Clean up YAML formatting
    def clean_yaml_text(text):
        text = re.sub(r"```yaml\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return text

    predicted_reasoning = clean_yaml_text(predicted_reasoning)
    teacher_reasoning = clean_yaml_text(teacher_reasoning)

    # Parse YAML sections
    def extract_sections(yaml_text):
        try:
            yaml_content = yaml.safe_load("---\n" + yaml_text)
            if isinstance(yaml_content, dict):
                return {
                    k: v
                    for k, v in yaml_content.items()
                    if k in ["reasoning", "understanding", "conclusion", "analysis"]
                }
        except:
            sections = {}
            for section in ["reasoning", "understanding", "conclusion", "analysis"]:
                pattern = rf"{section}:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)"
                match = re.search(pattern, yaml_text)
                if match:
                    sections[section] = match.group(1).strip()
            return sections
        return {}

    predicted_sections = extract_sections(predicted_reasoning)
    teacher_sections = extract_sections(teacher_reasoning)

    # Calculate section-by-section similarity
    section_scores = []
    common_sections = set(predicted_sections.keys()) & set(teacher_sections.keys())

    if not common_sections:
        # Use simple text similarity if no common sections
        if tokenizer:
            encoding1 = tokenizer(
                predicted_reasoning,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding="max_length",
            )
            encoding2 = tokenizer(
                teacher_reasoning,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                padding="max_length",
            )
            similarity = (encoding1.input_ids * encoding2.input_ids).sum().item() / 512
            return max(0.0, min(1.0, similarity / 100))
        return 0.0

    # Calculate similarity for each section
    for section in common_sections:
        pred_text = predicted_sections[section]
        teacher_text = teacher_sections[section]

        if tokenizer:
            encoding1 = tokenizer(
                pred_text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                padding="max_length",
            )
            encoding2 = tokenizer(
                teacher_text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                padding="max_length",
            )
            similarity = (encoding1.input_ids * encoding2.input_ids).sum().item() / 256
            section_scores.append(similarity / 100)

    return max(0.0, min(1.0, sum(section_scores) / len(section_scores))) if section_scores else 0.0

def calculate_combined_score(accuracy: float, reasoning_quality: float, weight: float = 0.6) -> float:
    """
    Calculate combined score from accuracy and reasoning quality

    Args:
        accuracy: Model accuracy
        reasoning_quality: Reasoning quality score
        weight: Weight for reasoning quality (1-weight for accuracy)

    Returns:
        Combined score between 0 and 1
    """
    return weight * reasoning_quality + (1 - weight) * accuracy

def calculate_perplexity(model, tokenizer, prompt: str, answer: str) -> float:
    """
    Calculate perplexity for a prompt-answer pair

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        answer: Generated answer

    Returns:
        Perplexity score
    """
    inputs = tokenizer(prompt + answer, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
