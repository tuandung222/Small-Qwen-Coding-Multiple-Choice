#!/usr/bin/env python3
# teacher_synthesis.py
import argparse
import concurrent.futures
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from openai import OpenAI
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Configuration for the model being tested"""

    name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    system_prompt: str = "You are a helpful assistant that answers multiple choice questions with detailed reasoning."

    def to_dict(self):
        """Convert to dictionary, excluding the API key"""
        config_dict = asdict(self)
        config_dict.pop("api_key")
        return config_dict


@dataclass
class PromptConfig:
    """Configuration for the prompt generation"""

    template_type: str = "reasoning"  # Options: "basic", "reasoning", "custom"
    custom_template: Optional[str] = None
    add_system_prompt: bool = True
    include_step_numbers: bool = True
    yaml_format: bool = True
    custom_template_fn: Optional[Callable] = None

    def get_template_function(self):
        """Get the appropriate template function based on configuration"""
        if self.template_type == "basic":
            return self._basic_template
        elif self.template_type == "reasoning":
            return self._reasoning_template
        elif self.template_type == "custom" and self.custom_template:
            return lambda q, c: self._format_custom_template(self.custom_template, q, c)
        elif self.custom_template_fn:
            return self.custom_template_fn
        else:
            # Default to reasoning template
            return self._reasoning_template

    def _basic_template(self, question: str, choices: List[str]) -> str:
        """Basic template that just asks for an answer letter"""
        prompt = "MULTIPLE CHOICE QUESTION:\n"
        prompt += "-" * 50 + "\n\n"

        # Add the question
        prompt += f"QUESTION:\n{question}\n\n"

        # Add the choices section header
        prompt += "CHOICES:\n"

        # Add each choice with a letter identifier
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D, ...
            prompt += f"{letter}. {choice}\n"

        # Add separator
        prompt += "\n" + "-" + "\n"

        return prompt

    def _reasoning_template(self, question: str, choices: List[str]) -> str:
        """Reasoning template that asks for step-by-step reasoning"""
        formatted_choices = "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        max_letter = chr(65 + len(choices) - 1)  # Find the last valid letter based on choice count

        steps_text = ""
        if self.include_step_numbers:
            steps_text = """Think through this step-by-step:
1. Understand what the question is asking
2. Analyze each option carefully
3. Reason about why each option might be correct or incorrect
4. Select the most appropriate answer
"""
        else:
            steps_text = """Think through this step-by-step:
- Understand what the question is asking
- Analyze each option carefully
- Reason about why each option might be correct or incorrect
- Select the most appropriate answer
"""

        if self.yaml_format:
            response_format = f"""Your response should be in YAML format:
understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter} representing your final answer>
"""
        else:
            response_format = f"""Provide your reasoning first, followed by your answer.

REASONING:
<your step-by-step reasoning here>

ANSWER: <single letter A through {max_letter} representing your final answer>
"""

        return f"""
{question}

{formatted_choices}

{steps_text}

{response_format}
"""

    def _reasoning_with_label_template(
        self, question: str, choices: List[str], correct_answer: str
    ) -> str:
        """
        Template for teacher models that includes the correct answer.
        The teacher will generate a YAML-formatted explanation based on all information.

        Args:
            question: The question text
            choices: List of possible answers
            correct_answer: The correct answer (usually a letter A, B, C, etc.)

        Returns:
            Formatted prompt for teacher models
        """
        formatted_choices = "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        max_letter = chr(65 + len(choices) - 1)  # Find the last valid letter based on choice count

        # If correct_answer is an index, convert to letter
        if isinstance(correct_answer, int):
            correct_answer = chr(65 + correct_answer)

        return f"""
            TASK: You are a teacher creating an exemplary explanation for a multiple-choice question.

            QUESTION:
            {question}

            CHOICES:
            {formatted_choices}

            CORRECT ANSWER: {correct_answer}

            INSTRUCTIONS:
            Using your knowledge of the correct answer, create a comprehensive, educational explanation.
            Your explanation should provide clear reasoning that would help a student understand why {correct_answer} is correct
            and why the other options are incorrect.

            Your response MUST be in YAML format as follows:

            understanding: |
            <explanation of what the question is asking and the key concepts involved>
            analysis: |
            <detailed analysis of each option, explaining why each is correct or incorrect>
            reasoning: |
            <step-by-step reasoning process that leads to the correct answer>
            conclusion: |
            <final explanation summarizing why answer {correct_answer} is correct>
            answer: {correct_answer}

            IMPORTANT: The "answer" field MUST contain ONLY a single character letter (A through {max_letter}) representing the correct option.
            Do not include any explanations or additional text in the answer field.
        """

    def _format_custom_template(self, template: str, question: str, choices: List[str]) -> str:
        """Format a custom template with the question and choices"""
        # Create a formatted choice string
        formatted_choices = "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )

        # Replace placeholders in the template
        return template.replace("{{question}}", question).replace("{{choices}}", formatted_choices)


@dataclass
class TestConfig:
    """Configuration for the test run"""

    output_dir: str = "./mc_test_results"
    sample_size: Optional[int] = None
    random_seed: int = 42
    save_results: bool = True
    create_visualizations: bool = True
    show_progress: bool = True
    save_individual_examples: bool = False
    question_key: str = "question"
    choices_key: str = "list_choices"
    answer_key: str = "answer"
    task_id_key: str = "task_id"
    batch_size: Optional[int] = None  # For batch processing
    verbose: bool = True


class MCTestingFramework:
    """
    Enhanced framework for testing LLMs on multiple choice questions.

    Features:
    - Flexible configuration for models, prompts, and test parameters
    - Support for custom prompt templates
    - Detailed result tracking and analysis
    - Visualization capabilities
    - Batch processing options
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        prompt_config: PromptConfig = None,
        test_config: TestConfig = None,
    ):
        """
        Initialize the testing framework with configurations.

        Args:
            model_config: Configuration for the model
            prompt_config: Configuration for prompt generation
            test_config: Configuration for the test run
        """
        # Use provided configs or defaults
        self.model_config = model_config or ModelConfig()
        self.prompt_config = prompt_config or PromptConfig()
        self.test_config = test_config or TestConfig()

        # Initialize the OpenAI client
        api_key = self.model_config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key provided. Set it in ModelConfig or as environment variable."
            )
        self.client = OpenAI(api_key=api_key)

        # Get the prompt template function
        self.prompt_template_fn = self.prompt_config.get_template_function()

        # Results tracking
        self.reset_results()

        # Ensure output directory exists
        if self.test_config.save_results and self.test_config.output_dir:
            os.makedirs(self.test_config.output_dir, exist_ok=True)

    def reset_results(self):
        """Reset the results tracking"""
        self.results = {
            "task_id": [],
            "correct_answer": [],
            "predicted_answer": [],
            "is_correct": [],
            "reasoning": [],
            "response_time": [],
            "prompt": [],
            "raw_response": [],
        }

    def generate_prompt(self, question: str, choices: List[str]) -> str:
        """Generate a prompt using the configured template function"""
        return self.prompt_template_fn(question, choices)

    def parse_response(self, response_text: str) -> tuple:
        """
        Parse the model's response to extract answer and reasoning.

        Args:
            response_text: Raw response from the model

        Returns:
            Tuple of (answer, reasoning)
        """
        # Default values
        answer = None
        reasoning = ""

        try:
            # First attempt to parse as YAML if using YAML format
            if self.prompt_config.yaml_format:
                try:
                    # Add --- to ensure it's treated as YAML block
                    yaml_text = "---\n" + response_text
                    parsed = yaml.safe_load(yaml_text)
                    if isinstance(parsed, dict):
                        if "answer" in parsed:
                            answer = parsed["answer"]

                        # Extract reasoning from various possible fields
                        reasoning_parts = []
                        for key in ["understanding", "analysis", "reasoning", "conclusion"]:
                            if key in parsed and parsed[key]:
                                reasoning_parts.append(f"{key.upper()}:\n{parsed[key]}")

                        if reasoning_parts:
                            reasoning = "\n\n".join(reasoning_parts)

                        if answer:
                            return answer, reasoning
                except Exception:
                    # If YAML parsing fails, fall back to text parsing
                    pass

            # Text-based parsing as fallback
            if "answer:" in response_text.lower():
                # Find answer line
                answer_line = [
                    line
                    for line in response_text.split("\n")
                    if line.lower().strip().startswith("answer:")
                ][0]
                answer = answer_line.split(":", 1)[1].strip()

            # Extract reasoning based on the format
            if "reasoning:" in response_text.lower() and "answer:" in response_text.lower():
                # Extract reasoning between reasoning: and answer:
                reasoning_start = response_text.lower().find("reasoning:")
                answer_start = response_text.lower().find("answer:")

                if reasoning_start < answer_start:
                    reasoning_text = response_text[reasoning_start:answer_start].strip()
                    reasoning = reasoning_text.replace("reasoning:", "", 1).strip()
                else:
                    # If answer comes before reasoning, get everything after reasoning:
                    reasoning = response_text[reasoning_start:].replace("reasoning:", "", 1).strip()
            elif "reasoning:" in response_text.lower():
                # Just get everything after reasoning:
                reasoning_start = response_text.lower().find("reasoning:")
                reasoning = response_text[reasoning_start:].replace("reasoning:", "", 1).strip()

            # Clean up answer - extract just the letter
            if answer and len(answer) > 1:
                for char in answer:
                    if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        answer = char
                        break

            return answer, reasoning

        except Exception as e:
            if self.test_config.verbose:
                print(f"Error parsing response: {e}")
                print(f"Original response: {response_text}")
            return None, ""

    def generate_answer(self, question: str, choices: List[str]) -> tuple:
        """
        Generate an answer (and reasoning if configured) for a question.

        Args:
            question: The question text
            choices: List of possible answers

        Returns:
            Tuple of (answer, reasoning, response_time, raw_response, prompt)
        """
        prompt = self.generate_prompt(question, choices)

        try:
            start_time = time.time()

            messages = []
            if self.prompt_config.add_system_prompt:
                messages.append({"role": "system", "content": self.model_config.system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                top_p=self.model_config.top_p,
                frequency_penalty=self.model_config.frequency_penalty,
                presence_penalty=self.model_config.presence_penalty,
            )

            response_time = time.time() - start_time
            raw_response = response.choices[0].message.content.strip()

            answer, reasoning = self.parse_response(raw_response)

            return answer, reasoning, response_time, raw_response, prompt

        except Exception as e:
            if self.test_config.verbose:
                print(f"Error generating answer: {e}")
            return None, "", 0, "", prompt

    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single example and return the results.

        Args:
            example: Dictionary containing a single example

        Returns:
            Dictionary with evaluation results
        """
        question = example[self.test_config.question_key]
        choices = example[self.test_config.choices_key]
        correct_answer = example[self.test_config.answer_key]
        task_id = example[self.test_config.task_id_key]

        # Generate prediction
        predicted_answer, reasoning, response_time, raw_response, prompt = self.generate_answer(
            question, choices
        )

        # Check correctness
        is_correct = predicted_answer == correct_answer

        return {
            "task_id": task_id,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "reasoning": reasoning,
            "response_time": response_time,
            "raw_response": raw_response,
            "prompt": prompt,
        }

    def evaluate_batch(self, dataset, start_idx: int, end_idx: int) -> None:
        """
        Evaluate a batch of examples and update results.

        Args:
            dataset: The dataset containing examples
            start_idx: Starting index for this batch
            end_idx: Ending index (exclusive) for this batch
        """
        iterator = range(start_idx, end_idx)
        if self.test_config.show_progress:
            # Only show batch number if we're using batches
            desc = (
                f"Batch {start_idx//self.test_config.batch_size + 1}"
                if self.test_config.batch_size
                else f"Evaluating {self.model_config.name}"
            )
            iterator = tqdm(iterator, desc=desc)

        for i in iterator:
            example = dataset[i]
            result = self.evaluate_example(example)

            # Store results
            for key in self.results:
                if key in result:
                    self.results[key].append(result[key])

            # Optional delay between requests
            # time.sleep(0.1)

    def evaluate_dataset(self, dataset) -> Dict[str, List]:
        """
        Evaluate the model on a dataset of multiple choice questions.

        Args:
            dataset: Dataset containing questions, choices, and answers

        Returns:
            Dictionary of results
        """
        # Reset results
        self.reset_results()

        # Determine sample size
        total_examples = len(dataset)
        sample_size = self.test_config.sample_size or total_examples
        sample_size = min(sample_size, total_examples)

        if self.test_config.verbose:
            print(f"Evaluating {self.model_config.name} on {sample_size} examples...")

        # Process in batches if configured
        if self.test_config.batch_size:
            batch_size = self.test_config.batch_size
            for start_idx in range(0, sample_size, batch_size):
                end_idx = min(start_idx + batch_size, sample_size)

                if self.test_config.verbose:
                    print(f"Processing batch: examples {start_idx} to {end_idx-1}")

                self.evaluate_batch(dataset, start_idx, end_idx)

                # Save intermediate results if requested
                if self.test_config.save_results:
                    batch_num = start_idx // batch_size + 1
                    self.save_current_results(f"batch_{batch_num}_results.csv")

                    if self.test_config.verbose:
                        accuracy = sum(self.results["is_correct"]) / len(self.results["is_correct"])
                        print(f"Batch {batch_num} complete. Running accuracy: {accuracy:.2%}")
        else:
            # Process all at once
            self.evaluate_batch(dataset, 0, sample_size)

        # Calculate final accuracy
        accuracy = sum(self.results["is_correct"]) / len(self.results["is_correct"])

        if self.test_config.verbose:
            print(f"\nEvaluation complete.")
            print(
                f"Accuracy: {accuracy:.2%} ({sum(self.results['is_correct'])}/{len(self.results['is_correct'])})"
            )

        # Save results if requested
        if self.test_config.save_results:
            self.save_all_results()

        return self.results

    def save_current_results(self, filename: str) -> None:
        """Save current results to a CSV file"""
        output_path = os.path.join(self.test_config.output_dir, filename)
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)

        if self.test_config.verbose:
            print(f"Intermediate results saved to {output_path}")

    def save_all_results(self) -> Dict[str, str]:
        """
        Save all results and generate analysis.

        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            self.test_config.output_dir, f"{self.model_config.name.replace('-', '_')}_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save all configurations
        config = {
            "model": self.model_config.to_dict(),
            "prompt": asdict(self.prompt_config),
            "test": asdict(self.test_config),
        }
        config_path = os.path.join(output_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save detailed results
        results_path = os.path.join(output_dir, "detailed_results.csv")
        df = pd.DataFrame(self.results)
        df.to_csv(results_path, index=False)

        # Calculate metrics
        accuracy = sum(self.results["is_correct"]) / len(self.results["is_correct"])
        avg_response_time = sum(self.results["response_time"]) / len(self.results["response_time"])

        metrics = {
            "model": self.model_config.name,
            "total_examples": len(df),
            "correct_answers": sum(df["is_correct"]),
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "timestamp": timestamp,
        }

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Create visualizations if requested
        viz_paths = {}
        if self.test_config.create_visualizations:
            # Overall accuracy
            plt.figure(figsize=(8, 6))
            plt.bar(
                ["Correct", "Incorrect"],
                [metrics["correct_answers"], len(df) - metrics["correct_answers"]],
            )
            plt.title(f"{self.model_config.name} Performance")
            plt.ylabel("Number of Examples")
            plt.ylim(0, len(df))

            # Add percentage labels
            for i, v in enumerate(
                [metrics["correct_answers"], len(df) - metrics["correct_answers"]]
            ):
                plt.text(i, v + 5, f"{v/len(df):.1%}", ha="center")

            accuracy_path = os.path.join(output_dir, "accuracy.png")
            plt.savefig(accuracy_path)
            plt.close()
            viz_paths["accuracy"] = accuracy_path

        # Save individual examples if requested
        example_paths = []
        if self.test_config.save_individual_examples:
            examples_dir = os.path.join(output_dir, "examples")
            os.makedirs(examples_dir, exist_ok=True)

            for i, task_id in enumerate(self.results["task_id"]):
                example = {
                    "task_id": task_id,
                    "correct_answer": self.results["correct_answer"][i],
                    "predicted_answer": self.results["predicted_answer"][i],
                    "is_correct": self.results["is_correct"][i],
                    "reasoning": self.results["reasoning"][i],
                    "prompt": self.results["prompt"][i],
                    "raw_response": self.results["raw_response"][i],
                    "response_time": self.results["response_time"][i],
                }

                # Skip examples with no prediction (errors)
                if example["predicted_answer"] is None:
                    continue

                correctness = "correct" if example["is_correct"] else "incorrect"
                example_path = os.path.join(examples_dir, f"{task_id}_{correctness}.json")

                with open(example_path, "w") as f:
                    json.dump(example, f, indent=2)

                example_paths.append(example_path)

        if self.test_config.verbose:
            print(f"All results and analysis saved to {output_dir}")

        return {
            "config": config_path,
            "results": results_path,
            "metrics": metrics_path,
            "visualizations": viz_paths,
            "examples": example_paths,
            "output_dir": output_dir,
        }


def create_special_validation_dataset(
    data_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_val",
    num_samples: int = 100,
    random_seed: int = 42,
    output_dir: str = "./datasets",
    save_dataset: bool = True,
    verbose: bool = True,
):
    """
    Create a special validation dataset with randomly sampled examples.

    Args:
        data_path: Path to the validation dataset
        num_samples: Number of examples to sample
        random_seed: Random seed for reproducibility
        output_dir: Directory to save the dataset info
        save_dataset: Whether to save the dataset information
        verbose: Whether to print information

    Returns:
        The sampled dataset
    """
    if verbose:
        print(f"Loading validation data from {data_path}...")

    # Load the validation dataset
    try:
        split_val = datasets.load_from_disk(data_path)

        if verbose:
            print(f"Validation dataset loaded with {len(split_val)} examples")
            print(f"Dataset features: {split_val.features}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get the total number of examples
    total_val_examples = len(split_val)

    # Ensure we're not sampling more than available
    num_samples = min(num_samples, total_val_examples)

    # Sample random indices without replacement
    random_indices = random.sample(range(total_val_examples), num_samples)

    # Create the special validation dataset
    special_val_dataset = split_val.select(random_indices)

    if verbose:
        print(f"Created special validation dataset with {len(special_val_dataset)} examples")
        print("\nFirst few examples from special validation dataset:")
        for i in range(min(5, len(special_val_dataset))):
            print(f"Example {i}: {special_val_dataset[i]['task_id']}")

    # Save dataset information if requested
    if save_dataset:
        os.makedirs(output_dir, exist_ok=True)

        # Save the task_ids for reference
        task_ids = [example["task_id"] for example in special_val_dataset]
        task_ids_df = pd.DataFrame({"task_id": task_ids})
        task_ids_path = os.path.join(output_dir, "special_val_task_ids.csv")
        task_ids_df.to_csv(task_ids_path, index=False)

        if verbose:
            print(f"\nSaved task IDs to {task_ids_path}")

        # Create a distribution of task_id prefixes
        if all(isinstance(id, str) for id in task_ids):
            prefixes = [id[:1] if len(id) > 0 else "unknown" for id in task_ids]
            prefix_counts = pd.Series(prefixes).value_counts().reset_index()
            prefix_counts.columns = ["prefix", "count"]

            # Save prefix distribution
            prefix_path = os.path.join(output_dir, "task_prefix_distribution.csv")
            prefix_counts.to_csv(prefix_path, index=False)

            if verbose:
                print(f"Saved task prefix distribution to {prefix_path}")

            # Create a visualization of the distribution
            plt.figure(figsize=(10, 6))
            sns.barplot(x="prefix", y="count", data=prefix_counts)
            plt.title("Distribution of Task ID Prefixes")
            plt.xlabel("Prefix")
            plt.ylabel("Count")
            plt.tight_layout()

            viz_path = os.path.join(output_dir, "task_prefix_distribution.png")
            plt.savefig(viz_path)
            plt.close()

            if verbose:
                print(f"Saved visualization to {viz_path}")

    # Print summary statistics
    if verbose and "num_choices" in special_val_dataset.features:
        choice_counts = (
            pd.Series([ex["num_choices"] for ex in special_val_dataset]).value_counts().sort_index()
        )
        print("\nDistribution of number of choices:")
        for choices, count in choice_counts.items():
            print(f"  {choices} choices: {count} examples ({count/len(special_val_dataset):.1%})")

    return special_val_dataset


def create_special_training_dataset(
    data_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_train",
    num_samples: int = 100,
    random_seed: int = 42,
    output_dir: str = "./datasets",
    save_dataset: bool = True,
    verbose: bool = True,
):
    """
    Create a special training dataset with randomly sampled examples.

    Args:
        data_path: Path to the training dataset
        num_samples: Number of examples to sample
        random_seed: Random seed for reproducibility
        output_dir: Directory to save the dataset info
        save_dataset: Whether to save the dataset information
        verbose: Whether to print information

    Returns:
        The sampled dataset
    """
    if verbose:
        print(f"Loading training data from {data_path}...")

    # Load the training dataset
    try:
        split_train = datasets.load_from_disk(data_path)

        if verbose:
            print(f"Training dataset loaded with {len(split_train)} examples")
            print(f"Dataset features: {split_train.features}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get the total number of examples
    total_train_examples = len(split_train)

    # Ensure we're not sampling more than available
    num_samples = min(num_samples, total_train_examples)

    # Sample random indices without replacement
    random_indices = random.sample(range(total_train_examples), num_samples)

    # Create the special training dataset
    special_train_dataset = split_train.select(random_indices)

    if verbose:
        print(f"Created special training dataset with {len(special_train_dataset)} examples")
        print("\nFirst few examples from special training dataset:")
        for i in range(min(5, len(special_train_dataset))):
            print(f"Example {i}: {special_train_dataset[i]['task_id']}")

    # Save dataset information if requested
    if save_dataset:
        os.makedirs(output_dir, exist_ok=True)

        # Save the task_ids for reference
        task_ids = [example["task_id"] for example in special_train_dataset]
        task_ids_df = pd.DataFrame({"task_id": task_ids})
        task_ids_path = os.path.join(output_dir, "special_train_task_ids.csv")
        task_ids_df.to_csv(task_ids_path, index=False)

        if verbose:
            print(f"\nSaved task IDs to {task_ids_path}")

        # Create a distribution of task_id prefixes
        if all(isinstance(id, str) for id in task_ids):
            prefixes = [id[:1] if len(id) > 0 else "unknown" for id in task_ids]
            prefix_counts = pd.Series(prefixes).value_counts().reset_index()
            prefix_counts.columns = ["prefix", "count"]

            # Save prefix distribution
            prefix_path = os.path.join(output_dir, "train_task_prefix_distribution.csv")
            prefix_counts.to_csv(prefix_path, index=False)

            if verbose:
                print(f"Saved task prefix distribution to {prefix_path}")

            # Create a visualization of the distribution
            plt.figure(figsize=(10, 6))
            sns.barplot(x="prefix", y="count", data=prefix_counts)
            plt.title("Distribution of Training Task ID Prefixes")
            plt.xlabel("Prefix")
            plt.ylabel("Count")
            plt.tight_layout()

            viz_path = os.path.join(output_dir, "train_task_prefix_distribution.png")
            plt.savefig(viz_path)
            plt.close()

            if verbose:
                print(f"Saved visualization to {viz_path}")

    # Print summary statistics
    if verbose and "num_choices" in special_train_dataset.features:
        choice_counts = (
            pd.Series([ex["num_choices"] for ex in special_train_dataset])
            .value_counts()
            .sort_index()
        )
        print("\nDistribution of number of choices:")
        for choices, count in choice_counts.items():
            print(f"  {choices} choices: {count} examples ({count/len(special_train_dataset):.1%})")

    return special_train_dataset


# Additional utility functions for dataset filtering
def create_filtered_dataset(dataset, prefix="k", max_samples=50):
    """Create a dataset filtered by task_id prefix"""
    filtered_examples = [i for i in range(len(dataset)) if dataset[i]["task_id"].startswith(prefix)]

    # Sample if we have more than requested
    if len(filtered_examples) > max_samples:
        filtered_examples = random.sample(filtered_examples, max_samples)

    return dataset.select(filtered_examples)


def create_balanced_dataset(dataset, samples_per_choice_count=25):
    """Create a dataset with balanced number of choices"""
    # Group by number of choices
    choice_groups = {}
    for i in range(len(dataset)):
        num_choices = dataset[i]["num_choices"]
        if num_choices not in choice_groups:
            choice_groups[num_choices] = []
        choice_groups[num_choices].append(i)

    # Sample from each group
    balanced_indices = []
    for num_choices, indices in choice_groups.items():
        sample_size = min(samples_per_choice_count, len(indices))
        balanced_indices.extend(random.sample(indices, sample_size))

    return dataset.select(balanced_indices)


# Example usage
if __name__ == "__main__":
    # Create the special validation dataset
    special_val_dataset = create_special_validation_dataset(
        num_samples=10,  # Sample 10 examples
        random_seed=42,  # Fixed seed for reproducibility
        output_dir="./dataset_info",
        save_dataset=True,
        verbose=True,
    )

    # Now you can use special_val_dataset with the MCTestingFramework
    print(f"\nTotal unique task_ids: {len(set([ex['task_id'] for ex in special_val_dataset]))}")

    # Basic example of using the framework
    # Uncomment to run a test
    # """
    from config import OPENAI_API_KEY

    # 1. Set up configurations
    model_config = ModelConfig(
        name="gpt-3.5-turbo", temperature=0.0, max_tokens=1024, api_key=OPENAI_API_KEY
    )

    # 2. Create custom prompt template
    custom_template = """
    QUESTION:
    {{question}}

    ANSWER CHOICES:
    {{choices}}

    Take your time to think through this problem step-by-step.
    First, analyze the question carefully.
    Then, consider each choice one by one.
    Explain your reasoning clearly.

    Finally, provide your answer in this format:
    REASONING: <your detailed reasoning>
    ANSWER: <letter of your answer>
    """

    prompt_config = PromptConfig(
        template_type="custom", custom_template=custom_template, yaml_format=False
    )

    # 3. Configure the test run
    test_config = TestConfig(
        output_dir="./custom_test_results",
        sample_size=2,
        batch_size=5,
        save_individual_examples=True,
        verbose=True,
    )

    # 4. Initialize and run the framework
    tester = MCTestingFramework(
        model_config=model_config, prompt_config=prompt_config, test_config=test_config
    )

    # 5. Run the test
    results = tester.evaluate_dataset(special_val_dataset)
    # """


class TeacherSynthesisFramework:
    """
    Framework for generating synthetic explanations using teacher models with access to labels.

    This framework:
    1. Loads examples from the training set
    2. Uses a teacher model with access to the correct answer to generate explanations
    3. Saves explanations in YAML format for each example
    4. Evaluates the teacher model's performance
    """

    def __init__(
        self,
        model_config: ModelConfig,
        output_dir: str = "./synthesis_results",
        concurrent_requests: int = 5,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the synthesis framework.

        Args:
            model_config: Configuration for the teacher model
            output_dir: Directory to save synthetic explanations
            concurrent_requests: Number of concurrent API requests
            sample_size: Number of examples to process (None for all)
            random_seed: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        self.model_config = model_config
        self.output_dir = output_dir
        self.concurrent_requests = concurrent_requests
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.verbose = verbose

        # Initialize metrics tracking
        self.metrics = {
            "total_examples": 0,
            "successful_generations": 0,
            "correct_answer_preserved": 0,
            "failed_generations": 0,
            "avg_response_time": 0,
            "total_time": 0,
        }

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize prompt config to use reasoning_with_label_template
        self.prompt_config = PromptConfig()
        # We need to save the original template function for later restoration
        self._original_template_fn = self.prompt_config.get_template_function
        self.prompt_config.get_template_function = lambda: self._get_reasoning_with_label_template

        # Initialize the client framework
        self.framework = MCTestingFramework(
            model_config=model_config,
            prompt_config=self.prompt_config,
            test_config=TestConfig(
                output_dir=output_dir, save_results=False, show_progress=False, verbose=False
            ),
        )

    def _get_reasoning_with_label_template(self, question: str, choices: List[str]) -> str:
        """
        Adapter to handle standard template function signature but retrieve label from choices.
        This shouldn't be called directly as we override generate_synthetic_explanation.
        """
        return "Error: This template requires a correct answer. Use generate_synthetic_explanation instead."

    def generate_synthetic_explanation(
        self, question: str, choices: List[str], correct_answer: str
    ) -> Tuple[str, float]:
        """
        Generate a synthetic explanation using the teacher model.

        Args:
            question: The question text
            choices: List of possible answers
            correct_answer: The correct answer (letter or index)

        Returns:
            Tuple of (yaml_explanation, response_time)
        """
        # Use the reasoning_with_label_template from prompt_config
        prompt = self.prompt_config._reasoning_with_label_template(
            question, choices, correct_answer
        )

        try:
            start_time = time.time()

            # Build messages
            messages = []
            if self.prompt_config.add_system_prompt:
                messages.append({"role": "system", "content": self.model_config.system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call the API
            response = self.framework.client.chat.completions.create(
                model=self.model_config.name,
                messages=messages,
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                top_p=self.model_config.top_p,
                frequency_penalty=self.model_config.frequency_penalty,
                presence_penalty=self.model_config.presence_penalty,
            )

            response_time = time.time() - start_time
            yaml_response = response.choices[0].message.content.strip()

            return yaml_response, response_time

        except Exception as e:
            if self.verbose:
                print(f"Error generating explanation: {e}")
            return "", 0

    def _process_example(self, example: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Process a single example and generate synthetic explanation.

        Args:
            example: Dictionary containing a single example
            index: Index of the example

        Returns:
            Dictionary with results
        """
        # Check if we already have a file for this example
        if "task_id" in example:
            task_id = example["task_id"]
            filename = f"{index:06d}_{task_id}.yaml"
            output_path = os.path.join(self.output_dir, filename)

            if os.path.exists(output_path):
                # File already exists, read it and validate
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        yaml_response = f.read()

                    # Parse the response to get the answer
                    answer = None
                    correct_answer = example.get("answer", "")
                    try:
                        # Add --- to ensure it's treated as YAML block
                        yaml_text = "---\n" + yaml_response
                        parsed = yaml.safe_load(yaml_text)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            answer = parsed["answer"]
                    except Exception:
                        # YAML parsing failed
                        pass

                    is_correct = answer == correct_answer

                    # Return the result without making an API call
                    return {
                        "task_id": task_id,
                        "yaml_response": yaml_response,
                        "correct_answer": correct_answer,
                        "predicted_answer": answer,
                        "is_correct": is_correct,
                        "response_time": 0,  # We didn't make an API call
                        "output_file": output_path,
                        "reused_existing": True,
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Error reading existing file {output_path}: {e}")
                    # Continue with normal processing if there was an error

        # Validate example has all required fields
        required_fields = ["question", "choices", "answer", "task_id"]
        for field in required_fields:
            if field not in example:
                if self.verbose:
                    print(f"Example {index} missing required field: '{field}'")
                return {
                    "task_id": example.get("task_id", f"unknown_{index}"),
                    "error": f"Missing required field: {field}",
                    "is_correct": False,
                    "response_time": 0,
                    "yaml_response": "",
                }

        # Extract fields with validation
        question = example["question"]
        task_id = example["task_id"]
        correct_answer = example["answer"]

        # Parse choices from string representation
        choices_str = example["choices"]
        try:
            # Handle different string formats
            if isinstance(choices_str, str):
                if choices_str.startswith("[") and choices_str.endswith("]"):
                    # Try to safely parse the string representation of a list
                    import ast

                    choices = ast.literal_eval(choices_str)
                else:
                    # If it's not a list format, split by delimiter (comma or newline)
                    choices = [c.strip() for c in choices_str.replace("\n", ",").split(",")]
            else:
                # If it's already a list or other iterable
                choices = list(choices_str)

            # Clean up choices
            choices = [str(c).strip() for c in choices if c]
        except Exception as e:
            if self.verbose:
                print(f"Example {index} error parsing choices: {e}")
                print(f"Original choices string: {choices_str}")
            return {
                "task_id": task_id,
                "error": f"Error parsing choices: {str(e)}",
                "is_correct": False,
                "response_time": 0,
                "yaml_response": "",
            }

        # Additional validation - ensure choices is a non-empty list
        if not choices or len(choices) == 0:
            if self.verbose:
                print(f"Example {index} has empty choices after parsing: {choices_str}")
            return {
                "task_id": task_id,
                "error": "Empty choices after parsing",
                "is_correct": False,
                "response_time": 0,
                "yaml_response": "",
            }

        # Extract single-letter answer if answer starts with "ANSWER: "
        if isinstance(correct_answer, str) and correct_answer.startswith("ANSWER:"):
            correct_answer = correct_answer.replace("ANSWER:", "").strip()
            # Take just the first character if it's a letter
            if len(correct_answer) > 0:
                correct_answer = correct_answer[0]

        # Generate synthetic explanation
        yaml_response, response_time = self.generate_synthetic_explanation(
            question, choices, correct_answer
        )

        # Parse the response to verify correctness
        answer = None
        if yaml_response:
            try:
                # Add --- to ensure it's treated as YAML block
                yaml_text = "---\n" + yaml_response
                parsed = yaml.safe_load(yaml_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                # YAML parsing failed
                pass

        # Check if the answer was preserved correctly
        is_correct = answer == correct_answer

        # Create the result dictionary
        result = {
            "task_id": task_id,
            "yaml_response": yaml_response,
            "correct_answer": correct_answer,
            "predicted_answer": answer,
            "is_correct": is_correct,
            "response_time": response_time,
        }

        # Save the synthetic explanation to a file
        if yaml_response:
            try:
                filename = f"{index:06d}_{task_id}.yaml"
                output_path = os.path.join(self.output_dir, filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(yaml_response)

                result["output_file"] = output_path
            except Exception as e:
                if self.verbose:
                    print(f"Error saving synthetic explanation: {e}")

        return result

    def process_dataset(self, dataset) -> Dict[str, Any]:
        """
        Process the dataset and generate synthetic explanations.

        Args:
            dataset: Dataset containing questions, choices, and answers

        Returns:
            Dictionary with metrics
        """
        # Determine sample size
        total_examples = len(dataset)
        sample_size = self.sample_size or total_examples
        sample_size = min(sample_size, total_examples)

        if self.verbose:
            print(
                f"Generating synthetic explanations using {self.model_config.name} for {sample_size} examples..."
            )
            print(f"Saving results to {self.output_dir}")

        # Initialize start time and store it
        start_time = time.time()
        self.metrics["start_time"] = start_time
        results = []

        # Check for existing progress file to resume from a crash
        progress_file = os.path.join(self.output_dir, "synthesis_progress.json")
        processed_indices = set()

        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    progress_data = json.load(f)
                    processed_indices = set(progress_data.get("processed_indices", []))

                    if self.verbose:
                        print(
                            f"Resuming from previous run. {len(processed_indices)} examples already processed."
                        )

                    # Restore previous results if available
                    previous_results = progress_data.get("partial_results", [])
                    if previous_results:
                        results = previous_results
            except Exception as e:
                if self.verbose:
                    print(f"Could not load progress file: {e}. Starting from scratch.")

        # Check for existing files in the output directory
        existing_files = set()
        for file in os.listdir(self.output_dir):
            if file.endswith(".yaml") and file.startswith(
                tuple(f"{i:06d}_" for i in range(sample_size))
            ):
                try:
                    index = int(file.split("_")[0])
                    existing_files.add(index)
                    if index not in processed_indices:
                        if self.verbose:
                            print(f"Found existing file {file} for index {index}")
                        processed_indices.add(index)
                except ValueError:
                    pass

        if existing_files and self.verbose:
            print(f"Found {len(existing_files)} existing output files that will be skipped.")

        # Determine which examples to process (skip already processed ones)
        indices_to_process = [i for i in range(sample_size) if i not in processed_indices]

        if self.verbose:
            print(f"Processing {len(indices_to_process)} examples out of {sample_size} total.")

        # Configuration for periodic saves
        save_interval = 50  # Save after every 50 examples
        time_interval = 600  # Save after every 10 minutes (600 seconds)
        last_save_time = time.time()

        # Process examples concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrent_requests
        ) as executor:
            # Submit tasks for unprocessed examples
            future_to_idx = {
                executor.submit(self._process_example, dataset[i], i): i for i in indices_to_process
            }

            # Process results as they complete
            completed = 0
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(indices_to_process),
                desc="Generating explanations",
                disable=not self.verbose,
            ):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed_indices.add(idx)
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing example {idx}: {e}")
                    results.append(
                        {
                            "task_id": dataset[idx]["task_id"]
                            if "task_id" in dataset[idx]
                            else str(idx),
                            "error": str(e),
                            "is_correct": False,
                        }
                    )
                    processed_indices.add(idx)

                # Periodically save progress
                completed += 1
                current_time = time.time()
                if (
                    completed % save_interval == 0
                    or (current_time - last_save_time) > time_interval
                ):
                    self._save_intermediate_metrics(results, processed_indices, progress_file)
                    last_save_time = current_time

                    if self.verbose and completed % save_interval == 0:
                        print(
                            f"Progress saved after {completed}/{len(indices_to_process)} examples"
                        )

        # Calculate and save final metrics
        self._calculate_metrics(results)

        # Save final metrics and clean up progress file
        metrics_path = os.path.join(self.output_dir, "synthesis_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Clean up progress file after successful completion
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except Exception:
                pass

        if self.verbose:
            success_rate = self.metrics["successful_generations"] / self.metrics["total_examples"]
            preservation_rate = (
                self.metrics["correct_answer_preserved"] / self.metrics["total_examples"]
            )

            print(f"\nSynthesis complete in {self.metrics['total_time']:.2f}s")
            print(f"Average response time: {self.metrics['avg_response_time']:.2f}s")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Answer preservation rate: {preservation_rate:.2%}")
            print(f"Metrics saved to {metrics_path}")

        return self.metrics

    def _save_intermediate_metrics(self, results, processed_indices, progress_file):
        """
        Save intermediate metrics and progress information.

        Args:
            results: List of results processed so far
            processed_indices: Set of indices that have been processed
            progress_file: Path to save the progress information
        """
        # Calculate current metrics
        self._calculate_metrics(results)

        # Save progress information
        progress_data = {
            "processed_indices": list(processed_indices),
            "metrics": self.metrics,
            "partial_results": results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save to file
        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

        # Also save current metrics to a metrics file
        metrics_path = os.path.join(self.output_dir, "synthesis_metrics_latest.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _calculate_metrics(self, results):
        """
        Calculate metrics based on current results.

        Args:
            results: List of results to calculate metrics from
        """
        self.metrics["total_examples"] = len(results)
        self.metrics["successful_generations"] = sum(
            1 for r in results if "yaml_response" in r and r["yaml_response"]
        )
        self.metrics["correct_answer_preserved"] = sum(
            1 for r in results if r.get("is_correct", False)
        )
        self.metrics["failed_generations"] = (
            self.metrics["total_examples"] - self.metrics["successful_generations"]
        )
        self.metrics["reused_existing_files"] = sum(
            1 for r in results if r.get("reused_existing", False)
        )

        response_times = [
            r["response_time"] for r in results if "response_time" in r and r["response_time"] > 0
        ]
        self.metrics["avg_response_time"] = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Calculate total time using stored start_time
        if "start_time" in self.metrics:
            self.metrics["total_time"] = time.time() - self.metrics["start_time"]
        else:
            self.metrics["total_time"] = 0

        # Add timestamp
        self.metrics["last_updated"] = datetime.now().isoformat()


def run_synthesis(
    model_name: str = "gpt-4o",
    data_path: str = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_train",
    sample_size: int = 100,
    output_dir: str = "./synthesis_results",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    concurrent_requests: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
    system_prompt: Optional[str] = None,
):
    """
    Run the synthesis process to generate explanations on training data.

    Args:
        model_name: Name of the OpenAI model to use
        data_path: Path to the training dataset
        sample_size: Number of examples to process
        output_dir: Directory to save outputs
        api_key: OpenAI API key (None to use environment variable)
        temperature: Temperature for model generation
        max_tokens: Maximum tokens for model generation
        concurrent_requests: Number of concurrent API requests
        random_seed: Random seed for reproducibility
        verbose: Whether to print detailed information
        system_prompt: Custom system prompt for the model
    """
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"{model_name.replace('-', '_')}_{timestamp}")

    # Set up model config
    default_system_prompt = "You are an expert teacher creating high-quality, concise explanations for multiple choice questions. Focus on the most important concepts and key distinctions between choices. Be thorough but efficient with your explanations - prioritize clarity and precision over verbosity."

    model_config = ModelConfig(
        name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        system_prompt=system_prompt if system_prompt else default_system_prompt,
    )

    # Create a custom PromptConfig for concise explanations
    prompt_config = PromptConfig()

    # Modify the _reasoning_with_label_template method to emphasize conciseness
    original_template = prompt_config._reasoning_with_label_template

    def concise_reasoning_with_label_template(question, choices, correct_answer):
        formatted_choices = "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        max_letter = chr(65 + len(choices) - 1)

        # If correct_answer is an index, convert to letter
        if isinstance(correct_answer, int):
            correct_answer = chr(65 + correct_answer)

        return f"""
TASK: You are a teacher creating a concise, precise explanation for a multiple-choice question.

QUESTION:
{question}

CHOICES:
{formatted_choices}

CORRECT ANSWER: {correct_answer}

INSTRUCTIONS:
Create a focused, high-quality explanation that clearly demonstrates why {correct_answer} is correct
and why other options are incorrect. Be thorough but concise - focus on key concepts and avoid unnecessary details.

Your response MUST be in YAML format as follows:

understanding: |
  <brief explanation of what the question is asking - focus on key concepts only>
analysis: |
  <concise analysis of each option, prioritizing the critical distinctions>
reasoning: |
  <focused reasoning that leads directly to the correct answer>
conclusion: |
  <brief summary of why answer {correct_answer} is correct>
answer: {correct_answer}

IMPORTANT: The "answer" field MUST contain ONLY a single character letter (A through {max_letter}).
Aim for clarity and precision in your explanations.
"""

    # Override the method
    prompt_config._reasoning_with_label_template = concise_reasoning_with_label_template

    # Load dataset
    if verbose:
        print(f"Loading training dataset from {data_path}...")

    # For None sample_size, we need to first load the dataset to get its size
    # and then process everything
    if sample_size is None:
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(data_path)
            actual_sample_size = len(dataset)
            if verbose:
                print(f"Processing all {actual_sample_size} examples from dataset")
        except Exception as e:
            print(f"Error loading dataset directly: {e}")
            print("Falling back to default sample size of 100")
            actual_sample_size = 100
    else:
        actual_sample_size = sample_size

    train_dataset = create_special_training_dataset(
        data_path=data_path,
        num_samples=actual_sample_size,
        random_seed=random_seed,
        save_dataset=False,
        verbose=verbose,
    )

    if not train_dataset:
        print("Failed to load dataset. Exiting.")
        return

    # Create and run the synthesis framework with the custom prompt config
    synthesizer = TeacherSynthesisFramework(
        model_config=model_config,
        output_dir=output_dir,
        concurrent_requests=concurrent_requests,
        sample_size=actual_sample_size,
        verbose=verbose,
    )

    # Replace the prompt_config with our custom one
    synthesizer.prompt_config = prompt_config

    # Process the dataset
    metrics = synthesizer.process_dataset(train_dataset)

    # Save config information
    config_info = {
        "model": asdict(model_config),
        "dataset": {"path": data_path, "sample_size": sample_size},
        "metrics": metrics,
    }

    # Remove API key from saved config
    if "api_key" in config_info["model"]:
        config_info["model"]["api_key"] = None

    config_path = os.path.join(output_dir, "synthesis_config.json")
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    if verbose:
        print(f"Synthesis complete. Results saved to {output_dir}")

    return output_dir, metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic explanations for multiple-choice questions using OpenAI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        help="OpenAI model to use for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for model generation (higher = more creative)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens for model generation"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not provided, will use environment variable)",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None, help="Custom system prompt for the model"
    )

    # Dataset configuration
    parser.add_argument(
        "--data-path",
        type=str,
        default="/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_train",
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Number of examples to process (None for all)"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="./synthesis_results", help="Directory to save outputs"
    )

    # Processing configuration
    parser.add_argument(
        "--concurrent-requests", type=int, default=5, help="Number of concurrent API requests"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: No OpenAI API key provided. Please provide it with --api-key or set the OPENAI_API_KEY environment variable."
        )
        exit(1)

    # Run synthesis with parsed arguments
    try:
        output_dir, metrics = run_synthesis(
            model_name=args.model,
            data_path=args.data_path,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrent_requests=args.concurrent_requests,
            random_seed=args.random_seed,
            verbose=not args.quiet,
            system_prompt=args.system_prompt,
        )

        # Print summary
        if not args.quiet:
            print("\n" + "=" * 50)
            print("Synthesis Summary:")
            print(f"Model: {args.model}")
            print(f"Dataset: {args.data_path}")
            print(f"Sample size: {args.sample_size if args.sample_size else 'All'}")
            print(f"Output directory: {output_dir}")
            print(
                f"Success rate: {metrics['successful_generations']/metrics['total_examples']:.2%}"
            )
            print(
                f"Answer preservation rate: {metrics['correct_answer_preserved']/metrics['total_examples']:.2%}"
            )
            print(f"Average response time: {metrics['avg_response_time']:.2f}s")
            print("=" * 50)

    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback

        traceback.print_exc()
