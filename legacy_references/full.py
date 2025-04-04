# %% [markdown]
# # Dataset Preprocessing to Hugging Face Datasets 's Parquet Format

# %% [markdown]
# * In the phase of data preprocessing, we conducted in several notebooks.
# * This part is the code I sum

# %%
## Phase 1: Data Loading and Transformation utilizing the Hugging Face Datasets library

import datasets

# loading and transforming the train data
train_data = datasets.load_dataset(
    "csv",
    data_files="/teamspace/studios/this_studio/workspace_1/data/raw/b6_train_data.csv",
    split="train",
)
choices = train_data["choices"]
list_choices = [eval(choice) for choice in choices]
train_data = train_data.add_column("list_choices", list_choices)
train_data = train_data.add_column("num_choices", [len(choice) for choice in list_choices])

# save the full training data
train_data.save_to_disk(
    "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/train_data"
)

# shuffle the train_data
sf_train_data = train_data.shuffle(seed=42)

# Split the dataset into train and validation sets
dict_data = sf_train_data.train_test_split(test_size=0.1, seed=42)

# split_train, split_val = dict_data['train'], dict_data['test']
split_train, split_val = dict_data["train"], dict_data["test"]

# dump the split_train and split_val to /teamspace/studios/this_studio/workspace_1/data/raw/parquet_format
split_train.save_to_disk(
    "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_train"
)
split_val.save_to_disk(
    "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_val"
)

# %% [markdown]
# # Script for generate synthetic YAML-based reasoning dataset from a teacher model GPT-4o
# - **Synthesis is published at https://huggingface.co/datasets/tuandunghcmut/normal_dataset**
# - **Just for summary code, do not run below cell in this notebook. Please run the cells of training part.**
# - I create this notebook for showing my training and inference process.
# - The name is `normal_dataset` because it is sensitive Kaggle competition dataset.
# - Please inspect the `yml_str` column to see the YAML-based reasoning dataset.
# - Please toggle to turn off this below cell.

import json

# %%
# refactor.py
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

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


# # Example usage
# if __name__ == "__main__":
#     # Create the special validation dataset
#     special_val_dataset = create_special_validation_dataset(
#         num_samples=10,  # Sample 10 examples
#         random_seed=42,   # Fixed seed for reproducibility
#         output_dir="./dataset_info",
#         save_dataset=True,
#         verbose=True
#     )

#     # Now you can use special_val_dataset with the MCTestingFramework
#     print(f"\nTotal unique task_ids: {len(set([ex['task_id'] for ex in special_val_dataset]))}")

#     # Basic example of using the framework
#     # Uncomment to run a test
#     # """
#     from config import OPENAI_API_KEY

#     # 1. Set up configurations
#     model_config = ModelConfig(
#         name="gpt-3.5-turbo",
#         temperature=0.0,
#         max_tokens=1024,
#         api_key=OPENAI_API_KEY
#     )

#     # 2. Create custom prompt template
#     custom_template = '''
#     QUESTION:
#     {{question}}

#     ANSWER CHOICES:
#     {{choices}}

#     Take your time to think through this problem step-by-step.
#     First, analyze the question carefully.
#     Then, consider each choice one by one.
#     Explain your reasoning clearly.

#     Finally, provide your answer in this format:
#     REASONING: <your detailed reasoning>
#     ANSWER: <letter of your answer>
#     '''

#     prompt_config = PromptConfig(
#         template_type="custom",
#         custom_template=custom_template,
#         yaml_format=False
#     )

#     # 3. Configure the test run
#     test_config = TestConfig(
#         output_dir="./custom_test_results",
#         sample_size=2,
#         batch_size=5,
#         save_individual_examples=True,
#         verbose=True
#     )

#     # 4. Initialize and run the framework
#     tester = MCTestingFramework(
#         model_config=model_config,
#         prompt_config=prompt_config,
#         test_config=test_config
#     )

#     # 5. Run the test
#     results = tester.evaluate_dataset(special_val_dataset)
#     # """

import concurrent.futures
import json

# teacher_synthesis.py
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# # Import the necessary classes from refactor.py
# from refactor import (
#     ModelConfig,
#     PromptConfig,
#     TestConfig,
#     MCTestingFramework,
#     create_special_training_dataset
# )


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
):
    """
    Run the synthesis process to generate explanations on training data.

    Args:
        model_name: Name of the OpenAI model to use
        data_path: Path to the training dataset
        sample_size: Number of examples to process
        output_dir: Directory to save outputs
        api_key: OpenAI API key (None to use environment variable)
    """
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join(output_dir, f"{model_name.replace('-', '_')}_{timestamp}")
    output_dir = (
        "/teamspace/studios/this_studio/workspace_1/full_synthesis_results/gpt_4o_20250329_151040"
    )
    # Set up model config
    model_config = ModelConfig(
        name=model_name,
        temperature=0.2,  # Slight temperature for variety in explanations
        max_tokens=2048,  # Longer context for detailed explanations
        api_key=api_key,
        system_prompt="You are an expert teacher creating high-quality, concise explanations for multiple choice questions. Focus on the most important concepts and key distinctions between choices. Be thorough but efficient with your explanations - prioritize clarity and precision over verbosity.",
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
    print(f"Loading training dataset from {data_path}...")

    # For None sample_size, we need to first load the dataset to get its size
    # and then process everything
    if sample_size is None:
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(data_path)
            actual_sample_size = len(dataset)
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
        random_seed=42,
        save_dataset=False,
        verbose=True,
    )

    if not train_dataset:
        print("Failed to load dataset. Exiting.")
        return

    # Create and run the synthesis framework with the custom prompt config
    synthesizer = TeacherSynthesisFramework(
        model_config=model_config,
        output_dir=output_dir,
        concurrent_requests=5,  # Adjust based on API rate limits
        sample_size=actual_sample_size,  # Use the validated sample size
        verbose=True,
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

    print(f"Synthesis complete. Results saved to {output_dir}")


# if __name__ == "__main__":
#     # Example usage - uncomment to run
#     try:
#         # Import API key
#         from config import OPENAI_API_KEY

#         # Run synthesis
#         run_synthesis(
#             model_name="gpt-4o",  # Use GPT-4o for high quality explanations
#             sample_size=None,  # None means process the entire dataset
#             output_dir="./full_synthesis_results",
#             api_key=OPENAI_API_KEY
#         )
#     except ImportError:
#         print("Please provide an OpenAI API key in config.py or as an environment variable.")

# %%
## Phase 3: Preprocessing the synthesis results

import os

import yaml


def load_yaml_from_string(yaml_string, file_name):
    # Remove the ```yaml at the start if present
    if yaml_string.strip().startswith("```yaml"):
        yaml_string = yaml_string[len("```yaml") :].strip()
    # Remove the ``` at the end if present
    if yaml_string.strip().endswith("```"):
        yaml_string = yaml_string[: yaml_string.rfind("```")].strip()

    # Load the YAML content into a Python object
    try:
        yaml_object = yaml.safe_load(yaml_string)

        # Check if yaml_object is a dictionary, if not create a new dictionary
        if not isinstance(yaml_object, dict):
            # yaml_object = {}
            return None
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        return None
    yaml_object_return = {}
    # Now we can safely add keys to the dictionary
    yaml_object_return["yaml_str"] = yaml_string
    yaml_object_return["dataset_index"] = int(file_name.split("/")[-1].split("_")[0])
    return yaml_object_return


def load_yaml_from_file(file_path):
    with open(file_path, "r") as file:
        yaml_content = file.read()
    return load_yaml_from_string(yaml_content, file_path)


# Example directory path
directory = (
    "/teamspace/studios/this_studio/workspace_1/full_synthesis_results/gpt_4o_20250329_151040"
)

# Load a specific YAML file from the dataset directory
file_path = os.path.join(directory, "000000_rt01317.yaml")
yaml_object = load_yaml_from_file(file_path)
print(yaml_object)
# Load all YAML files from the directory and combine into a list of objects
yaml_objects = []
for filename in os.listdir(directory):
    if filename.endswith(".yaml"):
        file_path = os.path.join(directory, filename)
        obj = load_yaml_from_file(file_path)
        if obj is not None:
            yaml_objects.append(obj)
print(f"Loaded {len(yaml_objects)} YAML files")

# Remove None values
yaml_objects = [obj for obj in yaml_objects if obj is not None]
print(f"Filtered {len(yaml_objects)} None values")

# Convert the list of dictionaries to a datasets.Dataset
import datasets

yml_dataset = datasets.Dataset.from_list(yaml_objects)
yml_dataset

index_yml_dataset = list(yml_dataset["dataset_index"])

split_train = split_train.select(index_yml_dataset)


def map(example, index):
    return {
        "task_id": example["task_id"],
        "question": example["question"],
        "choices": example["choices"],
        "answer": example["answer"],
        "yml_str": yml_dataset[index]["yaml_str"],
    }


split_train = split_train.map(map, with_indices=True)
split_train


# Function to clean the answer field by removing "ANSWER: " prefix
def clean_answer(example):
    if (
        "answer" in example
        and example["answer"] is not None
        and example["answer"].startswith("ANSWER: ")
    ):
        example["answer"] = example["answer"][8:]  # Remove "ANSWER: " prefix
    return example


# Apply the cleaning function to both datasets
print("Cleaning answer field in train dataset...")
split_train = split_train.map(clean_answer)

print("Cleaning answer field in validation dataset...")
# split_val_cleaned = split_val.map(clean_answer)

# dump split_train to dataset_ready_for_training
split_train.save_to_disk(
    "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/dataset_ready_for_training"
)


# %%


# %% [markdown]
# # Training Part (Recommended to run on a GPU)

# %% [markdown]
# - **Latest training model is automatically saved in the `tuandunghcmut/Qwen25_Coder_MultipleChoice` repository.**
# - **In my notebook, there are some cells for inference on some examples after loading the latest model from the Hugging Face repository. I claim that the model is working well on the dataset. You can check the inference part in the notebook.**
# - Unfortunately, I haven't conducted the experiment on the full test dataset yet, due to the time limit. I haven minimized my effort in this notebook to submit a minimal version of my working. In fact, I can't control the short time limit of the competition.
# - At the first time, I was so excited to participate in this competition. I invested money and time on dataset synthesis and GPU time training. I wanted to make it at a good project with good structure. But there are some issues that I can't control.
# - But at least, the model is working. Only one thing is that I'm in progress of inference on the whole full test dataset.
# - I will be very happy if you could give me any feedback or suggestion. Thank you for your time and attention.
#
#
#
#

import json

# %%
import os
import time
import warnings
from datetime import datetime

import pandas as pd
from datasets import Dataset, load_from_disk

# Suppress warnings
warnings.filterwarnings("ignore")

# Import Unsloth
import unsloth

# Import Wandb for experiment tracking
import wandb

# Import HuggingFace libraries

# Try to import HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")

# Disable HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%


# %%


# %%
class PromptCreator:
    """
    Creates and formats prompts for multiple choice questions
    Supports different prompt styles for training and inference
    """

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = (
        "teacher"  # Same YAML format as YAML_REASONING but using teacher completions for training
    )

    def __init__(self, prompt_type=BASIC):
        """
        Initialize prompt creator with the specified type

        Args:
            prompt_type: Type of prompts to generate - "basic", "yaml", or "teacher"
                         Note: "teacher" uses same prompt format as "yaml" but with teacher completions
        """
        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        # The difference only matters during training when using teacher completions
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        # Store the original prompt type to track if we're using teacher mode
        self.original_type = prompt_type

    def format_choices(self, choices):
        """Format choices as a lettered list"""
        return "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])

    def get_max_letter(self, choices):
        """Get the maximum letter based on number of choices"""
        return chr(65 + len(choices) - 1)

    def create_inference_prompt(self, question, choices):
        """
        Create a prompt for inference based on current prompt type

        Args:
            question: The question text
            choices: List of choices

        Returns:
            Formatted prompt string
        """
        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        if self.prompt_type == self.YAML_REASONING:
            return self._create_yaml_prompt(question, formatted_choices, max_letter)
        else:
            return self._create_basic_prompt(question, formatted_choices, max_letter)

    def _create_basic_prompt(self, question, formatted_choices, max_letter):
        """Create a basic prompt asking for just the answer letter"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Answer with a single letter from A through {max_letter} without any additional explanation or commentary.
"""

    def _create_yaml_prompt(self, question, formatted_choices, max_letter):
        """Create a prompt requesting YAML-formatted reasoning"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Your response MUST be in YAML format as follows:

understanding: |
  <your understanding of what the question is asking>
analysis: |
  <your analysis of each option>
reasoning: |
  <your step-by-step reasoning process>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>

The answer field MUST contain ONLY a single character letter.
"""

    def create_training_prompt(self, question, choices):
        """
        Create a prompt for training with the current prompt type

        Args:
            question: The question text
            choices: List of choices

        Returns:
            Formatted prompt string for training
        """
        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        if self.prompt_type == self.YAML_REASONING:
            return self._create_yaml_training_prompt(question, formatted_choices, max_letter)
        else:
            return self._create_basic_training_prompt(question, formatted_choices, max_letter)

    def _create_basic_training_prompt(self, question, formatted_choices, max_letter):
        """Create a basic training prompt"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

The answer is a single letter (A, B, C, etc.). Only provide ONE character as your answer:
"""

    def _create_yaml_training_prompt(self, question, formatted_choices, max_letter):
        """Create a YAML-formatted training prompt"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Follow the YAML format in your response:

understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>
"""

    def set_prompt_type(self, prompt_type):
        """Set the prompt type"""
        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        self.original_type = prompt_type  # Store the original type

        if prompt_type == self.TEACHER_REASONED:
            # prompt_type = self.YAML_REASONING
            pass

        self.prompt_type = prompt_type
        return self

    def is_teacher_mode(self):
        """Check if we're using teacher mode (for training with teacher completions)"""
        return self.original_type == self.TEACHER_REASONED


# %%
class QwenModelHandler:
    """Handler for Qwen models with inference and saving capabilities using Unsloth"""

    def __init__(
        self,
        model_name="unsloth/Qwen2.5-7B",
        max_seq_length=768,
        quantization=None,
        device_map="auto",
        cache_dir=None,
    ):
        """
        Initialize model and tokenizer using Unsloth

        Args:
            model_name: Name or path of the model (preferably an unsloth model)
            max_seq_length: Maximum sequence length for the model
            quantization: Quantization type (None, '4bit', '8bit') - for compatibility
            device_map: Device mapping strategy
            cache_dir: Cache directory for models
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self.quantization = quantization
        self.cache_dir = cache_dir

        # Convert quantization parameter to load_in_4bit parameter for Unsloth
        self.load_in_4bit = quantization == "4bit"

        # Load tokenizer and model
        self.tokenizer, self.model = self._load_model()
        self.response_parser = ResponseParser()

    def _load_model(self):
        """Load model and tokenizer with Unsloth for optimization"""
        import torch
        from unsloth import FastLanguageModel

        print(f"Loading {self.model_name} with Unsloth, max_seq_length={self.max_seq_length}")

        # Set dtype based on hardware
        dtype = None  # None for auto detection

        # Load model and tokenizer with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir=self.cache_dir,
        )

        return tokenizer, model

    def generate_with_streaming(self, prompt, temperature=0.7, max_tokens=1024, stream=True):
        """
        Generate completion with optional streaming using Unsloth's optimized inference
        """
        # Enable faster inference
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        # Generate with streaming if requested
        if stream:
            import threading

            from transformers import TextIteratorStreamer

            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Start generation in a thread
            generation_kwargs = {
                "input_ids": model_inputs.input_ids,
                "attention_mask": model_inputs.attention_mask,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
                "do_sample": temperature > 0.0,
                "use_cache": True,  # Important for Unsloth performance
                "min_p": 0.1
                if temperature > 0.0
                else None,  # Optional: Unsloth recommends this for better quality
            }

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Return the streamer that yields text chunks
            return streamer
        else:
            # Generate without streaming
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                use_cache=True,  # Important for Unsloth performance
                min_p=0.1 if temperature > 0.0 else None,  # Optional: Unsloth recommends this
            )

            # Decode the generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0][model_inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            return generated_text

    def calculate_perplexity(self, prompt, answer, temperature=0.0):
        """
        Calculate perplexity for a prompt and answer pair

        Args:
            prompt: The input prompt
            answer: The expected answer
            temperature: Sampling temperature

        Returns:
            Perplexity score
        """
        import torch

        # Format chat for perplexity calculation
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the text
        encodings = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)

        # Get loss and calculate perplexity
        neg_log_likelihood = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

        return perplexity

    def save_model(self, output_dir, save_method="lora"):
        """
        Save model to disk using Unsloth's optimized methods

        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_16bit", "merged_4bit", "gguf")
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Use Unsloth's saving methods
        if save_method == "lora":
            # Save LoRA weights
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            # Save merged model in float16
            self.model.save_pretrained_merged(
                output_dir, self.tokenizer, save_method="merged_16bit"
            )
        elif save_method == "merged_4bit":
            # Save merged model in 4bit
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_4bit")
        elif save_method == "gguf":
            # Save in GGUF format for llama.cpp
            self.model.save_pretrained_gguf(
                output_dir, self.tokenizer, quantization_method="q4_k_m"
            )
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        print(f"Model saved to {output_dir} using method {save_method}")
        return output_dir

    def push_to_hub(self, repo_id, token=None, save_method="lora", private=False):
        """
        Push model to Hugging Face Hub using Unsloth's optimized methods
        """
        # Use Unsloth's hub methods directly
        if save_method == "lora":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="lora", token=token)
        elif save_method == "merged_16bit":
            self.model.push_to_hub_merged(
                repo_id, self.tokenizer, save_method="merged_16bit", token=token
            )
        elif save_method == "merged_4bit":
            self.model.push_to_hub_merged(
                repo_id, self.tokenizer, save_method="merged_4bit", token=token
            )
        elif save_method == "gguf":
            # Push multiple GGUF variants
            self.model.push_to_hub_gguf(
                repo_id, self.tokenizer, quantization_method=["q4_k_m", "q5_k_m"], token=token
            )
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"


# %%
class QwenTrainer:
    """Training handler for Qwen models with optional HuggingFace Hub integration"""

    def __init__(
        self,
        model,
        tokenizer,
        prompt_creator=None,
        lora_config=None,
        hub_token=None,
        hub_model_id=None,
    ):
        """
        Initialize the trainer with model, tokenizer and optional LoRA config

        Args:
            model: The model to fine-tune
            tokenizer: The tokenizer for the model
            prompt_creator: Optional PromptCreator for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient fine-tuning
            hub_token: Optional HuggingFace Hub token for pushing models
            hub_model_id: Optional model ID for pushing to HuggingFace Hub
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_creator = prompt_creator or PromptCreator(PromptCreator.BASIC)
        self.lora_config = lora_config
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        self.peft_model = None

        # Ensure we have a proper max sequence length
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_seq_length = min(2048, self.model.config.max_position_embeddings)
        else:
            self.max_seq_length = 2048  # Default fallback

    def prepare_model_for_training(self):
        """Apply Unsloth's LoRA configuration instead of PEFT"""
        if self.lora_config:
            from unsloth import FastLanguageModel

            # Extract parameters from the PEFT LoraConfig
            r = self.lora_config.r
            lora_alpha = self.lora_config.lora_alpha
            lora_dropout = self.lora_config.lora_dropout
            target_modules = self.lora_config.target_modules

            # Use Unsloth's LoRA implementation
            self.peft_model = FastLanguageModel.get_peft_model(
                self.model,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            # Print parameters info
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            trainable_params = sum(
                p.numel() for p in self.peft_model.parameters() if p.requires_grad
            )
            print(
                f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}"
            )

            return self.peft_model
        return self.model

    def prepare_dataset(self, dataset, prompt_type=None, verbose=False):
        """Prepare dataset for SFTTrainer by returning text field"""
        # Temporarily set prompt type if provided
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.prompt_type = prompt_type

        if verbose:
            print(f"Preparing dataset with prompt type: {self.prompt_creator.prompt_type}")
            if self.prompt_creator.is_teacher_mode() and "yml_str" not in dataset.features:
                print("WARNING: Teacher mode enabled but 'yml_str' field not found in dataset")

        def format_example(example):
            """Transform function applied to each example during training"""
            questions = example["question"]
            choices = example["choices"]
            answer = example["answer"]

            # For teacher mode, use the yml_str if available
            assistant_response = None
            if self.prompt_creator.is_teacher_mode() and "yml_str" in example:
                assistant_response = example["yml_str"]

            # Call create_training_prompt without task_id parameter
            user_prompt = self.prompt_creator.create_training_prompt(
                question=questions, choices=choices
            )

            if assistant_response is None:
                # Default to simple answer format if no teacher completion available
                assistant_response = answer

            # Apply chat template for training
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            # Return text field for SFTTrainer
            return {"text": text}

        # Use map instead of with_transform for reliable transformation
        columns_to_remove = [
            col
            for col in ["task_id", "question", "choices", "answer", "yml_str"]
            if col in dataset.features
        ]

        transformed_dataset = dataset.map(
            format_example, remove_columns=columns_to_remove, batched=False
        )

        # Preview the transformed data
        if verbose:
            print("Preview of transformed data:")
            print(f"Keys: {list(transformed_dataset[0].keys())}")
            sample_text = transformed_dataset[0]["text"]
            sample_length = len(self.tokenizer.encode(sample_text))
            print(f"Text sample: {sample_text[:100]}...")
            print(f"Encoded length of first sample: {sample_length} tokens")

            # Check for potential length issues
            if sample_length > self.max_seq_length:
                print(
                    f"WARNING: Sample exceeds max sequence length ({sample_length} > {self.max_seq_length})"
                )

        # Restore original prompt type if changed
        if original_prompt_type is not None:
            self.prompt_creator.prompt_type = original_prompt_type

        return transformed_dataset

    def train(
        self,
        dataset,
        prompt_type=None,
        output_dir="./model_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        max_steps=None,
        learning_rate=2e-4,
        train_on_inputs=True,
        packing=False,
        logging_steps=10,
        save_steps=100,
        verbose=True,
        push_to_hub=False,
    ):
        """Train the model using Unsloth's optimized training"""
        # Prepare dataset with on-the-fly transformation
        prepared_dataset = self.prepare_dataset(dataset, prompt_type, verbose)

        # Import Unsloth's optimized trainer and utilities
        import os

        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import is_bfloat16_supported

        # Prepare the model with Unsloth's LoRA
        model_to_train = self.prepare_model_for_training()

        # Setup training arguments with proper handling of max_steps
        training_args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "optim": "paged_adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
            "push_to_hub": push_to_hub,
            "hub_model_id": self.hub_model_id if push_to_hub else None,
            "hub_token": self.hub_token if push_to_hub else None,
        }

        # Only add max_steps if it's not None
        if max_steps is not None and max_steps > 0:
            training_args_dict["max_steps"] = max_steps

        # Create TrainingArguments with the prepared dictionary
        training_args = TrainingArguments(**training_args_dict)

        # Use SFTTrainer with Unsloth-specific settings
        trainer = SFTTrainer(
            model=model_to_train,
            tokenizer=self.tokenizer,
            train_dataset=prepared_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,  # Use our tracked max sequence length
            args=training_args,
            packing=False,  # Set packing to False to avoid shape issues
        )

        if verbose:
            # Log dataset and training info
            train_size = len(prepared_dataset)
            steps_per_epoch = train_size // (
                per_device_train_batch_size * gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * num_train_epochs if max_steps is None else max_steps

            print(f"Training with dataset size: {train_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total training steps: {total_steps}")
            print(f"Max sequence length: {self.max_seq_length}")
            print(f"Push to HuggingFace Hub: {push_to_hub}")
            if push_to_hub:
                print(f"Hub model ID: {self.hub_model_id}")

        # Start training
        trainer_stats = trainer.train()

        # Update self.model to the fine-tuned version
        self.model = model_to_train

        return trainer_stats

    def save_results(self, results, output_dir="./results"):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results with max sequence length included
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
            "max_sequence_length": self.max_seq_length,  # Track max sequence length
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast

                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    # Keep as-is if conversion fails
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json

            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file


# %%
class ResponseParser:
    """
    Parser for model responses with support for different formats
    Extracts answers and reasoning from model outputs
    """

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"  # Parse YAML formatted response with reasoning

    def __init__(self, parser_mode=BASIC):
        """
        Initialize with specified parser mode

        Args:
            parser_mode: Mode to use for parsing - "basic" or "yaml"
        """
        self.parser_mode = parser_mode

    def parse(self, response_text):
        """
        Parse the model's response according to the current mode

        Args:
            response_text: Raw response text from the model

        Returns:
            Tuple of (answer, reasoning)
        """
        if self.parser_mode == self.YAML:
            return self._parse_yaml_response(response_text)
        else:
            return self._parse_basic_response(response_text)

    def _parse_basic_response(self, response_text):
        """
        Parse basic response looking for a letter answer

        For basic mode, we look for a single letter (A-Z) with minimal reasoning
        """
        import re

        # Try to extract a single letter answer (A-Z)
        answer_match = re.search(r"(?:^|\s)([A-Z])(?:\s|$|\.)", response_text)
        if answer_match:
            answer = answer_match.group(1)
        else:
            # Take first character if it's a letter
            if response_text and response_text[0].isalpha():
                answer = response_text[0].upper()
            else:
                answer = None

        # For basic mode, we don't extract detailed reasoning
        reasoning = ""

        return answer, reasoning

    def _parse_yaml_response(self, response_text):
        """
        Parse YAML formatted response extracting answer and reasoning

        For YAML mode, we try to extract both the answer and structured reasoning
        """
        import re

        import yaml

        # First try to find answer in YAML format
        yaml_match = re.search(r"answer:\s*([A-Z])", response_text)
        if yaml_match:
            answer = yaml_match.group(1)
        else:
            # Fall back to basic extraction if YAML parsing fails
            answer_match = re.search(r"(?:^|\s)([A-Z])(?:\s|$|\.)", response_text)
            if answer_match:
                answer = answer_match.group(1)
            elif response_text and response_text[0].isalpha():
                answer = response_text[0].upper()
            else:
                answer = None

        # Try to parse reasoning from YAML format
        reasoning = ""
        if "reasoning:" in response_text:
            yaml_content = yaml.safe_load("---\n" + response_text)
            if isinstance(yaml_content, dict) and "reasoning" in yaml_content:
                reasoning = yaml_content["reasoning"]

                # Add other YAML fields if available
                if "understanding" in yaml_content:
                    reasoning = f"Understanding: {yaml_content['understanding']}\n\n{reasoning}"
                if "conclusion" in yaml_content:
                    reasoning = f"{reasoning}\n\nConclusion: {yaml_content['conclusion']}"
        else:
            # Use the full response as reasoning if not in YAML format
            reasoning = response_text

        return answer, reasoning

    def set_parser_mode(self, parser_mode):
        """Set the parser mode"""
        self.parser_mode = parser_mode
        return self

    @classmethod
    def from_prompt_type(cls, prompt_type):
        """
        Create a parser instance with mode matching the prompt type

        Args:
            prompt_type: Prompt type from PromptCreator

        Returns:
            ResponseParser instance with appropriate mode
        """
        if (
            prompt_type == PromptCreator.YAML_REASONING
            or prompt_type == PromptCreator.TEACHER_REASONED
        ):
            return cls(parser_mode=cls.YAML)
        else:
            return cls(parser_mode=cls.BASIC)


# %%
class QwenTrainer:
    """Training handler for Qwen models with optional HuggingFace Hub integration"""

    def __init__(
        self,
        model,
        tokenizer,
        prompt_creator=None,
        lora_config=None,
        hub_token=None,
        hub_model_id=None,
    ):
        """
        Initialize the trainer with model, tokenizer and optional LoRA config

        Args:
            model: The model to fine-tune
            tokenizer: The tokenizer for the model
            prompt_creator: Optional PromptCreator for formatting prompts
            lora_config: Optional LoRA configuration for parameter-efficient fine-tuning
            hub_token: Optional HuggingFace Hub token for pushing models
            hub_model_id: Optional model ID for pushing to HuggingFace Hub
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_creator = prompt_creator or PromptCreator(PromptCreator.BASIC)
        self.lora_config = lora_config
        self.hub_token = hub_token
        self.hub_model_id = hub_model_id
        self.peft_model = None

        # Ensure we have a proper max sequence length
        if hasattr(self.model.config, "max_position_embeddings"):
            self.max_seq_length = min(2048, self.model.config.max_position_embeddings)
        else:
            self.max_seq_length = 2048  # Default fallback

    def prepare_model_for_training(self):
        """Apply Unsloth's LoRA configuration instead of PEFT"""
        if self.lora_config:
            from unsloth import FastLanguageModel

            # Extract parameters from the PEFT LoraConfig
            r = self.lora_config.r
            lora_alpha = self.lora_config.lora_alpha
            lora_dropout = self.lora_config.lora_dropout
            target_modules = self.lora_config.target_modules

            # Use Unsloth's LoRA implementation
            self.peft_model = FastLanguageModel.get_peft_model(
                self.model,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            # Print parameters info
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            trainable_params = sum(
                p.numel() for p in self.peft_model.parameters() if p.requires_grad
            )
            print(
                f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}"
            )

            return self.peft_model
        return self.model

    def prepare_dataset(self, dataset, prompt_type=None, verbose=False):
        """Prepare dataset for SFTTrainer by returning text field"""
        # Temporarily set prompt type if provided
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.prompt_type = prompt_type

        if verbose:
            print(f"Preparing dataset with prompt type: {self.prompt_creator.prompt_type}")
            if self.prompt_creator.is_teacher_mode() and "yml_str" not in dataset.features:
                print("WARNING: Teacher mode enabled but 'yml_str' field not found in dataset")

        def format_example(example):
            """Transform function applied to each example during training"""
            questions = example["question"]
            choices = example["choices"]
            answer = example["answer"]

            # For teacher mode, use the yml_str if available
            assistant_response = None
            if self.prompt_creator.is_teacher_mode() and "yml_str" in example:
                assistant_response = example["yml_str"]

            # Call create_training_prompt without task_id parameter
            user_prompt = self.prompt_creator.create_training_prompt(
                question=questions, choices=choices
            )

            if assistant_response is None:
                # Default to simple answer format if no teacher completion available
                assistant_response = answer

            # Apply chat template for training
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            # Return text field for SFTTrainer
            return {"text": text}

        # Use map instead of with_transform for reliable transformation
        columns_to_remove = [
            col
            for col in ["task_id", "question", "choices", "answer", "yml_str"]
            if col in dataset.features
        ]

        transformed_dataset = dataset.map(
            format_example, remove_columns=columns_to_remove, batched=False
        )

        # Preview the transformed data
        if verbose:
            print("Preview of transformed data:")
            print(f"Keys: {list(transformed_dataset[0].keys())}")
            sample_text = transformed_dataset[0]["text"]
            sample_length = len(self.tokenizer.encode(sample_text))
            print(f"Text sample: {sample_text[:100]}...")
            print(f"Encoded length of first sample: {sample_length} tokens")

            # Check for potential length issues
            if sample_length > self.max_seq_length:
                print(
                    f"WARNING: Sample exceeds max sequence length ({sample_length} > {self.max_seq_length})"
                )

        # Restore original prompt type if changed
        if original_prompt_type is not None:
            self.prompt_creator.prompt_type = original_prompt_type

        return transformed_dataset

    def train(
        self,
        dataset,
        prompt_type=None,
        output_dir="./model_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        max_steps=None,
        learning_rate=2e-4,
        train_on_inputs=True,
        packing=False,
        logging_steps=10,
        save_steps=100,
        verbose=True,
        push_to_hub=False,
    ):
        """Train the model using Unsloth's optimized training"""
        # Prepare dataset with on-the-fly transformation
        prepared_dataset = self.prepare_dataset(dataset, prompt_type, verbose)

        # Import Unsloth's optimized trainer and utilities
        import os

        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import is_bfloat16_supported

        # Prepare the model with Unsloth's LoRA
        model_to_train = self.prepare_model_for_training()

        # Setup training arguments with proper handling of max_steps
        training_args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "warmup_steps": warmup_steps,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "fp16": not is_bfloat16_supported(),
            "bf16": is_bfloat16_supported(),
            "optim": "paged_adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "report_to": "wandb" if os.environ.get("WANDB_PROJECT") else "none",
            "push_to_hub": push_to_hub,
            "hub_model_id": self.hub_model_id if push_to_hub else None,
            "hub_token": self.hub_token if push_to_hub else None,
        }

        # Only add max_steps if it's not None
        if max_steps is not None and max_steps > 0:
            training_args_dict["max_steps"] = max_steps

        # Create TrainingArguments with the prepared dictionary
        training_args = TrainingArguments(**training_args_dict)

        # Use SFTTrainer with Unsloth-specific settings
        trainer = SFTTrainer(
            model=model_to_train,
            tokenizer=self.tokenizer,
            train_dataset=prepared_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,  # Use our tracked max sequence length
            args=training_args,
            packing=False,  # Set packing to False to avoid shape issues
        )

        if verbose:
            # Log dataset and training info
            train_size = len(prepared_dataset)
            steps_per_epoch = train_size // (
                per_device_train_batch_size * gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * num_train_epochs if max_steps is None else max_steps

            print(f"Training with dataset size: {train_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Total training steps: {total_steps}")
            print(f"Max sequence length: {self.max_seq_length}")
            print(f"Push to HuggingFace Hub: {push_to_hub}")
            if push_to_hub:
                print(f"Hub model ID: {self.hub_model_id}")

        # Start training
        trainer_stats = trainer.train()

        # Update self.model to the fine-tuned version
        self.model = model_to_train

        return trainer_stats

    def save_results(self, results, output_dir="./results"):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results with max sequence length included
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
            "max_sequence_length": self.max_seq_length,  # Track max sequence length
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast

                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    # Keep as-is if conversion fails
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json

            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file


# %%


# %%
def create_default_dataset(
    path=None, parquet_file=None, task_ids=None, limit=None, output_dir=None, verbose=False
):
    """
    Load or create default dataset from disk or parquet files

    Args:
        path: Optional path to dataset saved to disk
        parquet_file: Optional path to parquet file
        task_ids: Optional list of task IDs to filter by
        limit: Optional limit on number of examples
        output_dir: Optional output directory to save task IDs
        verbose: Whether to print verbose output

    Returns:
        Loaded or created dataset
    """
    from datasets import Dataset, load_from_disk

    # First try loading from parquet if provided
    if parquet_file:
        if verbose:
            print(f"Loading dataset from parquet file: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        dataset = Dataset.from_pandas(df)
    # Otherwise load from disk if path provided
    elif path:
        if verbose:
            print(f"Loading dataset from disk: {path}")
        dataset = load_from_disk(path)
    else:
        if verbose:
            print("No dataset path or parquet file provided. Returning empty dataset.")
        return Dataset.from_dict({"question": [], "choices": [], "answer": []})

    # Filter by task IDs if provided
    if task_ids:
        if "task_id" in dataset.features:
            if verbose:
                print(f"Filtering dataset to {len(task_ids)} task IDs")
            dataset = dataset.filter(lambda x: x["task_id"] in task_ids)
        else:
            print("WARNING: 'task_id' field not found in dataset but task_ids filter was provided")

    # Apply limit if provided
    if limit and limit < len(dataset):
        if verbose:
            print(f"Limiting dataset to {limit} examples (from {len(dataset)})")
        dataset = dataset.select(range(limit))

    # Print dataset statistics
    if verbose:
        print(f"Dataset features: {dataset.features}")
        print(f"Dataset size: {len(dataset)} examples")

    # Save task IDs to output directory if provided
    if output_dir and "task_id" in dataset.features:
        import os

        os.makedirs(output_dir, exist_ok=True)
        task_ids_path = os.path.join(output_dir, "task_ids.txt")
        with open(task_ids_path, "w") as f:
            for task_id in dataset["task_id"]:
                f.write(f"{task_id}\n")
        if verbose:
            print(f"Saved {len(dataset)} task IDs to {task_ids_path}")

    return dataset


# %%
class EarlyStopping:
    """
    Early stopping handler to monitor validation performance and stop training when necessary
    """

    def __init__(self, patience=3, min_delta=0.001):
        """
        Initialize early stopping handler

        Args:
            patience: Number of epochs without improvement to stop training
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, val_score, model=None):
        """
        Check if training should stop

        Args:
            val_score: Current validation metric (higher is better)
            model: Model to save if performance improves

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            if model is not None:
                self.best_state = model.state_dict()
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            if model is not None:
                self.best_state = model.state_dict()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# %%


# %%
class MultipleChoiceTester:
    """Framework for testing Qwen models on multiple choice questions"""

    def __init__(self, model_handler, prompt_creator=None):
        """
        Initialize with model handler and prompt configuration

        Args:
            model_handler: The QwenModelHandler instance
            prompt_creator: Optional PromptCreator instance (will create one if not provided)
        """
        self.model_handler = model_handler
        self.prompt_creator = prompt_creator or PromptCreator(PromptCreator.BASIC)
        # Create a response parser matching the prompt type
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)

    def infer_example(
        self, example, temperature=0.7, max_tokens=1024, prompt_type=None, stream=False
    ):
        """
        Mode 1: Inference on a single example for visualization/demonstration

        Args:
            example: Single example to infer (dict with question, choices, etc.)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            prompt_type: Optional override for prompt type
            stream: Whether to stream the output

        Returns:
            Dictionary with prediction and metrics
        """
        # Allow temporary override of prompt type
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.set_prompt_type(prompt_type)
            # Update response parser to match prompt type
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Prepare data
        question = example["question"]

        # Handle different formats of choices
        if isinstance(example["choices"], list):
            choices = example["choices"]
        elif isinstance(example["choices"], str) and example["choices"].startswith("["):
            # Parse string representation of list
            import ast

            choices = (
                ast.literal_eval(example["choices"])
                if "[" in example["choices"]
                else example["choices"].split(",")
            )
        else:
            choices = str(example["choices"]).split(",")

        # Generate the prompt using prompt creator
        prompt = self.prompt_creator.create_inference_prompt(question, choices)

        # Start timing
        start_time = time.time()

        if stream:
            # Use streaming generation
            streamer = self.model_handler.generate_with_streaming(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=True
            )

            # Collect output from streamer
            raw_response = ""
            print("Model response:")
            for text_chunk in streamer:
                print(text_chunk, end="", flush=True)
                raw_response += text_chunk
            print("\n")
        else:
            # Generate without streaming
            raw_response = self.model_handler.generate_with_streaming(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=False
            )

        response_time = time.time() - start_time

        # Parse the response using the response parser
        predicted_answer, reasoning = self.response_parser.parse(raw_response)

        # Prepare results
        result = {
            "question": question,
            "choices": choices,
            "predicted_answer": predicted_answer,
            "reasoning": reasoning,
            "response_time": response_time,
            "raw_response": raw_response,
            "prompt_type": self.prompt_creator.prompt_type,
        }

        # Add task_id if available
        if "task_id" in example:
            result["task_id"] = example["task_id"]

        # Calculate metrics if label is provided
        if "answer" in example:
            label = example["answer"]
            result["correct_answer"] = label
            result["is_correct"] = predicted_answer == label

            # Calculate perplexity if requested
            if hasattr(self.model_handler, "calculate_perplexity"):
                perplexity = self.model_handler.calculate_perplexity(prompt, raw_response)
                result["perplexity"] = perplexity

        # Restore original prompt type if it was overridden
        if original_prompt_type is not None:
            self.prompt_creator.set_prompt_type(original_prompt_type)
            # Restore the original response parser
            self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return result

    def infer_batch(
        self, examples, temperature=0.7, max_tokens=1024, prompt_type=None, batch_size=4
    ):
        """
        Mode 2: Inference on a batch of examples

        Args:
            examples: List of examples to infer
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            prompt_type: Optional override for prompt type
            batch_size: Size of batches for processing

        Returns:
            List of result dictionaries and summary metrics
        """
        # Allow temporary override of prompt type
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.set_prompt_type(prompt_type)
            # Update response parser to match prompt type
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Prepare all prompts
        prompts = []
        metadata = []

        for i, example in enumerate(examples):
            # Extract data
            question = example["question"]

            # Handle different formats of choices
            if isinstance(example["choices"], list):
                choices = example["choices"]
            elif isinstance(example["choices"], str) and example["choices"].startswith("["):
                # Parse string representation of list
                import ast

                choices = (
                    ast.literal_eval(example["choices"])
                    if "[" in example["choices"]
                    else example["choices"].split(",")
                )
            else:
                choices = str(example["choices"]).split(",")

            # Generate the prompt using prompt creator
            prompt = self.prompt_creator.create_inference_prompt(question, choices)
            prompts.append(prompt)

            # Store metadata for later
            meta = {
                "question": question,
                "choices": choices,
                "index": i,
            }

            # Add label if available
            if "answer" in example:
                meta["label"] = example["answer"]

            if "task_id" in example:
                meta["task_id"] = example["task_id"]

            metadata.append(meta)

        # Process in batches
        results = []
        correct_count = 0
        total_count = 0
        perplexities = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size]

            # Process batch
            start_time = time.time()
            batch_responses = []

            for prompt in batch_prompts:
                response = self.model_handler.generate_with_streaming(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens, stream=False
                )
                batch_responses.append(response)

            batch_time = time.time() - start_time

            # Process each response in the batch
            for j, (response, meta) in enumerate(zip(batch_responses, batch_meta)):
                # Parse response
                predicted_answer, reasoning = self.response_parser.parse(response)

                # Create result
                result = {
                    "question": meta["question"],
                    "choices": meta["choices"],
                    "predicted_answer": predicted_answer,
                    "reasoning": reasoning,
                    "raw_response": response,
                    "prompt_type": self.prompt_creator.prompt_type,
                    "response_time": batch_time / len(batch_prompts),  # Approximate individual time
                }

                # Add task_id if available
                if "task_id" in meta:
                    result["task_id"] = meta["task_id"]

                # Add metrics if label available
                if "label" in meta:
                    label = meta["label"]
                    result["correct_answer"] = label
                    result["is_correct"] = predicted_answer == label

                    # Update counts for accuracy
                    total_count += 1
                    if result["is_correct"]:
                        correct_count += 1

                    # Calculate perplexity if possible
                    if hasattr(self.model_handler, "calculate_perplexity"):
                        prompt = batch_prompts[j]
                        perplexity = self.model_handler.calculate_perplexity(prompt, response)
                        result["perplexity"] = perplexity
                        perplexities.append(perplexity)

                results.append(result)

        # Calculate aggregate metrics
        summary_metrics = {}
        if total_count > 0:
            summary_metrics["accuracy"] = correct_count / total_count
            summary_metrics["correct_count"] = correct_count
            summary_metrics["total_count"] = total_count

            if perplexities:
                summary_metrics["avg_perplexity"] = sum(perplexities) / len(perplexities)
                summary_metrics["min_perplexity"] = min(perplexities)
                summary_metrics["max_perplexity"] = max(perplexities)

        # Restore original prompt type if it was overridden
        if original_prompt_type is not None:
            self.prompt_creator.set_prompt_type(original_prompt_type)
            # Restore the original response parser
            self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return results, summary_metrics

    def evaluate_dataset(
        self,
        dataset,
        temperature=0.7,
        max_tokens=1024,
        num_examples=None,
        verbose=True,
        prompt_type=None,
        batch_size=4,
        log_to_wandb=False,
    ):
        """
        Mode 3: Inference on a whole dataset with metrics calculation

        Args:
            dataset: Dataset to evaluate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            num_examples: Number of examples to evaluate (None for all)
            verbose: Whether to print progress information
            prompt_type: Override the prompt type for this evaluation
            batch_size: Size of batches for processing
            log_to_wandb: Whether to log results to wandb

        Returns:
            Summary dictionary with results and metrics
        """
        # Allow overriding the prompt type for this evaluation
        original_prompt_type = self.prompt_creator.prompt_type
        if prompt_type is not None:
            self.prompt_creator.set_prompt_type(prompt_type)
            # Update response parser to match prompt type
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Select subset if specified
        if num_examples is not None:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        results = []
        correct_count = 0
        total_count = 0
        perplexities = []

        # Process examples in batches
        for i in range(0, len(dataset), batch_size):
            batch_examples = dataset[i : i + batch_size]

            if verbose:
                batch_desc = (
                    f"Batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}"
                )
                print(f"\nProcessing {batch_desc} with {len(batch_examples)} examples...")

            # Infer batch
            batch_results, batch_metrics = self.infer_batch(
                examples=batch_examples,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
            )

            # Update metrics
            results.extend(batch_results)
            if "correct_count" in batch_metrics:
                correct_count += batch_metrics["correct_count"]
                total_count += batch_metrics["total_count"]

                if verbose:
                    batch_accuracy = batch_metrics["accuracy"]
                    overall_accuracy = correct_count / total_count
                    print(
                        f"Batch accuracy: {batch_accuracy:.2%}, Overall: {overall_accuracy:.2%} ({correct_count}/{total_count})"
                    )

            # Collect perplexities
            if "avg_perplexity" in batch_metrics:
                for result in batch_results:
                    if "perplexity" in result:
                        perplexities.append(result["perplexity"])

        # Calculate final accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        if verbose:
            prompt_type_str = self.prompt_creator.prompt_type
            print(
                f"\nFinal accuracy with {prompt_type_str} prompts: {accuracy:.2%} ({correct_count}/{total_count})"
            )
            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                print(f"Average perplexity: {avg_perplexity:.4f}")

        # Prepare comprehensive summary
        summary = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "prompt_type": self.prompt_creator.prompt_type,
            "results": results,
        }

        # Add perplexity metrics if available
        if perplexities:
            summary["avg_perplexity"] = sum(perplexities) / len(perplexities)
            summary["min_perplexity"] = min(perplexities)
            summary["max_perplexity"] = max(perplexities)

        # Log results to wandb if requested
        if log_to_wandb and wandb.run is not None:
            metrics = {
                "test/accuracy": accuracy,
                "test/correct_count": correct_count,
                "test/total_count": total_count,
            }
            if perplexities:
                metrics["test/avg_perplexity"] = summary["avg_perplexity"]
                metrics["test/min_perplexity"] = summary["min_perplexity"]
                metrics["test/max_perplexity"] = summary["max_perplexity"]

            wandb.log(metrics)

            # Create a table of results for visualization if task_id exists
            if "task_id" in dataset.features:
                columns = [
                    "task_id",
                    "question",
                    "correct_answer",
                    "predicted_answer",
                    "is_correct",
                ]
                table = wandb.Table(columns=columns)

                for res in results[: min(100, len(results))]:  # Limit to 100 examples
                    table.add_data(
                        res.get("task_id", "unknown"),
                        res["question"][:100] + "...",
                        res.get("correct_answer", ""),
                        res.get("predicted_answer", ""),
                        res.get("is_correct", False),
                    )

                wandb.log({"test_samples": table})

        # Restore original prompt type
        self.prompt_creator.set_prompt_type(original_prompt_type)
        # Restore the original response parser
        self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return summary

    def save_results(self, results, output_dir="./results"):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast

                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    # Keep as-is if conversion fails
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json

            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file


# %%

import os

from peft import LoraConfig

# Dataset pathsdate
val_path = "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/split_val_filtered"
train_path = (
    "/teamspace/studios/this_studio/workspace_1/data/raw/parquet_format/dataset_ready_for_training"
)

# Check if datasets exist, or create sample dataset for demonstration

# Load existing datasets
from datasets import load_from_disk

train_dataset = load_from_disk(train_path)
val_dataset = load_from_disk(val_path)
print(f"Loaded datasets: train={len(train_dataset)} examples, val={len(val_dataset)} examples")


# %%
# train_dataset = train_dataset.select(range(256))

# %%
# Initialize model handler (using 4-bit quantization for efficiency)
model_handler = QwenModelHandler(
    model_name="Unsloth/Qwen2.5-Coder-1.5B-Instruct",
    # quantization="4bit",
    device_map="auto",
    max_seq_length=2048,
)

# Initialize LoRA configuration for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# %%
# HuggingFace Hub credentials
hf_token = os.environ.get("HF_TOKEN", None)  # Set your token as environment variable
hub_model_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice"

# Initialize trainer with PromptCreator for training with teacher reasoning
prompt_creator = PromptCreator(PromptCreator.TEACHER_REASONED)

# %%

prompt_creator.is_teacher_mode = lambda *args: True
trainer = QwenTrainer(
    model=model_handler.model,
    tokenizer=model_handler.tokenizer,
    prompt_creator=prompt_creator,
    lora_config=lora_config,
    hub_token=hf_token,
    hub_model_id=hub_model_id,
)

# Train the model with HuggingFace Hub integration
print("Starting training with HuggingFace Hub integration")


# %%
train_dataset

# %%
trainer.train(
    dataset=train_dataset,
    output_dir="./model_output",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    push_to_hub=True if hf_token else False,
    verbose=True,
)

# %%
tester = MultipleChoiceTester(model_handler, prompt_creator=prompt_creator)

# %% [markdown]
# # Inference/Evaluation (Recommended to run on a GPU)

import os

import torch
from peft import PeftModel

# %%
# Load the latest model from HuggingFace Hub
from transformers import AutoModelForCausalLM, AutoTokenizer

# ````
# Set HuggingFace Hub credentials if available
hf_token = os.environ.get("HF_TOKEN")

# Model ID on HuggingFace Hub
hub_model_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice"

print(f"Loading model from HuggingFace Hub: {hub_model_id}")

# Load the model and tokenizer
try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hub_model_id, token=hf_token, trust_remote_code=True)

    # Load model with appropriate parameters for inference
    model = AutoModelForCausalLM.from_pretrained(
        hub_model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Create a new model handler with the loaded model and tokenizer
    # from model_handler import ModelHandler  # Assuming ModelHandler class is available

    lastest_model_handler_hub = QwenModelHandler(
        model_name=hub_model_id, max_seq_length=2048, quantization="4bit"
    )

    # Use FastLanguageModel
    from unsloth.models import FastLanguageModel

    FastLanguageModel.for_inference(lastest_model_handler_hub.model)
    prompt_creator = PromptCreator(PromptCreator.YAML_REASONING)
    # Create a tester with the loaded model
    latest_tester_hub = MultipleChoiceTester(
        lastest_model_handler_hub, prompt_creator=prompt_creator
    )

    print("Successfully loaded model from HuggingFace Hub!")

except Exception as e:
    print(f"Error loading model from HuggingFace Hub: {e}")
    print("Continuing with locally trained model...")

# %%
latest_tester_hub

# %% [markdown]
# - This model is loading with 4bit quantization. So the inference is very fast, but the quality is not as good as the full precision model.

# %%
import yaml

# Python example
python_example = {
    "question": "Which of the following correctly defines a list comprehension in Python?",
    "choices": [
        "[x**2 for x in range(10)]",
        "for(x in range(10)) { return x**2; }",
        "map(lambda x: x**2, range(10))",
        "[for x in range(10): x**2]",
    ],
    "answer": "A",  # Optional ground truth
}

result = latest_tester_hub.infer_example(python_example, temperature=0.0, stream=True)
print(
    f"Python Example - Predicted: {result['predicted_answer']}, Correct: {result.get('is_correct', 'Unknown')}"
)
print("Reasoning:")
try:
    # Print raw reasoning without trying to parse YAML
    print(result["reasoning"], "\n\n\n\n")
except Exception as e:
    print(f"Error: {e}")

# Java example
java_example = {
    "question": "Which of the following correctly creates a new instance of a class in Java?",
    "choices": [
        "MyClass obj = new MyClass();",
        "var obj = MyClass();",
        "MyClass obj = MyClass.new();",
        "obj = new(MyClass);",
    ],
    "answer": "A",  # Optional ground truth
}

result = latest_tester_hub.infer_example(java_example, temperature=0.0, stream=True)
print(
    f"\nJava Example - Predicted: {result['predicted_answer']}, Correct: {result.get('is_correct', 'Unknown')}"
)
print("Reasoning:")
try:
    print(result["reasoning"], "\n\n")
except Exception as e:
    print(f"Error: {e}")


# %%
# test_file is
# /teamspace/studios/this_studio/workspace_1/data/raw/b6_test_data.csv

import random
import re

import datasets
import pandas as pd
import torch
from tqdm.auto import tqdm

test_dataset = datasets.load_dataset(
    "csv", data_files="/teamspace/studios/this_studio/workspace_1/data/raw/b6_test_data.csv"
)
test_dataset = test_dataset["train"]  # weird due to huggingface dataset format!
# DatasetDict({
#     train: Dataset({
#         features: ['task_id', 'question', 'choices'],
#         num_rows: 1253
#     })
# })

# {'task_id': 'k10171',
#  'question': 'Question: What will be output of the following c code?\n#include<stdio.h>\nint main()\n{\n    int a= sizeof(signed) +sizeof(unsigned);\n    int b=sizeof(const)+sizeof(volatile);\n    printf("%d",a+++b);\n    return 0;\n}',
#  'choices': "['10', '9', '8', 'Error']"}

# example submission format:
# task_id,answer
# k09698,A
# k00203,A
# k00137,A
# k10490,A
# rt03960,A
# k08953,A
# k04984,A


def generate_answer_csv(test_dataset, model_handler, prompt_creator):
    tester = MultipleChoiceTester(model_handler, prompt_creator=prompt_creator)

    # Create a dictionary to store results
    results_dict = {}

    # Create a list to store detailed results for debugging
    debug_results = []

    # Select a few random examples for detailed debugging
    debug_indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))

    # Process examples in batches
    batch_size = 64  # Adjust based on GPU memory
    all_examples = list(test_dataset)
    num_examples = len(all_examples)

    for batch_start in tqdm(range(0, num_examples, batch_size), desc="Processing batches"):
        # Get the current batch
        batch_end = min(batch_start + batch_size, num_examples)
        batch = all_examples[batch_start:batch_end]

        # Prepare batch inputs
        batch_inputs = []
        batch_task_ids = []
        batch_prompts = []

        for example in batch:
            # Create prompt for each example
            prompt = prompt_creator.create_inference_prompt(
                example["question"], eval(example["choices"])
            )

            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            chat_text = model_handler.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            batch_inputs.append(chat_text)
            batch_task_ids.append(example["task_id"])
            batch_prompts.append(prompt)

        model_handler.tokenizer.padding_side = "left"

        # Tokenize all inputs at once
        tokenized_inputs = model_handler.tokenizer(
            batch_inputs, return_tensors="pt", padding=True, truncation=True
        ).to(model_handler.model.device)

        # Perform batch inference using the model directly
        tokenizer.padding_side = "left"
        with torch.no_grad():
            generated_ids = model_handler.model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                temperature=0.0,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                pad_token_id=model_handler.tokenizer.pad_token_id,
                # decoder-only models
                # padding_side='left'
            )

            # Extract only the generated part (not the input)
            batch_outputs = []
            for i, gen_ids in enumerate(generated_ids):
                # Get the length of the input
                input_length = tokenized_inputs.input_ids[i].shape[0]
                # Decode only the generated part
                output_text = model_handler.tokenizer.decode(
                    gen_ids[input_length:], skip_special_tokens=True
                )
                batch_outputs.append(output_text)

        # Process batch results
        for i, (task_id, output) in enumerate(zip(batch_task_ids, batch_outputs)):
            # Extract predicted answer from model output
            # Instead of using tester.parse_output which doesn't exist, implement parsing directly
            result = {}

            # Simple regex to find the answer (A, B, C, or D)
            answer_match = re.search(r"answer\s*(?:is|:)?\s*([ABCD])", output, re.IGNORECASE)
            if answer_match:
                result["predicted_answer"] = answer_match.group(1).upper()
            else:
                # Fallback: look for the first occurrence of A, B, C, or D
                for letter in ["A", "B", "C", "D"]:
                    if (
                        f"({letter})" in output
                        or f"{letter}." in output
                        or f"answer {letter}" in output.lower()
                    ):
                        result["predicted_answer"] = letter
                        break
                else:
                    # If no answer found, default to A
                    result["predicted_answer"] = "A"

            # Extract reasoning if available
            result["reasoning"] = output
            result["task_id"] = task_id

            # Store the result
            results_dict[task_id] = result["predicted_answer"]

            # Print progress update for every 10th example or last in batch
            if i % 10 == 0 or i == len(batch) - 1:
                print(f"Batch {batch_start//batch_size + 1}: Processed {i+1}/{len(batch)} examples")

            # For selected examples, save detailed results for debugging
            global_index = batch_start + i
            if global_index in debug_indices:
                example = batch[i]
                debug_results.append(
                    {
                        "task_id": task_id,
                        "question": example["question"],
                        "choices": example["choices"],
                        "predicted_answer": result["predicted_answer"],
                        "reasoning": result.get("reasoning", "No reasoning provided"),
                    }
                )

                # Print detailed debug information for these examples
                print(f"\n--- DETAILED DEBUG FOR TASK {task_id} ---")
                print(f"Question: {example['question'][:100]}...")
                print(f"Choices: {example['choices']}")
                print(f"Predicted: {result['predicted_answer']}")
                print(f"Reasoning snippet: {str(result.get('reasoning', 'No reasoning'))[:200]}...")
                print("-----------------------------------\n")

    # Create a DataFrame from the results dictionary
    submission_df = pd.DataFrame(list(results_dict.items()), columns=["task_id", "answer"])

    # Save the DataFrame to a CSV file
    submission_file = "submission.csv"
    submission_df.to_csv(submission_file, index=False)

    # Save debug results to a separate file
    debug_file = "debug_results.csv"
    pd.DataFrame(debug_results).to_csv(debug_file, index=False)

    print(f"Submission saved to {submission_file}")
    print(f"Debug results saved to {debug_file}")


prompt_creator = PromptCreator(PromptCreator.YAML_REASONING)
generate_answer_csv(test_dataset, lastest_model_handler_hub, prompt_creator)


# example submission format:
# task_id,answer
