import concurrent.futures
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from ..prompt_creator import PromptCreator
from .config import ModelConfig, PromptConfig, TestConfig


class TeacherSynthesisFramework:
    """Framework for generating synthetic explanations using teacher models with access to labels"""

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
        Initialize the synthesis framework

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

        # Initialize OpenAI client
        api_key = self.model_config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key provided. Set it in ModelConfig or as environment variable."
            )
        self.client = OpenAI(api_key=api_key)

        # Initialize prompt config
        self.prompt_config = PromptConfig()

    def generate_synthetic_explanation(
        self, question: str, choices: List[str], correct_answer: str
    ) -> Tuple[str, float]:
        """
        Generate a synthetic explanation using the teacher model

        Args:
            question: The question text
            choices: List of possible answers
            correct_answer: The correct answer (letter or index)

        Returns:
            Tuple of (yaml_explanation, response_time)
        """
        # Create prompt
        prompt = self._create_teacher_prompt(question, choices, correct_answer)

        try:
            start_time = time.time()

            # Build messages
            messages = []
            if self.prompt_config.add_system_prompt:
                messages.append({"role": "system", "content": self.model_config.system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call the API
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
            yaml_response = response.choices[0].message.content.strip()

            return yaml_response, response_time

        except Exception as e:
            if self.verbose:
                print(f"Error generating explanation: {e}")
            return "", 0

    def _create_teacher_prompt(self, question: str, choices: List[str], correct_answer: str) -> str:
        """Create a prompt for the teacher model"""
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

    def _process_example(self, example: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single example and generate synthetic explanation"""
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
                        yaml_text = "---\n" + yaml_response
                        parsed = yaml.safe_load(yaml_text)
                        if isinstance(parsed, dict) and "answer" in parsed:
                            answer = parsed["answer"]
                    except Exception:
                        pass

                    is_correct = answer == correct_answer

                    return {
                        "task_id": task_id,
                        "yaml_response": yaml_response,
                        "correct_answer": correct_answer,
                        "predicted_answer": answer,
                        "is_correct": is_correct,
                        "response_time": 0,
                        "output_file": output_path,
                        "reused_existing": True,
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Error reading existing file {output_path}: {e}")

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

        # Extract fields
        question = example["question"]
        task_id = example["task_id"]
        correct_answer = example["answer"]

        # Parse choices
        choices_str = example["choices"]
        try:
            if isinstance(choices_str, str):
                if choices_str.startswith("[") and choices_str.endswith("]"):
                    import ast

                    choices = ast.literal_eval(choices_str)
                else:
                    choices = [c.strip() for c in choices_str.replace("\n", ",").split(",")]
            else:
                choices = list(choices_str)

            choices = [str(c).strip() for c in choices if c]
        except Exception as e:
            if self.verbose:
                print(f"Example {index} error parsing choices: {e}")
            return {
                "task_id": task_id,
                "error": f"Error parsing choices: {str(e)}",
                "is_correct": False,
                "response_time": 0,
                "yaml_response": "",
            }

        # Generate synthetic explanation
        yaml_response, response_time = self.generate_synthetic_explanation(
            question, choices, correct_answer
        )

        # Parse the response to verify correctness
        answer = None
        if yaml_response:
            try:
                yaml_text = "---\n" + yaml_response
                parsed = yaml.safe_load(yaml_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
            except Exception:
                pass

        # Check if the answer was preserved correctly
        is_correct = answer == correct_answer

        # Create result
        result = {
            "task_id": task_id,
            "yaml_response": yaml_response,
            "correct_answer": correct_answer,
            "predicted_answer": answer,
            "is_correct": is_correct,
            "response_time": response_time,
        }

        # Save the synthetic explanation
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
        """Process the dataset and generate synthetic explanations"""
        # Determine sample size
        total_examples = len(dataset)
        sample_size = self.sample_size or total_examples
        sample_size = min(sample_size, total_examples)

        if self.verbose:
            print(
                f"Generating synthetic explanations using {self.model_config.name} for {sample_size} examples..."
            )
            print(f"Saving results to {self.output_dir}")

        # Initialize start time
        start_time = time.time()
        self.metrics["start_time"] = start_time
        results = []

        # Check for existing progress
        progress_file = os.path.join(self.output_dir, "synthesis_progress.json")
        processed_indices = set()

        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    progress_data = json.load(f)
                    processed_indices = set(progress_data.get("processed_indices", []))
                    previous_results = progress_data.get("partial_results", [])
                    if previous_results:
                        results = previous_results

                    if self.verbose:
                        print(
                            f"Resuming from previous run. {len(processed_indices)} examples already processed."
                        )
            except Exception as e:
                if self.verbose:
                    print(f"Could not load progress file: {e}. Starting from scratch.")

        # Process examples concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrent_requests
        ) as executor:
            # Submit tasks for unprocessed examples
            future_to_idx = {
                executor.submit(self._process_example, dataset[i], i): i
                for i in range(sample_size)
                if i not in processed_indices
            }

            # Process results as they complete
            completed = 0
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(future_to_idx),
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

                # Save progress periodically
                completed += 1
                if completed % 50 == 0:
                    self._save_progress(results, processed_indices, progress_file)

        # Calculate and save final metrics
        self._calculate_metrics(results)

        # Save final metrics
        metrics_path = os.path.join(self.output_dir, "synthesis_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Clean up progress file
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

    def _save_progress(
        self, results: List[Dict[str, Any]], processed_indices: set, progress_file: str
    ):
        """Save intermediate progress"""
        # Calculate current metrics
        self._calculate_metrics(results)

        # Save progress information
        progress_data = {
            "processed_indices": list(processed_indices),
            "metrics": self.metrics,
            "partial_results": results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(progress_file, "w") as f:
            json.dump(progress_data, f)

        # Also save current metrics
        metrics_path = os.path.join(self.output_dir, "synthesis_metrics_latest.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def _calculate_metrics(self, results: List[Dict[str, Any]]):
        """Calculate metrics from results"""
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

        if "start_time" in self.metrics:
            self.metrics["total_time"] = time.time() - self.metrics["start_time"]
        else:
            self.metrics["total_time"] = 0

        self.metrics["last_updated"] = datetime.now().isoformat()
