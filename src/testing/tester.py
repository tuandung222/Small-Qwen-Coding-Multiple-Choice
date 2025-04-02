import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

import wandb

from ..data.prompt_creator import PromptCreator
from ..data.response_parser import ResponseParser
from ..model.qwen_handler import QwenModelHandler


class MultipleChoiceTester:
    """Framework for testing Qwen models on multiple choice questions"""

    def __init__(self, model_handler=None, prompt_creator=None, checkpoint_path=None):
        """
        Initialize with model handler and prompt configuration

        Args:
            model_handler: The QwenModelHandler instance
            prompt_creator: Optional PromptCreator instance
            checkpoint_path: Optional path to a checkpoint to load
        """
        self.model_handler = model_handler
        self.prompt_creator = prompt_creator or PromptCreator()
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if os.path.basename(checkpoint_path) != "best_checkpoint" and os.path.exists(
            os.path.join(checkpoint_path, "best_checkpoint")
        ):
            checkpoint_path = os.path.join(checkpoint_path, "best_checkpoint")
            print(f"Using best checkpoint at: {checkpoint_path}")

        if self.model_handler is None:
            self.model_handler = QwenModelHandler(
                model_name=checkpoint_path,
                max_seq_length=2048,
            )
        else:
            self.model_handler.model = self.model_handler.model.from_pretrained(checkpoint_path)
            self.model_handler.tokenizer = self.model_handler.tokenizer.from_pretrained(
                checkpoint_path
            )

        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return self.model_handler

    def evaluate_dataset(
        self,
        dataset,
        temperature=0.0,
        num_examples=None,
        verbose=False,
        prompt_type=None,
        batch_size=64,
        log_to_wandb=False,
    ):
        """Evaluate model on a dataset of multiple choice questions"""
        if prompt_type:
            self.prompt_creator.set_prompt_type(prompt_type)

        if num_examples and num_examples < len(dataset):
            dataset = dataset.select(range(num_examples))

        if verbose:
            print(f"Evaluating on {len(dataset)} examples with temperature={temperature}")
            print(f"Prompt type: {self.prompt_creator.prompt_type}")
            print(f"Batch size: {batch_size}")

        correct_count = 0
        results = []
        start_time = time.time()

        with torch.inference_mode():
            for i in range(0, len(dataset), batch_size):
                batch_indices = range(i, min(i + batch_size, len(dataset)))
                batch_examples = []

                for idx in batch_indices:
                    example = dataset[idx]
                    choices = example.get("choices", [])
                    if isinstance(choices, str):
                        try:
                            import json

                            choices = json.loads(choices)
                        except:
                            choices = [c.strip() for c in choices.split("\n") if c.strip()]

                    batch_examples.append(
                        {
                            "question": example.get("question", ""),
                            "choices": choices,
                            "answer": example.get("answer", ""),
                        }
                    )

                batch_results, batch_metrics = self.infer_batch(
                    batch_examples, temperature=temperature, batch_size=batch_size
                )

                correct_count += batch_metrics.get("correct_count", 0)
                results.extend(batch_results)

                if verbose and (i + batch_size) % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    progress = min(100, 100 * (i + batch_size) / len(dataset))
                    accuracy = 100 * correct_count / len(results)
                    print(
                        f"Progress: {progress:.1f}% | Accuracy: {accuracy:.2f}% | Elapsed: {elapsed:.1f}s"
                    )

        accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0

        if verbose:
            elapsed = time.time() - start_time
            print(f"Evaluation completed in {elapsed:.1f}s")
            print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(dataset)})")

        if log_to_wandb and wandb.run:
            wandb.log({"evaluation/accuracy": accuracy})

        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(dataset),
            "results": results,
        }

    def infer_batch(self, examples, temperature=0.0, batch_size=64):
        """Run inference on a batch of examples"""
        results = []
        correct_count = 0

        with torch.inference_mode():
            for example in examples:
                result = self.infer_example(example, temperature=temperature, stream=False)
                results.append(result)

                if result.get("is_correct", False):
                    correct_count += 1

        accuracy = correct_count / len(examples) if examples else 0
        metrics = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(examples),
        }

        return results, metrics

    def infer_example(self, example, temperature=0.7, stream=True):
        """Run inference on a single example"""
        prompt = self.prompt_creator.create_inference_prompt(
            example["question"], example["choices"]
        )

        if stream:
            response_streamer = self.model_handler.generate_with_streaming(
                prompt=prompt, temperature=temperature, stream=True
            )
            response_text = ""
            for text in response_streamer:
                response_text += text
        else:
            response_text = self.model_handler.generate_with_streaming(
                prompt=prompt, temperature=temperature, stream=False
            )

        predicted_answer, reasoning = self.response_parser.parse(response_text)

        is_correct = None
        if "answer" in example and example["answer"] and predicted_answer:
            is_correct = example["answer"].upper() == predicted_answer

        return {
            "question": example["question"],
            "choices": example["choices"],
            "ground_truth": example.get("answer"),
            "predicted_answer": predicted_answer,
            "reasoning": reasoning,
            "is_correct": is_correct,
            "response_text": response_text,
        }

    def save_results(self, results, output_dir="./results"):
        """Save evaluation results to disk"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        save_data = {
            "accuracy": results["accuracy"],
            "correct_count": results["correct_count"],
            "total_count": results["total_count"],
            "timestamp": timestamp,
            "prompt_type": self.prompt_creator.prompt_type,
            "examples": results["results"],
        }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath
