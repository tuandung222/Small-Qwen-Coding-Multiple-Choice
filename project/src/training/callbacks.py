from transformers import TrainerCallback
from typing import Optional, Dict, Any, List
import torch
import wandb
from ..model.qwen_handler import QwenModelHandler
from ..data.prompt_creator import PromptCreator
from ..data.response_parser import ResponseParser

class ValidationCallback(TrainerCallback):
    def __init__(
        self,
        trainer_instance,
        val_dataset,
        quality_val_dataset=None,
        validation_strategy="epoch",
        validation_steps=None,
        validation_epochs=1,
        steps_per_epoch=None,
        save_best_checkpoint=True,
        early_stopper=None,
        log_examples=True,
        num_examples_to_log=3,
        combined_metric_weight=0.6,
    ):
        self.trainer_instance = trainer_instance
        self.val_dataset = val_dataset
        self.quality_val_dataset = quality_val_dataset
        self.validation_strategy = validation_strategy
        self.validation_steps = validation_steps
        self.validation_epochs = validation_epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_best_checkpoint = save_best_checkpoint
        self.early_stopper = early_stopper
        self.log_examples = log_examples
        self.num_examples_to_log = num_examples_to_log
        self.combined_metric_weight = combined_metric_weight
        self.best_metric = None
        self.tester = self._create_tester()

    def _create_tester(self):
        """Create a tester instance for validation"""
        from ..testing.tester import MultipleChoiceTester
        return MultipleChoiceTester(
            model_handler=self.trainer_instance._create_temp_model_handler(),
            prompt_creator=self.trainer_instance.prompt_creator,
        )

    def _select_examples_to_log(self):
        """Select examples to log during validation"""
        if not self.log_examples:
            return []
        
        import random
        num_examples = min(self.num_examples_to_log, len(self.val_dataset))
        return random.sample(range(len(self.val_dataset)), num_examples)

    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins"""
        self.best_metric = None

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        if self.validation_strategy == "steps" and self.validation_steps:
            if state.global_step % self.validation_steps == 0:
                self._validate_and_save(state.epoch, state.global_step)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        if self.validation_strategy == "epoch":
            self._validate_and_save(state.epoch, state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        if self.validation_strategy == "epoch":
            self._validate_and_save(state.epoch, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        if self.save_best_checkpoint and self.best_metric is not None:
            self.trainer_instance.save_checkpoint(
                args.output_dir,
                val_metric=self.best_metric,
                is_best=True
            )

    def _log_examples_to_wandb(self, step):
        """Log example predictions to wandb"""
        if not self.log_examples:
            return

        examples_to_log = self._select_examples_to_log()
        logged_examples = []

        for idx in examples_to_log:
            example = self.val_dataset[idx]
            prediction = self.tester.infer_example(example, temperature=0.0)
            logged_examples.append({
                "question": example["question"],
                "choices": example["choices"],
                "prediction": prediction,
                "ground_truth": example["answer"],
            })

        if wandb.run is not None:
            wandb.log({
                f"validation_examples_step_{step}": wandb.Table(
                    data=[[
                        ex["question"],
                        str(ex["choices"]),
                        ex["prediction"],
                        ex["ground_truth"]
                    ] for ex in logged_examples],
                    columns=["Question", "Choices", "Prediction", "Ground Truth"]
                )
            })

    def _validate_and_save(self, epoch, step):
        """Run validation and save checkpoints if needed"""
        # Run validation
        results = self.tester.evaluate_dataset(
            self.val_dataset,
            temperature=0.0,
            batch_size=64,
            log_to_wandb=True
        )

        # Calculate combined metric if quality validation is enabled
        if self.quality_val_dataset is not None:
            quality_results = self.tester.evaluate_dataset(
                self.quality_val_dataset,
                temperature=0.0,
                batch_size=64,
                log_to_wandb=True
            )
            combined_score = (
                self.combined_metric_weight * quality_results["reasoning_quality"] +
                (1 - self.combined_metric_weight) * results["accuracy"]
            )
            current_metric = combined_score
        else:
            current_metric = results["accuracy"]

        # Update best metric
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            if self.save_best_checkpoint:
                self.trainer_instance.save_checkpoint(
                    self.trainer_instance.trainer.args.output_dir,
                    val_metric=self.best_metric,
                    is_best=True
                )

        # Log examples
        self._log_examples_to_wandb(step)

        # Early stopping check
        if self.early_stopper is not None:
            if self.early_stopper({"combined_score": current_metric}):
                control.should_training_stop = True

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=0.001, metric_key="combined_score"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_key = metric_key
        self.best_metric = None
        self.patience_counter = 0

    def __call__(self, metrics, model=None):
        current_metric = metrics.get(self.metric_key)
        if current_metric is None:
            return False

        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.patience
