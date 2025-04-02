import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import wandb
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.qwen_handler import QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.train import setup_model_and_trainer, setup_wandb_logging, train
from src.training.callbacks import EarlyStoppingCallback, ValidationCallback
from src.training.trainer import QwenTrainer
from src.utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)


def create_mock_dataset(num_examples=10):
    """Create a mock dataset for testing"""
    return Dataset.from_dict(
        {
            "question": [f"Test question {i}" for i in range(num_examples)],
            "choices": [["A", "B", "C", "D"] for _ in range(num_examples)],
            "answer": ["A" for _ in range(num_examples)],
            "reasoning": [f"Test reasoning {i}" for i in range(num_examples)],
        }
    )


@pytest.fixture(autouse=True)
def mock_imports():
    """Mock all necessary imports"""
    with patch("src.train.AutoModelForCausalLM") as mock_model_class, patch(
        "src.train.AutoTokenizer"
    ) as mock_tokenizer_class, patch("src.train.LoraConfig") as mock_lora_config, patch(
        "src.train.get_peft_model"
    ) as mock_get_peft_model, patch(
        "src.train.Trainer"
    ) as mock_trainer_class, patch(
        "src.train.TrainingArguments"
    ) as mock_training_args_class, patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_trainer = MagicMock()
        mock_training_args = MagicMock()
        mock_dataset = create_mock_dataset()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_trainer_class.return_value = mock_trainer
        mock_training_args_class.return_value = mock_training_args
        mock_get_peft_model.return_value = mock_model
        mock_load_from_disk.return_value = mock_dataset
        yield


@pytest.fixture(autouse=True)
def mock_cuda():
    """Mock CUDA availability"""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing"""
    with patch("src.utils.wandb_logger.wandb") as mock:
        mock.init.return_value = MagicMock()
        mock.util.generate_id.return_value = "test_id"
        mock.log = MagicMock()
        yield mock


@pytest.fixture
def mock_model_handler():
    """Mock model handler for testing"""
    handler = Mock(spec=QwenModelHandler)
    handler.model = Mock()
    handler.model.config = Mock()
    handler.model.config.max_position_embeddings = 4096
    handler.model.parameters = lambda: [Mock(numel=lambda: 1000) for _ in range(10)]
    handler.tokenizer = Mock()
    handler.setup = Mock()
    return handler


@pytest.fixture
def mock_trainer():
    """Mock trainer for testing"""
    trainer = Mock(spec=QwenTrainer)
    trainer.train.return_value = {"loss": 0.5, "learning_rate": 1e-4}
    trainer.best_val_metric = 0.8
    trainer.best_checkpoint_path = "test_checkpoint"
    trainer.prompt_creator = Mock()
    trainer.prompt_creator.prompt_type = PromptCreator.TEACHER_REASONED
    trainer.model = Mock()
    trainer.model.config = Mock()
    trainer.model.config.max_position_embeddings = 4096
    trainer.model.parameters = lambda: [Mock(numel=lambda: 1000) for _ in range(10)]
    trainer.tokenizer = Mock()
    trainer.training_stats = {"epoch": 1, "loss": 0.5}
    trainer.validation_stats = {"accuracy": 0.8}
    trainer.train_dataset = Mock()
    trainer.val_dataset = Mock()
    trainer.train_dataset.__iter__ = lambda: iter([])
    trainer.val_dataset.__iter__ = lambda: iter([])
    trainer.push_to_hub = Mock()
    trainer.setup = Mock()
    trainer.trainer = Mock()
    trainer.trainer.train.return_value = MagicMock(metrics={"loss": 0.5, "learning_rate": 1e-4})
    trainer.trainer.evaluate.return_value = {"eval_loss": 0.4}
    trainer.trainer.state = MagicMock(best_model_checkpoint="test_checkpoint")
    return trainer


@pytest.fixture
def mock_wandb_logger():
    """Mock WandBLogger for testing"""
    logger = Mock(spec=WandBLogger)
    logger.init_run = Mock()
    logger.log_model_info = Mock()
    logger.finish_run = Mock()
    logger.setup = Mock()
    logger.run = MagicMock()
    logger.run.summary = MagicMock()
    return logger


@pytest.fixture(autouse=True)
def setup_test_data():
    """Setup test data directories and files"""
    # Create test data directories
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "train_dataset"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "val_dataset"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "train_dataset_response"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "train_dataset_val"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "val_dataset_val"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "checkpoint_output"), exist_ok=True)

    # Create test datasets
    train_dataset = create_mock_dataset()
    val_dataset = create_mock_dataset(5)

    # Save datasets
    train_dataset.save_to_disk(os.path.join(TEST_DATA_DIR, "train_dataset"))
    val_dataset.save_to_disk(os.path.join(TEST_DATA_DIR, "val_dataset"))
    train_dataset.save_to_disk(os.path.join(TEST_DATA_DIR, "train_dataset_response"))
    train_dataset.save_to_disk(os.path.join(TEST_DATA_DIR, "train_dataset_val"))
    val_dataset.save_to_disk(os.path.join(TEST_DATA_DIR, "val_dataset_val"))

    yield

    # Cleanup test data
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


def test_setup_wandb_logging(mock_wandb, mock_wandb_logger):
    """Test wandb logging setup"""
    with patch("src.train.WandBLogger", return_value=mock_wandb_logger), patch(
        "src.train.WandBConfig"
    ) as mock_wandb_config:
        # Setup mock config
        mock_config = MagicMock()
        mock_wandb_config.return_value = mock_config

        # Run setup
        logger = setup_wandb_logging(
            project_name="test_project",
            run_name="test_run",
            log_memory=True,
            log_gradients=True,
        )

        # Verify logger setup
        assert logger == mock_wandb_logger
        mock_wandb_logger.setup.assert_called_once()

        # Verify config setup
        mock_wandb_config.assert_called_once_with(
            project_name="test_project",
            run_name="test_run",
            log_memory=True,
            log_gradients=True,
        )

        # Verify wandb initialization
        mock_wandb.init.assert_called_once_with(
            project="test_project",
            name="test_run",
            config=mock_config,
        )


def test_setup_model_and_trainer(mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger):
    """Test model and trainer setup"""
    with patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.QwenTrainer", return_value=mock_trainer
    ), patch("src.train.ModelHandler", return_value=mock_model_handler), patch(
        "src.train.LoraConfig"
    ) as mock_lora_config, patch(
        "src.train.get_peft_model"
    ) as mock_get_peft_model, patch(
        "src.train.AutoModelForCausalLM.from_pretrained"
    ) as mock_from_pretrained, patch(
        "src.train.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer_from_pretrained:
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Setup mock PEFT model
        mock_peft_model = MagicMock()
        mock_get_peft_model.return_value = mock_peft_model

        # Run setup
        model_handler, trainer = setup_model_and_trainer(
            model_name="test_model",
            output_dir=TEST_DATA_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            verbose=False,
        )

        # Verify model handler setup
        assert model_handler == mock_model_handler
        mock_model_handler.setup.assert_called_once()

        # Verify trainer setup
        assert trainer == mock_trainer
        mock_trainer.setup.assert_called_once()

        # Verify model loading
        mock_from_pretrained.assert_called_once_with(
            "test_model",
            trust_remote_code=True,
            device_map="auto",
        )

        # Verify tokenizer loading
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "test_model",
            trust_remote_code=True,
        )

        # Verify LoRA config
        mock_lora_config.assert_called_once_with(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Verify PEFT model setup
        mock_get_peft_model.assert_called_once_with(
            mock_model,
            mock_lora_config.return_value,
        )


def test_train_basic_functionality(mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger):
    """Test basic training functionality"""
    with patch(
        "src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)
    ), patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk:
        # Create test dataset
        train_dataset = create_mock_dataset()

        # Setup mock dataset loading
        mock_load_from_disk.return_value = train_dataset

        # Run basic training
        results = train(
            model_name="test_model",
            train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
            output_dir=TEST_DATA_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            verbose=False,
        )

        # Verify results
        assert isinstance(results, dict)
        assert "training_stats" in results
        assert "best_val_metric" in results
        assert "best_checkpoint_path" in results
        assert results["best_val_metric"] == 0.8
        assert results["best_checkpoint_path"] == "test_checkpoint"

        # Verify training was called with correct parameters
        mock_trainer.train.assert_called_once()
        train_args = mock_trainer.train.call_args[1]
        assert train_args["num_train_epochs"] == 1
        assert train_args["per_device_train_batch_size"] == 2
        assert train_args["output_dir"] == os.path.join(TEST_DATA_DIR, "checkpoints")
        assert train_args["save_strategy"] == "epoch"
        assert train_args["save_total_limit"] == 1
        assert train_args["load_best_model_at_end"] is True
        assert train_args["metric_for_best_model"] == "eval_loss"
        assert train_args["greater_is_better"] is False


def test_train_response_only_mode(mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger):
    """Test response-only training mode"""
    with patch(
        "src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)
    ), patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk, patch(
        "unsloth.chat_templates.train_on_responses_only"
    ) as mock_train_on_responses:
        # Create test dataset
        train_dataset = create_mock_dataset()
        train_path = os.path.join(TEST_DATA_DIR, "train_dataset_response")
        train_dataset.save_to_disk(train_path)

        # Setup mock dataset loading
        mock_load_from_disk.return_value = train_dataset

        # Mock Unsloth's train_on_responses_only function
        mock_train_on_responses.return_value = mock_trainer

        # Run training with response-only mode
        results = train(
            model_name="test_model",
            train_path=train_path,
            output_dir=TEST_DATA_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            verbose=False,
            train_on_response_only=True,
        )

        # Verify response-only mode was applied
        mock_train_on_responses.assert_called_once()
        call_args = mock_train_on_responses.call_args[1]
        assert call_args["instruction_part"] == "<|im_start|>user\n"
        assert call_args["response_part"] == "<|im_start|>assistant\n"

        # Verify results
        assert isinstance(results, dict)
        assert "training_stats" in results
        assert "best_val_metric" in results
        assert "best_checkpoint_path" in results
        assert results["best_val_metric"] == 0.8
        assert results["best_checkpoint_path"] == "test_checkpoint"


def test_train_validation_strategies(
    mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger
):
    """Test different validation strategies"""
    with patch(
        "src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)
    ), patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk:
        # Create test datasets
        train_dataset = create_mock_dataset()
        val_dataset = create_mock_dataset(5)

        # Setup mock dataset loading
        mock_load_from_disk.side_effect = (
            lambda path: train_dataset if "train_dataset" in path else val_dataset
        )

        # Test different validation strategies
        strategies = ["epoch", "steps", "no"]
        for strategy in strategies:
            results = train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset_val"),
                val_path=os.path.join(TEST_DATA_DIR, "val_dataset_val"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                validation_strategy=strategy,
                verbose=False,
            )

            assert isinstance(results, dict)
            assert "training_stats" in results
            assert "best_val_metric" in results
            assert "best_checkpoint_path" in results
            assert results["best_val_metric"] == 0.8
            assert results["best_checkpoint_path"] == "test_checkpoint"


def test_train_error_handling(mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger):
    """Test error handling in training process"""
    with patch(
        "src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)
    ), patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk:
        # Create test dataset
        train_dataset = create_mock_dataset()

        # Setup mock dataset loading
        mock_load_from_disk.return_value = train_dataset

        # Test invalid model name
        with pytest.raises(ValueError, match="Invalid model name"):
            train(
                model_name="invalid_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                verbose=False,
            )

        # Test invalid train path
        with pytest.raises(ValueError, match="Train path does not exist"):
            train(
                model_name="test_model",
                train_path="nonexistent_path",
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                verbose=False,
            )

        # Test invalid validation path
        with pytest.raises(ValueError, match="Validation path does not exist"):
            train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                val_path="nonexistent_path",
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                verbose=False,
            )

        # Test invalid validation strategy
        with pytest.raises(ValueError, match="Invalid validation strategy"):
            train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                validation_strategy="invalid_strategy",
                verbose=False,
            )

        # Test invalid checkpoint strategy
        with pytest.raises(ValueError, match="Invalid checkpoint strategy"):
            train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                checkpoint_strategy="invalid_strategy",
                verbose=False,
            )


def test_train_checkpoint_saving(mock_wandb, mock_trainer, mock_model_handler, mock_wandb_logger):
    """Test checkpoint saving during training"""
    with patch(
        "src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)
    ), patch("src.train.setup_wandb_logging", return_value=mock_wandb_logger), patch(
        "src.train.load_from_disk"
    ) as mock_load_from_disk, patch(
        "src.train.save_checkpoint"
    ) as mock_save_checkpoint:
        # Create test dataset
        train_dataset = create_mock_dataset()

        # Setup mock dataset loading
        mock_load_from_disk.return_value = train_dataset

        # Test different checkpoint strategies
        strategies = ["best", "last", "all"]
        for strategy in strategies:
            results = train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                checkpoint_strategy=strategy,
                verbose=False,
            )

            assert isinstance(results, dict)
            assert "training_stats" in results
            assert "best_val_metric" in results
            assert "best_checkpoint_path" in results
            assert results["best_val_metric"] == 0.8
            assert results["best_checkpoint_path"] == "test_checkpoint"

            # Verify checkpoint saving based on strategy
            if strategy == "best":
                mock_save_checkpoint.assert_called_with(
                    mock_trainer, os.path.join(TEST_DATA_DIR, "checkpoints", "best"), "best"
                )
            elif strategy == "last":
                mock_save_checkpoint.assert_called_with(
                    mock_trainer, os.path.join(TEST_DATA_DIR, "checkpoints", "last"), "last"
                )
            elif strategy == "all":
                assert mock_save_checkpoint.call_count == 2  # Called for both best and last


# Test commands for each function:

# To run all tests in this file:
# pytest test_train.py -v

# To run a specific test function:
# pytest test_train.py::test_setup_wandb_logging -v
# pytest test_train.py::test_setup_model_and_trainer -v
# pytest test_train.py::test_train_basic_functionality -v
# pytest test_train.py::test_train_checkpoint_saving -v

# To run tests with specific markers (if any were defined):
# pytest test_train.py -m <marker_name> -v

# To generate a test coverage report:
# pytest test_train.py --cov=src

# To output test results in JUnit XML format:
# pytest test_train.py --junitxml=test-results.xml

# To run tests in parallel (using pytest-xdist):
# pytest test_train.py -n 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
