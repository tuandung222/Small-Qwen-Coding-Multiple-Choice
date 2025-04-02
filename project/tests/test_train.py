import os
import sys
from pathlib import Path
import pytest
import torch
from unittest.mock import Mock, patch
from datasets import Dataset
import numpy as np

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.qwen_handler import QwenModelHandler
from src.data.prompt_creator import PromptCreator
from src.training.trainer import QwenTrainer
from src.training.callbacks import ValidationCallback, EarlyStoppingCallback
from src.utils.wandb_logger import WandBLogger, WandBConfig, WandBCallback
from src.train import train, setup_wandb_logging, setup_model_and_trainer

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def create_mock_dataset(num_examples=10):
    """Create a mock dataset for testing"""
    return Dataset.from_dict({
        "question": [f"Test question {i}" for i in range(num_examples)],
        "choices": [["A", "B", "C", "D"] for _ in range(num_examples)],
        "answer": ["A" for _ in range(num_examples)],
        "reasoning": [f"Test reasoning {i}" for i in range(num_examples)],
    })

@pytest.fixture
def mock_wandb():
    """Mock wandb for testing"""
    with patch("src.utils.wandb_logger.wandb") as mock:
        yield mock

@pytest.fixture
def mock_trainer():
    """Mock trainer for testing"""
    trainer = Mock(spec=QwenTrainer)
    trainer.train.return_value = {"loss": 0.5, "learning_rate": 1e-4}
    trainer.best_val_metric = 0.8
    trainer.best_checkpoint_path = "test_checkpoint"
    return trainer

@pytest.fixture
def mock_model_handler():
    """Mock model handler for testing"""
    handler = Mock(spec=QwenModelHandler)
    handler.model = Mock()
    handler.tokenizer = Mock()
    return handler

def test_setup_wandb_logging(mock_wandb):
    """Test wandb logging setup"""
    config = {"test": "config"}
    logger = setup_wandb_logging(
        model_name="test_model",
        run_name="test_run",
        config=config,
        project_name="test_project",
        entity="test_entity",
        tags=["test"],
        notes="test notes"
    )
    
    assert isinstance(logger, WandBLogger)
    mock_wandb.init.assert_called_once()

def test_setup_model_and_trainer(mock_model_handler):
    """Test model and trainer setup"""
    with patch("src.train.QwenModelHandler", return_value=mock_model_handler):
        model_handler, trainer = setup_model_and_trainer(
            model_name="test_model",
            max_seq_length=2048,
            quantization="4bit",
            device_map="auto",
            cache_dir=TEST_DATA_DIR,
            prompt_type=PromptCreator.TEACHER_REASONED,
            hub_token="test_token",
            hub_model_id="test_model_id"
        )
        
        assert isinstance(model_handler, Mock)
        assert isinstance(trainer, Mock)
        assert trainer.prompt_creator.prompt_type == PromptCreator.TEACHER_REASONED

def test_train_basic_functionality(mock_wandb, mock_trainer, mock_model_handler):
    """Test basic training functionality"""
    with patch("src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)):
        # Create test datasets
        train_dataset = create_mock_dataset()
        val_dataset = create_mock_dataset(5)
        
        # Save datasets to disk
        train_path = os.path.join(TEST_DATA_DIR, "train_dataset")
        val_path = os.path.join(TEST_DATA_DIR, "val_dataset")
        train_dataset.save_to_disk(train_path)
        val_dataset.save_to_disk(val_path)
        
        # Run training
        results = train(
            model_name="test_model",
            train_path=train_path,
            val_path=val_path,
            output_dir=TEST_DATA_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            verbose=False,
            train_on_response_only=True,
        )
        
        assert isinstance(results, dict)
        assert "training_stats" in results
        assert "best_val_metric" in results
        assert "best_checkpoint_path" in results
        assert results["best_val_metric"] == 0.8
        assert results["best_checkpoint_path"] == "test_checkpoint"

def test_train_response_only_mode(mock_wandb, mock_trainer, mock_model_handler):
    """Test response-only training mode"""
    with patch("src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)):
        with patch("src.train.train_on_responses_only") as mock_train_on_responses:
            # Create test dataset
            train_dataset = create_mock_dataset()
            train_path = os.path.join(TEST_DATA_DIR, "train_dataset_response")
            train_dataset.save_to_disk(train_path)
            
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

def test_train_validation_strategies(mock_wandb, mock_trainer, mock_model_handler):
    """Test different validation strategies"""
    with patch("src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)):
        # Create test datasets
        train_dataset = create_mock_dataset()
        val_dataset = create_mock_dataset(5)
        
        # Save datasets to disk
        train_path = os.path.join(TEST_DATA_DIR, "train_dataset_val")
        val_path = os.path.join(TEST_DATA_DIR, "val_dataset_val")
        train_dataset.save_to_disk(train_path)
        val_dataset.save_to_disk(val_path)
        
        # Test different validation strategies
        strategies = ["epoch", "steps", "no"]
        for strategy in strategies:
            results = train(
                model_name="test_model",
                train_path=train_path,
                val_path=val_path,
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                validation_strategy=strategy,
                verbose=False,
            )
            
            assert isinstance(results, dict)
            assert "training_stats" in results

def test_train_error_handling(mock_wandb, mock_trainer, mock_model_handler):
    """Test error handling in training"""
    with patch("src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)):
        # Test with invalid dataset path
        with pytest.raises(Exception):
            train(
                model_name="test_model",
                train_path="invalid_path",
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                verbose=False,
            )
        
        # Test with invalid validation strategy
        with pytest.raises(ValueError):
            train(
                model_name="test_model",
                train_path=os.path.join(TEST_DATA_DIR, "train_dataset"),
                output_dir=TEST_DATA_DIR,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                validation_strategy="invalid",
                verbose=False,
            )

def test_train_checkpoint_saving(mock_wandb, mock_trainer, mock_model_handler):
    """Test checkpoint saving functionality"""
    with patch("src.train.setup_model_and_trainer", return_value=(mock_model_handler, mock_trainer)):
        # Create test dataset
        train_dataset = create_mock_dataset()
        train_path = os.path.join(TEST_DATA_DIR, "train_dataset_checkpoint")
        train_dataset.save_to_disk(train_path)
        
        # Run training with checkpoint saving
        results = train(
            model_name="test_model",
            train_path=train_path,
            output_dir=TEST_DATA_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=1,
            save_best_checkpoint=True,
            verbose=False,
        )
        
        assert "best_checkpoint_path" in results
        assert results["best_checkpoint_path"] == "test_checkpoint"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 