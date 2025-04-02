import os
import sys
import shutil
from pathlib import Path
import pytest
import torch
import wandb
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, load_from_disk
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from unsloth import FastLanguageModel
import yaml
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.qwen_handler import QwenModelHandler, ModelSource
from src.data.prompt_creator import PromptCreator
from src.training.trainer import QwenTrainer
from src.training.callbacks import ValidationCallback, EarlyStoppingCallback
from src.utils.wandb_logger import WandBLogger, WandBConfig, WandBCallback
from src.train import train, setup_wandb_logging, setup_model_and_trainer

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables needed for authentication"""
    with patch.dict(os.environ, {
        'HF_TOKEN': 'mock_hf_token',
        'WANDB_API_KEY': 'mock_wandb_key',
        'OPENAI_API_KEY': 'mock_openai_key'
    }):
        yield

@pytest.fixture(autouse=True)
def mock_hf_api():
    """Mock HuggingFace Hub API"""
    with patch("huggingface_hub.HfApi") as mock_api:
        # Mock successful API responses
        mock_api.return_value.model_info.return_value = MagicMock(
            id="unsloth/Qwen2.5-Coder-1.5B-Instruct",
            sha="mock_sha",
            config={
                "model_type": "qwen",
                "architectures": ["Qwen2ForCausalLM"],
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768
            }
        )
        
        # Mock repository contents
        mock_api.return_value.list_repo_files.return_value = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        
        # Mock repository validation
        def mock_repo_exists(*args, **kwargs):
            repo_id = kwargs.get("repo_id", "")
            return "unsloth" in repo_id or "Qwen" in repo_id
            
        mock_api.return_value.repo_exists = mock_repo_exists
        
        # Mock commit history
        mock_api.return_value.list_repo_commits.return_value = [{
            "id": "mock_commit_id",
            "title": "Initial commit",
            "date": "2024-03-29T00:00:00Z"
        }]
        
        yield mock_api

@pytest.fixture(autouse=True)
def mock_hf_hub():
    """Mock HuggingFace Hub operations"""
    with patch("huggingface_hub.hf_hub_download") as mock_download, \
         patch("huggingface_hub.snapshot_download") as mock_snapshot, \
         patch("huggingface_hub.login") as mock_login, \
         patch("huggingface_hub.file_download") as mock_file_download:
        
        # Configure mock downloads to return valid file paths
        def mock_download_func(repo_id, filename, **kwargs):
            if "config" in filename:
                return os.path.join(TEST_DATA_DIR, "config.json")
            elif "model" in filename:
                return os.path.join(TEST_DATA_DIR, "model.safetensors")
            elif "tokenizer" in filename:
                return os.path.join(TEST_DATA_DIR, "tokenizer.json")
            return os.path.join(TEST_DATA_DIR, filename)
            
        mock_download.side_effect = mock_download_func
        mock_snapshot.return_value = TEST_DATA_DIR
        mock_login.return_value = True
        mock_file_download.side_effect = mock_download_func
        
        # Create mock files
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        with open(os.path.join(TEST_DATA_DIR, "config.json"), "w") as f:
            f.write('{"model_type": "qwen", "architectures": ["Qwen2ForCausalLM"]}')
        
        yield {
            "download": mock_download,
            "snapshot": mock_snapshot,
            "login": mock_login,
            "file_download": mock_file_download
        }

@pytest.fixture
def mock_unsloth():
    """Mock Unsloth's FastLanguageModel with proper error handling"""
    with patch("unsloth.FastLanguageModel") as mock:
        # Create mock model and tokenizer with proper configurations
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Configure mock tokenizer
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.apply_chat_template.return_value = "mock chat template"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "mock response"
        
        # Configure mock model
        mock_model.config = MagicMock(
            model_type="qwen",
            max_position_embeddings=2048,
            hidden_size=1024,
            num_attention_heads=16
        )
        mock_model.device = "cuda" if torch.cuda.is_available() else "cpu"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.dtype = torch.float16
        
        # Configure FastLanguageModel class methods
        def mock_from_pretrained(*args, **kwargs):
            # Simulate potential HF Hub errors
            if "error" in kwargs.get("model_name", ""):
                raise HfHubHTTPError("Mock HF Hub Error", response=MagicMock(status_code=500))
            return mock_model, mock_tokenizer
            
        mock.from_pretrained.side_effect = mock_from_pretrained
        mock.for_inference = MagicMock()
        mock.get_peft_model = MagicMock(return_value=mock_model)
        
        yield mock

def test_model_initialization_with_error_handling(mock_unsloth, mock_env_vars):
    """Test model initialization with error handling"""
    # Test successful Unsloth initialization
    model_handler = QwenModelHandler(
        model_name="unsloth/Qwen2.5-7B",
        model_source=ModelSource.UNSLOTH,
        quantization="4bit",
        max_seq_length=2048
    )
    assert model_handler.model is not None
    assert model_handler.tokenizer is not None
    
    # Test handling of HF Hub errors
    with pytest.raises(HfHubHTTPError):
        QwenModelHandler(
            model_name="error/model",
            model_source=ModelSource.UNSLOTH
        )
    
    # Test fallback to HuggingFace
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model, \
         patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        model_handler = QwenModelHandler(
            model_name="Qwen/Qwen1.5-7B",
            model_source=ModelSource.HUGGINGFACE,
            quantization="4bit",
            max_seq_length=2048
        )
        assert model_handler.model is not None
        assert model_handler.tokenizer is not None

def test_model_loading_retry(mock_unsloth, mock_env_vars):
    """Test model loading with retry mechanism"""
    # Configure mock to fail first then succeed
    fail_count = [0]
    def mock_from_pretrained(*args, **kwargs):
        if fail_count[0] < 1:
            fail_count[0] += 1
            raise HfHubHTTPError("Mock HF Hub Error", response=MagicMock(status_code=500))
        return mock_unsloth.from_pretrained.return_value
    
    mock_unsloth.from_pretrained.side_effect = mock_from_pretrained
    
    # Test model loading with retry
    model_handler = QwenModelHandler(
        model_name="unsloth/Qwen2.5-7B",
        model_source=ModelSource.UNSLOTH,
        quantization="4bit",
        max_seq_length=2048
    )
    
    assert model_handler.model is not None
    assert model_handler.tokenizer is not None
    assert fail_count[0] == 1  # Verify retry occurred

def test_model_generation_error_handling(mock_unsloth, mock_env_vars):
    """Test model generation with error handling"""
    model_handler = QwenModelHandler(
        model_name="unsloth/Qwen2.5-7B",
        model_source=ModelSource.UNSLOTH,
        quantization="4bit",
        max_seq_length=2048
    )
    
    # Test normal generation
    response = model_handler.generate_with_streaming(
        prompt="Test prompt",
        temperature=0.0,
        stream=False
    )
    assert response is not None
    
    # Test generation with CUDA OOM error
    with patch.object(model_handler.model, "generate") as mock_generate:
        mock_generate.side_effect = torch.cuda.OutOfMemoryError()
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            model_handler.generate_with_streaming(
                prompt="Test prompt",
                temperature=0.0,
                stream=False
            )

def create_mock_dataset(num_examples=10):
    """Create a mock dataset for testing"""
    return Dataset.from_dict({
        "task_id": [f"test_{i}" for i in range(num_examples)],
        "question": [f"Test question {i}" for i in range(num_examples)],
        "choices": [["A", "B", "C", "D"] for _ in range(num_examples)],
        "answer": ["A" for _ in range(num_examples)],
        "yml_str": [
            """
            understanding: |
              This is a test question
            analysis: |
              A is correct because...
            reasoning: |
              The reasoning is...
            conclusion: |
              Therefore, A is the answer
            answer: A
            """ for _ in range(num_examples)
        ]
    })

@pytest.fixture
def mock_wandb():
    """Mock wandb for testing"""
    with patch("wandb.init") as mock_init, \
         patch("wandb.log") as mock_log, \
         patch("wandb.finish") as mock_finish:
        mock_init.return_value = MagicMock()
        yield {
            "init": mock_init,
            "log": mock_log,
            "finish": mock_finish
        }

@pytest.fixture
def test_dataset():
    """Create and save test datasets"""
    train_data = create_mock_dataset(20)
    val_data = create_mock_dataset(10)
    
    train_path = os.path.join(TEST_DATA_DIR, "train")
    val_path = os.path.join(TEST_DATA_DIR, "val")
    
    train_data.save_to_disk(train_path)
    val_data.save_to_disk(val_path)
    
    yield {
        "train_path": train_path,
        "val_path": val_path,
        "train_data": train_data,
        "val_data": val_data
    }
    
    # Cleanup
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

def test_training_setup(mock_wandb, test_dataset, mock_env_vars):
    """Test training setup workflow"""
    # Setup model and trainer
    model_handler, trainer = setup_model_and_trainer(
        model_name="unsloth/Qwen2.5-7B",
        output_dir=TEST_DATA_DIR,
        model_source=ModelSource.UNSLOTH,
        num_train_epochs=1,
        per_device_train_batch_size=4
    )
    
    # Verify trainer configuration
    assert isinstance(trainer.prompt_creator, PromptCreator)
    assert trainer.prompt_creator.prompt_type == PromptCreator.TEACHER_REASONED
    assert trainer.lora_config is not None
    assert trainer.lora_config.r == 8
    assert trainer.lora_config.lora_alpha == 32

def test_wandb_integration(mock_wandb, mock_env_vars):
    """Test Weights & Biases integration"""
    logger = setup_wandb_logging(
        project_name="test_project",
        run_name="test_run"
    )
    
    assert isinstance(logger, WandBLogger)
    mock_wandb["init"].assert_called_once()

def test_full_training_workflow(mock_wandb, mock_unsloth, test_dataset, mock_env_vars):
    """Test the complete training workflow"""
    results = train(
        model_name="unsloth/Qwen2.5-7B",
        train_path=test_dataset["train_path"],
        val_path=test_dataset["val_path"],
        output_dir=TEST_DATA_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        model_source=ModelSource.UNSLOTH
    )
    
    # Verify training completed
    assert results is not None
    
    # Verify model checkpoints were saved
    checkpoint_dir = os.path.join(TEST_DATA_DIR, "checkpoints")
    assert os.path.exists(checkpoint_dir)

def test_inference_workflow(mock_unsloth, mock_env_vars):
    """Test the inference workflow"""
    model_handler = QwenModelHandler(
        model_name="unsloth/Qwen2.5-7B",
        model_source=ModelSource.UNSLOTH,
        quantization="4bit",
        max_seq_length=2048
    )
    
    # Test generation
    prompt = "What is 2+2?"
    response = model_handler.generate_with_streaming(
        prompt=prompt,
        temperature=0.0,
        stream=False
    )
    assert response is not None
    
    # Test streaming generation
    streamer = model_handler.generate_with_streaming(
        prompt=prompt,
        temperature=0.0,
        stream=True
    )
    assert hasattr(streamer, "__iter__")

def test_error_handling(mock_env_vars):
    """Test error handling in the workflow"""
    # Test invalid model source
    with pytest.raises(ValueError):
        QwenModelHandler(
            model_name="test",
            model_source="invalid"
        )
    
    # Test invalid training paths
    with pytest.raises(ValueError, match="Train path does not exist"):
        train(
            model_name="test",
            train_path="nonexistent",
            output_dir=TEST_DATA_DIR
        )
    
    # Test invalid validation strategy
    with pytest.raises(ValueError, match="Invalid validation strategy"):
        train(
            model_name="test",
            train_path=TEST_DATA_DIR,
            validation_strategy="invalid"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 