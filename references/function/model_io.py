import logging
import os
import json
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from huggingface_hub import HfApi, Repository

logger = logging.getLogger(__name__)


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    model_name: Optional[str] = None,
    save_format: str = "pytorch",
    save_optimizer: bool = False,
    save_scheduler: bool = False,
    save_rng: bool = False,
    save_config: bool = True,
    save_tokenizer: bool = True,
    save_metadata: bool = True,
) -> str:
    """
    Save a model and its components
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        model_name: Name of the model
        save_format: Format to save in (pytorch, safetensors)
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        save_rng: Whether to save RNG state
        save_config: Whether to save model config
        save_tokenizer: Whether to save tokenizer
        save_metadata: Whether to save metadata
        
    Returns:
        str: Path to the saved model
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if save_format == "safetensors":
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        else:
            model.save_pretrained(output_dir)
            
        # Save tokenizer
        if save_tokenizer:
            tokenizer.save_pretrained(output_dir)
            
        # Save optimizer state
        if save_optimizer and hasattr(model, "optimizer"):
            torch.save(model.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            
        # Save scheduler state
        if save_scheduler and hasattr(model, "scheduler"):
            torch.save(model.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            
        # Save RNG state
        if save_rng:
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pt"))
            
        # Save metadata
        if save_metadata:
            metadata = {
                "model_name": model_name or model.config.model_type,
                "save_time": datetime.now().isoformat(),
                "save_format": save_format,
                "model_config": model.config.to_dict() if save_config else None,
            }
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Saved model to {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(
    model_path: str,
    model_class: Optional[type] = None,
    device: Optional[torch.device] = None,
    load_optimizer: bool = False,
    load_scheduler: bool = False,
    load_rng: bool = False,
    load_config: bool = True,
    load_tokenizer: bool = True,
    load_metadata: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]:
    """
    Load a model and its components
    
    Args:
        model_path: Path to the model
        model_class: Class to use for loading the model
        device: Device to load the model to
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        load_rng: Whether to load RNG state
        load_config: Whether to load model config
        load_tokenizer: Whether to load tokenizer
        load_metadata: Whether to load metadata
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, Any]]: Model, tokenizer, and metadata
    """
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load model
        if model_class is None:
            from transformers import AutoModelForCausalLM
            model_class = AutoModelForCausalLM
            
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        
        # Load tokenizer
        tokenizer = None
        if load_tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        # Load optimizer state
        if load_optimizer and os.path.exists(os.path.join(model_path, "optimizer.pt")):
            optimizer_state = torch.load(os.path.join(model_path, "optimizer.pt"))
            if hasattr(model, "optimizer"):
                model.optimizer.load_state_dict(optimizer_state)
                
        # Load scheduler state
        if load_scheduler and os.path.exists(os.path.join(model_path, "scheduler.pt")):
            scheduler_state = torch.load(os.path.join(model_path, "scheduler.pt"))
            if hasattr(model, "scheduler"):
                model.scheduler.load_state_dict(scheduler_state)
                
        # Load RNG state
        if load_rng and os.path.exists(os.path.join(model_path, "rng_state.pt")):
            rng_states = torch.load(os.path.join(model_path, "rng_state.pt"))
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if torch.cuda.is_available() and rng_states["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_states["cuda"])
                
        # Load metadata
        metadata = {}
        if load_metadata and os.path.exists(os.path.join(model_path, "metadata.json")):
            with open(os.path.join(model_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
                
        logger.info(f"Loaded model from {model_path}")
        return model, tokenizer, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def push_to_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    repo_id: str,
    commit_message: str = "Update model",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Push a model to the Hugging Face Hub
    
    Args:
        model: Model to push
        tokenizer: Tokenizer to push
        repo_id: Repository ID
        commit_message: Commit message
        private: Whether the repository should be private
        token: Hugging Face token
        
    Returns:
        str: Repository URL
    """
    try:
        # Create repository
        api = HfApi()
        api.create_repo(repo_id, private=private, token=token, exist_ok=True)
        
        # Save model and tokenizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            
            # Push to hub
            repo = Repository(tmp_dir, clone_from=repo_id, token=token)
            repo.push_to_hub(commit_message=commit_message)
            
        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Pushed model to {repo_url}")
        return repo_url
        
    except Exception as e:
        logger.error(f"Error pushing model to hub: {str(e)}")
        raise


def download_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Download a model from the Hugging Face Hub
    
    Args:
        repo_id: Repository ID
        revision: Revision to download
        token: Hugging Face token
        cache_dir: Directory to cache downloads
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        
        logger.info(f"Downloaded model from {repo_id}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error downloading model from hub: {str(e)}")
        raise 