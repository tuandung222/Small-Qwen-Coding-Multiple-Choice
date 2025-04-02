import os
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
import wandb


# This file is project/src/utils/auth.py
# The env file is in project or in src




def setup_authentication():
    """
    Setup authentication for Hugging Face Hub and Weights & Biases.
    Loads credentials from .env file and logs in to both services.
    
    Raises:
        ValueError: If required environment variables are not set
    """
    # Load environment variables from possible locations
    # Try to find .env file in parent directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_levels = 5
    
    # Start from current directory and move up to find .env
    for i in range(max_levels):
        search_dir = os.path.abspath(os.path.join(current_dir, '../' * i))
        env_path = os.path.join(search_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    else:
        # If no .env file found, try default behavior
        load_dotenv()
    # Get API keys
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")
    
    # Validate environment variables
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    
    # Login to Hugging Face Hub
    try:
        hf_login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        print(f"Failed to login to Hugging Face Hub: {e}")
        raise
    
    # Login to Weights & Biases
    try:
        wandb.login(key=wandb_key)
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"Failed to login to Weights & Biases: {e}")
        raise 