import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login as hf_login

import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_env_file(start_path: str = None) -> str:
    """
    Find the .env file by searching up the directory tree.

    Args:
        start_path: Starting directory to search from. If None, uses current directory.

    Returns:
        str: Path to the .env file if found, None otherwise.
    """
    if start_path is None:
        start_path = os.getcwd()

    current_path = Path(start_path).resolve()

    # Search up to 5 levels in the directory tree
    for _ in range(5):
        env_path = current_path / ".env"
        if env_path.exists():
            return str(env_path)
        current_path = current_path.parent

    return None


def setup_authentication():
    """
    Setup authentication for Hugging Face Hub and Weights & Biases.
    Loads credentials from .env file and logs in to both services.

    Raises:
        ValueError: If required environment variables are not set
        FileNotFoundError: If .env file cannot be found
    """
    # Find and load .env file
    env_path = find_env_file()
    if not env_path:
        raise FileNotFoundError(
            "Could not find .env file in current directory or parent directories"
        )

    logger.info(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)

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
        logger.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face Hub: {e}")
        raise

    # Login to Weights & Biases
    try:
        wandb.login(key=wandb_key)
        logger.info("Successfully logged in to Weights & Biases")
    except Exception as e:
        logger.error(f"Failed to login to Weights & Biases: {e}")
        raise


if __name__ == "__main__":
    # Test the authentication setup
    try:
        setup_authentication()
        print("Authentication setup successful!")
    except Exception as e:
        print(f"Authentication setup failed: {e}")
