"""
Base callback module with common imports and utilities.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# Setup logger
logger = logging.getLogger(__name__)


class BaseCallback(TrainerCallback):
    """Base class for all callbacks with common utilities."""

    def __init__(self):
        self.trainer = None

    def _log_to_wandb(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """Helper method to log metrics to wandb."""
        try:
            import wandb

            if wandb.run:
                if prefix:
                    metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
                wandb.log(metrics, step=step)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def _save_to_file(self, data: Dict[str, Any], filepath: str):
        """Helper method to save data to a file."""
        try:
            import json

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to file {filepath}: {e}")

    def _get_output_dir(self, args: TrainingArguments) -> str:
        """Helper method to get and create output directory."""
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
