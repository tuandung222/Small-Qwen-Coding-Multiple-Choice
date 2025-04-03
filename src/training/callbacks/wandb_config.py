"""
Configuration class for Weights & Biases logging.
"""

from typing import Any, Dict, List, Optional


class WandBConfig:
    """
    Configuration for W&B logging with enhanced options.

    Features:
    1. Project and run configuration
    2. Memory usage tracking
    3. Gradient tracking
    4. Training metrics logging
    5. Validation metrics logging
    6. Example logging
    """

    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_memory: bool = True,
        log_gradients: bool = True,
        log_training: bool = True,
        log_validation: bool = True,
        log_examples: bool = True,
        log_interval: int = 100,
        example_batch_size: int = 5,
    ):
        """
        Initialize WandB configuration.

        Args:
            project_name: Name of the W&B project
            run_name: Optional name for this run
            entity: Optional W&B entity (username or team)
            tags: Optional list of tags for the run
            notes: Optional notes about the run
            config: Optional dictionary of run configuration
            log_memory: Whether to log memory usage
            log_gradients: Whether to log gradient statistics
            log_training: Whether to log training metrics
            log_validation: Whether to log validation metrics
            log_examples: Whether to log example predictions
            log_interval: Steps between logging
            example_batch_size: Number of examples to log per batch
        """
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.config = config or {}

        # Logging options
        self.log_memory = log_memory
        self.log_gradients = log_gradients
        self.log_training = log_training
        self.log_validation = log_validation
        self.log_examples = log_examples

        # Logging parameters
        self.log_interval = log_interval
        self.example_batch_size = example_batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "project_name": self.project_name,
            "run_name": self.run_name,
            "entity": self.entity,
            "tags": self.tags,
            "notes": self.notes,
            "config": self.config,
            "log_memory": self.log_memory,
            "log_gradients": self.log_gradients,
            "log_training": self.log_training,
            "log_validation": self.log_validation,
            "log_examples": self.log_examples,
            "log_interval": self.log_interval,
            "example_batch_size": self.example_batch_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WandBConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
