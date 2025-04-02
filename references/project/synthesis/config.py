from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for the teacher model"""

    name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    system_prompt: str = "You are a helpful assistant that answers multiple choice questions with detailed reasoning."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding the API key"""
        config_dict = asdict(self)
        config_dict.pop("api_key")
        return config_dict


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_type: str = "reasoning"  # Options: "basic", "reasoning", "custom"
    custom_template: Optional[str] = None
    add_system_prompt: bool = True
    include_step_numbers: bool = True
    yaml_format: bool = True
    custom_template_fn: Optional[Callable] = None


@dataclass
class TestConfig:
    """Configuration for test runs"""

    output_dir: str = "./mc_test_results"
    sample_size: Optional[int] = None
    random_seed: int = 42
    save_results: bool = True
    create_visualizations: bool = True
    show_progress: bool = True
    save_individual_examples: bool = False
    question_key: str = "question"
    choices_key: str = "list_choices"
    answer_key: str = "answer"
    task_id_key: str = "task_id"
    batch_size: Optional[int] = None  # For batch processing
    verbose: bool = True
