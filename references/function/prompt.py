import logging
from typing import Dict, Any, List, Optional
from src.config.training_config import PromptConfig, ResponseOnlyConfig

logger = logging.getLogger(__name__)


def format_prompt(
    instruction: str,
    input_text: Optional[str] = None,
    prompt_config: PromptConfig = None,
    response_only_config: ResponseOnlyConfig = None,
) -> str:
    """
    Format a prompt for the model
    
    Args:
        instruction: The instruction text
        input_text: Optional input text
        prompt_config: Prompt configuration
        response_only_config: Response-only configuration
        
    Returns:
        str: Formatted prompt
    """
    try:
        if prompt_config is None:
            prompt_config = PromptConfig()
            
        if response_only_config is None:
            response_only_config = ResponseOnlyConfig()
            
        if response_only_config.enabled:
            return instruction
            
        if input_text:
            return (
                prompt_config.instruction_template.format(instruction=instruction)
                + prompt_config.input_template.format(input=input_text)
            )
        else:
            return prompt_config.instruction_template.format(instruction=instruction)
            
    except Exception as e:
        logger.error(f"Error formatting prompt: {str(e)}")
        raise


def format_multiple_choice_prompt(
    instruction: str,
    choices: List[str],
    prompt_config: PromptConfig = None,
) -> str:
    """
    Format a multiple choice prompt
    
    Args:
        instruction: The instruction text
        choices: List of choices
        prompt_config: Prompt configuration
        
    Returns:
        str: Formatted prompt
    """
    try:
        if prompt_config is None:
            prompt_config = PromptConfig()
            
        # Format choices
        choices_text = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))
        
        # Combine instruction and choices
        full_instruction = f"{instruction}\n\nChoices:\n{choices_text}"
        
        return prompt_config.instruction_template.format(instruction=full_instruction)
        
    except Exception as e:
        logger.error(f"Error formatting multiple choice prompt: {str(e)}")
        raise


def format_coding_prompt(
    instruction: str,
    code: str,
    language: str,
    prompt_config: PromptConfig = None,
) -> str:
    """
    Format a coding prompt
    
    Args:
        instruction: The instruction text
        code: Code to include
        language: Programming language
        prompt_config: Prompt configuration
        
    Returns:
        str: Formatted prompt
    """
    try:
        if prompt_config is None:
            prompt_config = PromptConfig()
            
        # Format code block
        code_block = f"```{language}\n{code}\n```"
        
        # Combine instruction and code
        full_instruction = f"{instruction}\n\nCode:\n{code_block}"
        
        return prompt_config.instruction_template.format(instruction=full_instruction)
        
    except Exception as e:
        logger.error(f"Error formatting coding prompt: {str(e)}")
        raise


def format_debug_prompt(
    instruction: str,
    code: str,
    error: str,
    language: str,
    prompt_config: PromptConfig = None,
) -> str:
    """
    Format a debugging prompt
    
    Args:
        instruction: The instruction text
        code: Code to debug
        error: Error message
        language: Programming language
        prompt_config: Prompt configuration
        
    Returns:
        str: Formatted prompt
    """
    try:
        if prompt_config is None:
            prompt_config = PromptConfig()
            
        # Format code block and error
        code_block = f"```{language}\n{code}\n```"
        error_block = f"Error:\n{error}"
        
        # Combine instruction, code, and error
        full_instruction = f"{instruction}\n\nCode:\n{code_block}\n\n{error_block}"
        
        return prompt_config.instruction_template.format(instruction=full_instruction)
        
    except Exception as e:
        logger.error(f"Error formatting debug prompt: {str(e)}")
        raise 