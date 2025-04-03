import logging
import torch
from typing import Dict, Any, Optional


from src.config.training_config import AttentionConfig

logger = logging.getLogger(__name__)


def setup_attention(
    model: torch.nn.Module,
    attention_config: AttentionConfig,
) -> None:
    """
    Setup attention configuration for the model
    
    Args:
        model: Model to configure
        attention_config: Attention configuration
    """
    try:
        if attention_config.attention_mode == "flash_attention_2":
            from flash_attn import flash_attn_func
            
            # Replace attention function with flash attention
            for module in model.modules():
                if hasattr(module, "attention"):
                    module.attention = flash_attn_func
                    
            logger.info("Using Flash Attention 2")
            
        elif attention_config.attention_mode == "xformers":
            import xformers.ops as xops
            
            # Replace attention function with xformers
            for module in model.modules():
                if hasattr(module, "attention"):
                    module.attention = xops.memory_efficient_attention
                    
            logger.info("Using xFormers attention")
            
        elif attention_config.attention_mode == "scaled_dot_product":
            # Use default scaled dot product attention
            logger.info("Using scaled dot product attention")
            
        else:
            raise ValueError(f"Unknown attention mode: {attention_config.attention_mode}")
            
    except Exception as e:
        logger.error(f"Error setting up attention: {str(e)}")
        raise


def get_attention_mask(
    input_ids: torch.Tensor,
    padding_token_id: int,
) -> torch.Tensor:
    """
    Get attention mask for input ids
    
    Args:
        input_ids: Input token ids
        padding_token_id: ID of padding token
        
    Returns:
        torch.Tensor: Attention mask
    """
    try:
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = (input_ids != padding_token_id).long()
        
        return attention_mask
        
    except Exception as e:
        logger.error(f"Error creating attention mask: {str(e)}")
        raise


def apply_attention_mask(
    attention_scores: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_value: float = -10000.0,
) -> torch.Tensor:
    """
    Apply attention mask to attention scores
    
    Args:
        attention_scores: Attention scores
        attention_mask: Attention mask
        mask_value: Value to use for masked positions
        
    Returns:
        torch.Tensor: Masked attention scores
    """
    try:
        # Expand attention mask to match attention scores shape
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply mask
        masked_scores = attention_scores.masked_fill(
            expanded_mask == 0,
            mask_value,
        )
        
        return masked_scores
        
    except Exception as e:
        logger.error(f"Error applying attention mask: {str(e)}")
        raise


def compute_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_prob: float = 0.1,
) -> tuple:
    """
    Compute attention weights
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        attention_mask: Optional attention mask
        dropout_prob: Dropout probability
        
    Returns:
        tuple: Attention weights and output
    """
    try:
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = apply_attention_mask(attention_scores, attention_mask)
            
        # Apply softmax
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = torch.nn.functional.dropout(attention_weights, p=dropout_prob)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return attention_weights, output
        
    except Exception as e:
        logger.error(f"Error computing attention weights: {str(e)}")
        raise 