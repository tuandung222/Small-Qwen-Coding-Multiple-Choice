import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation
    """
    max_length: int = 100
    min_length: int = 1
    num_beams: int = 1
    num_return_sequences: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False
    do_sample: bool = False
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bad_words_ids: Optional[List[List[int]]] = None


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
) -> List[str]:
    """
    Generate text from a prompt
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for encoding/decoding
        prompt: Input prompt
        config: Generation configuration
        device: Device to generate on
        
    Returns:
        List[str]: Generated text sequences
    """
    try:
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_length,
                min_length=config.min_length,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id or tokenizer.pad_token_id,
                eos_token_id=config.eos_token_id or tokenizer.eos_token_id,
                bad_words_ids=config.bad_words_ids,
            )
            
        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
        
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise


def generate_with_guidance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> List[str]:
    """
    Generate text with classifier-free guidance
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for encoding/decoding
        prompt: Input prompt
        config: Generation configuration
        device: Device to generate on
        guidance_scale: Scale for classifier-free guidance
        num_inference_steps: Number of inference steps
        
    Returns:
        List[str]: Generated text sequences
    """
    try:
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate with guidance
        with torch.no_grad():
            # Get unconditional embeddings
            uncond_inputs = tokenizer([""], return_tensors="pt").to(device)
            uncond_embeddings = model.get_input_embeddings()(uncond_inputs.input_ids)
            
            # Get conditional embeddings
            cond_embeddings = model.get_input_embeddings()(inputs.input_ids)
            
            # Concatenate embeddings
            embeddings = torch.cat([uncond_embeddings, cond_embeddings])
            
            # Generate
            outputs = model.generate(
                inputs_embeds=embeddings,
                max_length=config.max_length,
                min_length=config.min_length,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id or tokenizer.pad_token_id,
                eos_token_id=config.eos_token_id or tokenizer.eos_token_id,
                bad_words_ids=config.bad_words_ids,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
        
    except Exception as e:
        logger.error(f"Error generating text with guidance: {str(e)}")
        raise


def generate_with_control(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
    control_codes: List[str],
) -> List[str]:
    """
    Generate text with control codes
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for encoding/decoding
        prompt: Input prompt
        config: Generation configuration
        device: Device to generate on
        control_codes: List of control codes to apply
        
    Returns:
        List[str]: Generated text sequences
    """
    try:
        # Encode prompt and control codes
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        control_inputs = tokenizer(control_codes, return_tensors="pt", padding=True).to(device)
        
        # Generate with control
        with torch.no_grad():
            # Get control embeddings
            control_embeddings = model.get_input_embeddings()(control_inputs.input_ids)
            
            # Get prompt embeddings
            prompt_embeddings = model.get_input_embeddings()(inputs.input_ids)
            
            # Concatenate embeddings
            embeddings = torch.cat([control_embeddings, prompt_embeddings])
            
            # Generate
            outputs = model.generate(
                inputs_embeds=embeddings,
                max_length=config.max_length,
                min_length=config.min_length,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id or tokenizer.pad_token_id,
                eos_token_id=config.eos_token_id or tokenizer.eos_token_id,
                bad_words_ids=config.bad_words_ids,
            )
            
        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
        
    except Exception as e:
        logger.error(f"Error generating text with control: {str(e)}")
        raise


def generate_with_retrieval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    config: GenerationConfig,
    device: torch.device,
    retrieval_model: nn.Module,
    retrieval_database: List[str],
    num_retrievals: int = 3,
) -> List[str]:
    """
    Generate text with retrieval augmentation
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer to use for encoding/decoding
        prompt: Input prompt
        config: Generation configuration
        device: Device to generate on
        retrieval_model: Model to use for retrieval
        retrieval_database: Database of texts to retrieve from
        num_retrievals: Number of texts to retrieve
        
    Returns:
        List[str]: Generated text sequences
    """
    try:
        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Retrieve relevant texts
        with torch.no_grad():
            # Get prompt embeddings
            prompt_embeddings = retrieval_model.get_embeddings(inputs.input_ids)
            
            # Get database embeddings
            database_inputs = tokenizer(retrieval_database, return_tensors="pt", padding=True).to(device)
            database_embeddings = retrieval_model.get_embeddings(database_inputs.input_ids)
            
            # Compute similarities
            similarities = torch.matmul(prompt_embeddings, database_embeddings.t())
            
            # Get top retrievals
            top_k = min(num_retrievals, len(retrieval_database))
            top_k_values, top_k_indices = torch.topk(similarities, top_k)
            
            # Get retrieved texts
            retrieved_texts = [retrieval_database[i] for i in top_k_indices[0]]
            
        # Generate with retrieved texts
        with torch.no_grad():
            # Encode retrieved texts
            retrieved_inputs = tokenizer(retrieved_texts, return_tensors="pt", padding=True).to(device)
            
            # Get retrieved embeddings
            retrieved_embeddings = model.get_input_embeddings()(retrieved_inputs.input_ids)
            
            # Get prompt embeddings
            prompt_embeddings = model.get_input_embeddings()(inputs.input_ids)
            
            # Concatenate embeddings
            embeddings = torch.cat([retrieved_embeddings, prompt_embeddings])
            
            # Generate
            outputs = model.generate(
                inputs_embeds=embeddings,
                max_length=config.max_length,
                min_length=config.min_length,
                num_beams=config.num_beams,
                num_return_sequences=config.num_return_sequences,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=config.early_stopping,
                do_sample=config.do_sample,
                use_cache=config.use_cache,
                pad_token_id=config.pad_token_id or tokenizer.pad_token_id,
                eos_token_id=config.eos_token_id or tokenizer.eos_token_id,
                bad_words_ids=config.bad_words_ids,
            )
            
        # Decode outputs
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_texts
        
    except Exception as e:
        logger.error(f"Error generating text with retrieval: {str(e)}")
        raise 