import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import huggingface_hub
import torch

try:
    import unsloth  # Import unsloth first to apply all optimizations and avoid warnings
except ImportError:
    pass
from huggingface_hub import HfApi, snapshot_download, upload_folder
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
except ImportError:
    pass

from src.utils.auth import setup_authentication
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSource(str, Enum):
    """Model source enumeration"""

    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"


@dataclass
class HubConfig:
    """Configuration for Hugging Face Hub integration"""

    model_id: str
    token: Optional[str] = None
    private: bool = False
    save_method: str = "lora"  # lora, merged_16bit, merged_4bit, gguf


class QwenModelHandler:
    """Handles loading, configuration, and inference with Qwen models"""

    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        quantization: Union[str, BitsAndBytesConfig] = "4bit",
        model_source: str = ModelSource.HUGGINGFACE,
        device_map: str = "auto",
        source_hub_config: Optional[HubConfig] = None,
        destination_hub_config: Optional[HubConfig] = None,
        attn_implementation: str = "default",
        force_attn_implementation: bool = False,
    ):
        """
        Initialize a Qwen model handler.

        Args:
            model_name: Name or path of the model to load
            max_seq_length: Maximum sequence length for tokenizer and model
            quantization: Quantization level (4bit, 8bit, or none) or BitsAndBytesConfig object
            model_source: Source of the model (huggingface or unsloth)
            device_map: Device mapping strategy for the model
            source_hub_config: Configuration for the source model on Hugging Face Hub
            destination_hub_config: Configuration for the destination model on Hugging Face Hub
            attn_implementation: Attention implementation to use (default, flash_attention_2, sdpa, eager, xformers)
            force_attn_implementation: Whether to force the attention implementation even if not optimal
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.quantization = quantization
        self.model_source = model_source
        self.device_map = device_map
        self.source_hub_config = source_hub_config
        self.destination_hub_config = destination_hub_config
        self.attn_implementation = attn_implementation
        self.force_attn_implementation = force_attn_implementation

        # Initialize model and tokenizer
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # Log model configuration
        logger.info(f"Loading {model_name} from {model_source}, max_seq_length={max_seq_length}")

        # Load the model based on the source
        self._load_model()

    def _check_attention_support(self):
        """Check if the specified attention implementation is supported on the current hardware"""
        # Check for Flash Attention 2 support
        has_flash_attn = False
        try:
            import flash_attn

            has_flash_attn = True
            logger.info("Flash Attention 2 is available (package flash-attn detected)")
            # Check flash_attn version
            try:
                logger.info(f"Flash Attention 2 version: {flash_attn.__version__}")
            except AttributeError:
                logger.info("Flash Attention 2 version information not available")
        except ImportError:
            logger.info("Flash Attention 2 is not available (package flash-attn not found)")
            if self.attn_implementation == "flash_attention_2":
                logger.info("To install: pip install flash-attn --no-build-isolation")

        # Check for xFormers support
        has_xformers = False
        try:
            import xformers

            has_xformers = True
            try:
                logger.info(f"xFormers is available (version: {xformers.__version__})")
            except AttributeError:
                logger.info("xFormers is available (version information not available)")
        except ImportError:
            logger.info("xFormers is not available (package not found)")
            if self.attn_implementation == "xformers":
                logger.info("To install: pip install xformers")

        # Check for CUDA availability for SDPA
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            try:
                cuda_version = torch.version.cuda
                logger.info(f"CUDA is available (version: {cuda_version})")
                # Check if CUDA version is sufficient for SDPA
                if self.attn_implementation == "sdpa" and cuda_version:
                    major, minor = map(int, cuda_version.split(".")[:2])
                    if major < 11 or (major == 11 and minor < 6):
                        logger.warning(f"SDPA works best with CUDA 11.6+, current: {cuda_version}")
            except:
                logger.info("CUDA is available (version information not available)")
        else:
            logger.info("CUDA is not available")

        # Check PyTorch version for SDPA
        if self.attn_implementation == "sdpa":
            from packaging import version

            torch_version = torch.__version__
            if version.parse(torch_version) < version.parse("2.0.0"):
                logger.warning(f"SDPA requires PyTorch 2.0+, current: {torch_version}")
                if not self.force_attn_implementation:
                    logger.warning("Falling back to default attention implementation")
                    return "default"

        # Return available implementations
        if self.attn_implementation == "flash_attention_2" and not has_flash_attn:
            if self.force_attn_implementation:
                logger.warning(
                    "Flash Attention 2 was requested but is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "Flash Attention 2 was requested but is not available. Falling back to default."
                )
                return "default"

        if self.attn_implementation == "xformers" and not has_xformers:
            if self.force_attn_implementation:
                logger.warning(
                    "xFormers was requested but is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "xFormers was requested but is not available. Falling back to default."
                )
                return "default"

        if self.attn_implementation == "sdpa" and not has_cuda:
            if self.force_attn_implementation:
                logger.warning(
                    "SDPA was requested but CUDA is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "SDPA was requested but CUDA is not available. Falling back to default."
                )
                return "default"

        logger.info(f"Using attention implementation: {self.attn_implementation}")
        return self.attn_implementation

    def _load_model(self):
        """Load the model and tokenizer based on the specified source"""
        try:
            if self.model_source == ModelSource.UNSLOTH:
                self._load_from_unsloth()
            else:
                self._load_from_huggingface()

            # Ensure tokenizer has pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Log model info
            logger.info(f"Model loaded successfully: {self.model_name}")
            if hasattr(self.model, "config"):
                logger.info(f"Model type: {self.model.config.model_type}")
                for key, value in self.model.config.to_dict().items():
                    if key in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "torch_dtype",
                    ]:
                        logger.info(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_from_huggingface(self):
        """Load model from HuggingFace Hub"""
        # Configure quantization
        quantization_config = None
        if isinstance(self.quantization, str):
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif isinstance(self.quantization, BitsAndBytesConfig):
            quantization_config = self.quantization

        # Check attention implementation
        attn_implementation = self._check_attention_support()

        model_kwargs = {
            "device_map": self.device_map,
            "token": self.source_hub_config.token if self.source_hub_config else None,
            "trust_remote_code": True,
        }

        # Add quantization config if specified
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Add attention implementation if not default
        if attn_implementation != "default":
            model_kwargs["attn_implementation"] = attn_implementation
            logger.info(f"Using attention implementation: {attn_implementation}")

        # Import AutoModelForCausalLM here to avoid early import
        from transformers import AutoModelForCausalLM

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.source_hub_config.token if self.source_hub_config else None,
            trust_remote_code=True,
            padding_side="right",
            model_max_length=self.max_seq_length,
        )

    def _load_from_unsloth(self):
        """Load model with Unsloth optimization"""
        try:
            # Import unsloth here to avoid early import
            from unsloth import FastLanguageModel

            # Check attention implementation
            attn_implementation = self._check_attention_support()

            # Determine max memory
            max_memory = None
            if torch.cuda.is_available():
                # Use 85% of available GPU memory
                max_memory = {
                    0: f"{int(torch.cuda.get_device_properties(0).total_memory * 0.85 / 1024 / 1024)}MiB"
                }
                logger.info(f"Setting max memory: {max_memory}")

            # Setup model args
            model_args = {
                "max_seq_length": self.max_seq_length,
                "device_map": self.device_map,
            }

            # Add quantization config
            if isinstance(self.quantization, str):
                if self.quantization == "4bit":
                    model_args["load_in_4bit"] = True
                elif self.quantization == "8bit":
                    model_args["load_in_8bit"] = True
            elif isinstance(self.quantization, BitsAndBytesConfig):
                if self.quantization.load_in_4bit:
                    model_args["load_in_4bit"] = True
                elif self.quantization.load_in_8bit:
                    model_args["load_in_8bit"] = True

            # Add attention implementation if not default
            if attn_implementation != "default":
                model_args["attn_implementation"] = attn_implementation
                logger.info(f"Using attention implementation: {attn_implementation}")

            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                token=self.source_hub_config.token if self.source_hub_config else None,
                max_memory=max_memory,
                **model_args,
            )

        except ImportError:
            logger.error("Unsloth import failed. Please install unsloth with: pip install unsloth")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation

        Returns:
            str: The generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :].strip()
        return response

    def generate_with_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        min_p: float = 0.1,
        stream: bool = True,
    ):
        """
        Generate a response from the model with streaming support.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation
            min_p: Minimum probability for sampling (recommended 0.1)
            stream: Whether to stream the output or return the full response

        Returns:
            If stream=True: TextIteratorStreamer object that yields tokens as they're generated
            If stream=False: Complete response as string
        """
        import threading

        from transformers import TextIteratorStreamer

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            pass

        # Enable faster inference if using Unsloth
        if self.model_source == ModelSource.UNSLOTH:
            # try:
            FastLanguageModel.for_inference(self.model)
            # except ImportError:
            #     pass

        # Format the prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(inputs)

        if stream:
            # Use TextIteratorStreamer for streaming output
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation args
            generation_args = {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "min_p": min_p,
                "use_cache": True,
            }

            # Start generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_args)
            thread.start()

            # Return the streamer object
            return streamer
        else:
            # Generate without streaming
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_p=min_p,
                use_cache=True,
            )

            # Decode the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            response = response[len(prompt_text) :].strip()

            return response

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: A list of input prompts
            max_new_tokens: Maximum number of tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation

        Returns:
            List[str]: A list of generated text responses
        """
        # Ensure tokenizer has padding token and set padding side
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left" # Use left padding for batch generation

        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length - max_new_tokens, # Ensure space for generation
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_ids_len = inputs["input_ids"].shape[1]

        # Generate responses for the batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id, # Explicitly set EOS token
            )

        # Decode the generated sequences, skipping special tokens
        # Slice outputs to only contain generated tokens (excluding prompt)
        generated_ids = outputs[:, input_ids_len:]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        return responses

    def calculate_perplexity(self, prompt: str, answer: str, temperature: float = 0.0) -> float:
        """
        Calculate perplexity of the given answer for a prompt.

        Args:
            prompt: The input prompt
            answer: The answer to evaluate
            temperature: Sampling temperature

        Returns:
            float: Perplexity score (lower is better)
        """
        import math

        # Combine prompt and answer
        full_text = prompt + answer

        # Tokenize
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.model.device)
        target_ids = input_ids.clone()

        # Determine where the answer starts
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]

        # Set prompt part to -100 so it's ignored in loss calculation
        target_ids[:, :prompt_length] = -100

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss.item()

        # Count tokens in answer
        answer_length = target_ids.shape[1] - prompt_length

        # Calculate perplexity: exp(average negative log-likelihood)
        perplexity = math.exp(neg_log_likelihood)

        return perplexity

    def calculate_answer_loss(self, prompt: str, answer: str) -> float:
        """
        Calculate the loss specifically on the answer portion of the text.

        Args:
            prompt: The input prompt
            answer: The answer to evaluate

        Returns:
            float: Loss value for the answer
        """
        # Combine prompt and answer
        full_text = prompt + answer

        # Tokenize
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.model.device)
        target_ids = input_ids.clone()

        # Determine where the answer starts
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]

        # Set prompt part to -100 so it's ignored in loss calculation
        target_ids[:, :prompt_length] = -100

        # Calculate loss on answer only
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss.item()

        return loss

    def save_to_hub(self, hub_config: HubConfig, merge_adapter: bool = False):
        """
        Save model to Hugging Face Hub.

        Args:
            hub_config: Configuration for Hub saving
            merge_adapter: Whether to merge the adapter weights before saving

        Returns:
            str: URL of the saved model on the Hub
        """
        try:
            logger.info(f"Saving model to {hub_config.model_id}...")

            # Create repository if needed
            if hub_config.token:
                from huggingface_hub import create_repo

                try:
                    create_repo(
                        hub_config.model_id, private=hub_config.private, token=hub_config.token
                    )
                    logger.info(f"Created repository: {hub_config.model_id}")
                except Exception as e:
                    # Repository likely already exists
                    logger.info(f"Repository exists or couldn't be created: {str(e)}")

            # Save based on method
            if hub_config.save_method == "lora":
                # Save LoRA adapter only
                if hasattr(self.model, "peft_config"):
                    logger.info("Saving LoRA adapter...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                    # Save tokenizer
                    self.tokenizer.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )
                else:
                    logger.warning("Model doesn't have LoRA adapter, saving full model...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

            elif hub_config.save_method == "merged_16bit":
                # Merge adapter and save in 16-bit
                if hasattr(self.model, "merge_and_unload"):
                    logger.info("Merging adapter and saving in 16-bit...")
                    merged_model = self.model.merge_and_unload()
                    merged_model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                    # Save tokenizer
                    self.tokenizer.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )
                else:
                    logger.warning("Model doesn't support merge_and_unload, saving as is...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

            elif hub_config.save_method == "merged_4bit":
                # Create optimized 4-bit model
                logger.info("Saving 4-bit quantized model is not fully supported yet")
                logger.info("Falling back to standard saving...")
                self.model.save_pretrained(
                    hub_config.model_id, token=hub_config.token, push_to_hub=True
                )

            elif hub_config.save_method == "gguf":
                logger.warning("GGUF export not yet supported, saving in standard format")
                self.model.save_pretrained(
                    hub_config.model_id, token=hub_config.token, push_to_hub=True
                )

            else:
                raise ValueError(f"Unsupported save method: {hub_config.save_method}")

            # Generate model URL
            hf_hub_url = f"https://huggingface.co/{hub_config.model_id}"
            logger.info(f"Model saved successfully to {hf_hub_url}")

            return hf_hub_url

        except Exception as e:
            logger.error(f"Error saving model to Hub: {str(e)}")
            raise

    def save_model(self, output_dir: str, save_method: str = "lora") -> str:
        """
        Save model to disk

        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_16bit", "merged_4bit", "gguf")

        Returns:
            Path to saved model
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.model_source == ModelSource.UNSLOTH:
            # Use Unsloth's saving methods
            if save_method == "lora":
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            elif save_method == "merged_16bit":
                self.model.save_pretrained_merged(
                    output_dir, self.tokenizer, save_method="merged_16bit"
                )
            elif save_method == "merged_4bit":
                self.model.save_pretrained_merged(
                    output_dir, self.tokenizer, save_method="merged_4bit"
                )
            elif save_method == "gguf":
                self.model.save_pretrained_gguf(
                    output_dir, self.tokenizer, quantization_method="q4_k_m"
                )
            else:
                raise ValueError(f"Unknown save method: {save_method}")
        else:
            # Use Hugging Face's saving methods
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir} using method {save_method}")
        return output_dir

    def push_to_hub(self, hub_config: HubConfig) -> str:
        """
        Push model to Hugging Face Hub

        Args:
            hub_config: Configuration for pushing to HuggingFace Hub

        Returns:
            URL of the pushed model
        """
        if self.model_source == ModelSource.UNSLOTH:
            # Use Unsloth's hub methods
            if hub_config.save_method == "lora":
                self.model.push_to_hub_merged(
                    hub_config.model_id, self.tokenizer, save_method="lora", token=hub_config.token
                )
            elif hub_config.save_method == "merged_16bit":
                self.model.push_to_hub_merged(
                    hub_config.model_id,
                    self.tokenizer,
                    save_method="merged_16bit",
                    token=hub_config.token,
                )
            elif hub_config.save_method == "merged_4bit":
                self.model.push_to_hub_merged(
                    hub_config.model_id,
                    self.tokenizer,
                    save_method="merged_4bit",
                    token=hub_config.token,
                )
            elif hub_config.save_method == "gguf":
                self.model.push_to_hub_gguf(
                    hub_config.model_id,
                    self.tokenizer,
                    quantization_method=["q4_k_m", "q5_k_m"],
                    token=hub_config.token,
                )
            else:
                raise ValueError(f"Unknown save method: {hub_config.save_method}")
        else:
            # Use Hugging Face's hub methods
            self.model.push_to_hub(
                hub_config.model_id, token=hub_config.token, private=hub_config.private
            )
            self.tokenizer.push_to_hub(
                hub_config.model_id, token=hub_config.token, private=hub_config.private
            )

        hub_url = f"https://huggingface.co/{hub_config.model_id}"
        print(f"Model successfully pushed to: {hub_url}")
        return hub_url
