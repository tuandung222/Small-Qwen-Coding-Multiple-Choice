import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
import unsloth  # Import unsloth first to apply all optimizations and avoid warnings
from huggingface_hub import HfApi, snapshot_download, upload_folder
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from unsloth import FastLanguageModel

from src.utils.auth import setup_authentication


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"


class HubConfig:
    """Configuration for HuggingFace Hub interactions"""

    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        private: bool = False,
        save_method: str = "lora",
    ):
        """
        Initialize hub configuration

        Args:
            model_id: Model ID on HuggingFace Hub (e.g., 'organization/model-name')
            token: HuggingFace token for authentication
            private: Whether to make the repository private
            save_method: Method to use for saving model ("lora", "merged_16bit", "merged_4bit", "gguf")
        """
        self.model_id = model_id
        self.token = token
        self.private = private
        self.save_method = save_method


class QwenModelHandler:
    """Handler for Qwen models with inference and saving capabilities"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-7B",
        max_seq_length: int = 2048,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        model_source: str = ModelSource.HUGGINGFACE,
        max_retries: int = 3,
        retry_delay: int = 5,
        source_hub_config: Optional[HubConfig] = None,
    ):
        """
        Initialize model and tokenizer

        Args:
            model_name: Name or path of the model
            max_seq_length: Maximum sequence length for the model
            quantization: Quantization type (None, '4bit', '8bit')
            device_map: Device mapping strategy
            cache_dir: Cache directory for models
            model_source: Source of the model ("huggingface" or "unsloth")
            max_retries: Maximum number of retries for model loading
            retry_delay: Delay between retries in seconds
            source_hub_config: Configuration for loading from HuggingFace Hub
        """
        # Setup authentication
        setup_authentication()

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.model_source = ModelSource(model_source)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.source_hub_config = source_hub_config

        # Convert quantization parameter to load_in_4bit parameter
        self.load_in_4bit = quantization == "4bit"

        # Load tokenizer and model
        self.tokenizer, self.model = self._load_model_with_retry()

    def _load_model_with_retry(self):
        """Load model with retry mechanism for handling HF Hub errors"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"Retrying model load (attempt {attempt + 1}/{self.max_retries})")
                return self._load_model()
            except HfHubHTTPError as e:
                last_error = e
                if e.response.status_code == 500:
                    print(
                        f"HuggingFace Hub server error, retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    continue
                raise  # Re-raise if not a 500 error
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

        # If we've exhausted retries, try falling back to HuggingFace
        if self.model_source == ModelSource.UNSLOTH:
            print("Failed to load with Unsloth, attempting fallback to HuggingFace...")
            self.model_source = ModelSource.HUGGINGFACE
            try:
                return self._load_model()
            except Exception as e:
                print(f"Fallback also failed: {str(e)}")
                raise last_error or e

        raise last_error or RuntimeError("Failed to load model after retries")

    def _load_model(self) -> tuple:
        """Load model and tokenizer based on the specified source"""
        print(
            f"Loading {self.model_name} from {self.model_source}, max_seq_length={self.max_seq_length}"
        )

        if self.model_source == ModelSource.UNSLOTH:
            return self._load_unsloth_model()
        elif self.model_source == ModelSource.HUGGINGFACE:
            return self._load_huggingface_model()
        else:
            raise ValueError(f"Unsupported model source: {self.model_source}")

    def _load_unsloth_model(self) -> tuple:
        """Load model and tokenizer using Unsloth"""
        # Set dtype based on hardware
        dtype = None  # None for auto detection

        # Load model and tokenizer with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir=self.cache_dir,
            token=self.source_hub_config.token if self.source_hub_config else None,
        )

        return tokenizer, model

    def _load_huggingface_model(self) -> tuple:
        """Load model and tokenizer using Hugging Face"""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=self.source_hub_config.token if self.source_hub_config else None,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
            cache_dir=self.cache_dir,
            token=self.source_hub_config.token if self.source_hub_config else None,
        )

        return tokenizer, model

    def generate_with_streaming(self, prompt, temperature=0.7, max_tokens=2048, stream=True):
        """
        Generate completion with optional streaming and error handling

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the output

        Returns:
            Generated text (even when streaming is enabled)
        """
        try:
            # Enable faster inference
            FastLanguageModel.for_inference(self.model)

            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize input with proper padding
            model_inputs = self.tokenizer(
                [chat_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_attention_mask=True,
            ).to(self.model.device)

            # Add position IDs explicitly
            seq_length = model_inputs.input_ids.shape[1]
            position_ids = (
                torch.arange(seq_length, dtype=torch.int32).unsqueeze(0).to(self.model.device)
            )
            model_inputs["position_ids"] = position_ids

            if stream:
                from transformers import TextStreamer

                # Use TextStreamer for better notebook integration
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                # Generate with streaming
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    position_ids=model_inputs.position_ids,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    do_sample=temperature > 0.0,
                    use_cache=True,
                    min_p=0.1 if temperature > 0.0 else None,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

                # Decode the generated text
                generated_text = self.tokenizer.decode(
                    generated_ids.sequences[0][model_inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                return generated_text

            else:
                # Generate without streaming
                try:
                    generated_ids = self.model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        position_ids=model_inputs.position_ids,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0.0,
                        use_cache=True,
                        min_p=0.1 if temperature > 0.0 else None,
                    )
                except RuntimeError as e:
                    if "shape" in str(e) and "position" in str(e).lower():
                        # Fallback: try without explicit position IDs
                        print("Falling back to generation without explicit position IDs...")
                        generated_ids = self.model.generate(
                            input_ids=model_inputs.input_ids,
                            attention_mask=model_inputs.attention_mask,
                            temperature=temperature,
                            max_new_tokens=max_tokens,
                            do_sample=temperature > 0.0,
                            use_cache=True,
                            min_p=0.1 if temperature > 0.0 else None,
                        )
                    else:
                        raise

                # Decode the generated text
                generated_text = self.tokenizer.decode(
                    generated_ids[0][model_inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                return generated_text

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return f"Error during generation: {str(e)}"

    def calculate_perplexity(self, prompt: str, answer: str, temperature: float = 0.0) -> float:
        """
        Calculate perplexity for a prompt and answer pair

        Args:
            prompt: The input prompt
            answer: The expected answer
            temperature: Sampling temperature

        Returns:
            Perplexity score
        """
        # Format chat for perplexity calculation
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the text
        encodings = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)

        # Get loss and calculate perplexity
        neg_log_likelihood = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

        return perplexity

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
