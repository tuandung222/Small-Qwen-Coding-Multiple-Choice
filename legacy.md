# Qwen25_Coder_MultipleChoice

*   This project focuses on distilling YAML-based structured multi-step reasoning capabilities from the GPT-4o teacher model into the smaller Qwen2.5 Coder 1.5B-Instruct LLM.

*   This document provides guidance on getting started with `tuandunghcmut/Qwen25_Coder_MultipleChoice`, a model fine-tuned for multiple-choice coding questions.

*   Future plans include refactoring the project into a well-structured GitHub repository, expanding the dataset, and retraining the model using distributed training for improved scalability.

*   A demonstration notebook is available on Google Colab (click the badge below). Please note that the training code has been omitted from this notebook. It is intended solely for testing and inference using the latest checkpoint.
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Q4jtRjIkFWIAM82pAg4OBPCLjpQ8ndpI/view?usp=sharing)

*   Note: The Qwen2.5 Coder 1.5B-Instruct model might be too small for this task, and the current training dataset may be insufficient. Future iterations will explore using a larger model and more extensive data. However, the current model successfully adheres to the desired YAML format and demonstrates structured reasoning.

*   The guide below provides an explanation of the code presented in the notebook.

## Installation

First, install the required dependencies:

```bash
# Install core dependencies
pip install transformers torch pandas

# For faster inference (important)
pip install unsloth accelerate bitsandbytes

# Flash Attention (highly recommended for speed)
pip install flash-attn --no-build-isolation

# For dataset handling and YAML parsing
pip install datasets pyyaml
```

## Environment Setup

This project requires several API keys for authentication. Create a `.env` file in the root directory with the following variables:

```
# API Keys for authentication
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
```

You can copy the provided `.env.example` file and fill in your credentials:

```bash
cp .env.example .env
# Edit the .env file with your actual API keys
```

These environment variables are required for:
- `HF_TOKEN`: Accessing Hugging Face models and datasets
- `WANDB_API_KEY`: Logging experiments to Weights & Biases
- `OPENAI_API_KEY`: Used if generating teacher completions with OpenAI models

## Key Classes

The project provides several key classes for working with the model:

### 1. QwenModelHandler
```python
class QwenModelHandler:
    """Handler for Qwen models with inference and saving capabilities using Unsloth"""

    def __init__(self, model_name="unsloth/Qwen2.5-7B", max_seq_length=768,
                 quantization=None, device_map="auto", cache_dir=None):
        """
        Initialize model and tokenizer using Unsloth

        Args:
            model_name: Name or path of the model (preferably an unsloth model)
            max_seq_length: Maximum sequence length for the model
            quantization: Quantization type (None, '4bit', '8bit') - for compatibility
            device_map: Device mapping strategy
            cache_dir: Cache directory for models
        """
```

This class handles the core model operations:
- Model loading and initialization
- Text generation with streaming support
- Perplexity calculation
- Model saving and pushing to HuggingFace Hub

### 2. PromptCreator
```python
class PromptCreator:
    """Creates and formats prompts for multiple choice questions"""

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = "teacher"  # Same YAML format but using teacher completions
```

This class manages prompt creation with three modes:
- Basic: Simple answer-only format
- YAML Reasoning: Structured reasoning in YAML format
- Teacher Reasoned: YAML format with teacher completions for training

### 3. ResponseParser
```python
class ResponseParser:
    """Parser for model responses with support for different formats"""

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"    # Parse YAML formatted response with reasoning
```

This class handles response parsing:
- Extracts answers from model responses
- Parses YAML-formatted reasoning
- Supports both basic and YAML formats

### 4. MultipleChoiceTester
```python
class MultipleChoiceTester:
    """Framework for testing Qwen models on multiple choice questions"""

    def __init__(self, model_handler, prompt_creator=None):
        """
        Initialize with model handler and prompt configuration

        Args:
            model_handler: The QwenModelHandler instance
            prompt_creator: Optional PromptCreator instance
        """
```

This class provides a complete testing framework:
- Single example inference
- Batch processing
- Dataset evaluation
- Performance metrics tracking
- Results saving and visualization

## Full Class Implementations

<details>
<summary>Click to expand/collapse full class implementations</summary>

### 1. QwenModelHandler
```python
class QwenModelHandler:
    """Handler for Qwen models with inference and saving capabilities using Unsloth"""

    def __init__(self, model_name="unsloth/Qwen2.5-7B", max_seq_length=768,
                 quantization=None, device_map="auto", cache_dir=None):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self.quantization = quantization
        self.cache_dir = cache_dir

        # Convert quantization parameter to load_in_4bit parameter for Unsloth
        self.load_in_4bit = quantization == "4bit"

        # Load tokenizer and model
        self.tokenizer, self.model = self._load_model()
        self.response_parser = ResponseParser()

    def _load_model(self):
        """Load model and tokenizer with Unsloth for optimization"""
        from unsloth import FastLanguageModel
        import torch

        print(f"Loading {self.model_name} with Unsloth, max_seq_length={self.max_seq_length}")

        # Set dtype based on hardware
        dtype = None  # None for auto detection

        # Load model and tokenizer with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit,
            cache_dir=self.cache_dir,
        )

        return tokenizer, model

    def generate_with_streaming(self, prompt, temperature=0.7, max_tokens=1024, stream=True):
        """Generate completion with optional streaming using Unsloth's optimized inference"""
        # Enable faster inference
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        # Generate with streaming if requested
        if stream:
            from transformers import TextIteratorStreamer
            import threading

            # Set up streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Start generation in a thread
            generation_kwargs = {
                "input_ids": model_inputs.input_ids,
                "attention_mask": model_inputs.attention_mask,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "streamer": streamer,
                "do_sample": temperature > 0.0,
                "use_cache": True,
                "min_p": 0.1 if temperature > 0.0 else None,
            }

            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return streamer
        else:
            # Generate without streaming
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.0,
                use_cache=True,
                min_p=0.1 if temperature > 0.0 else None,
            )

            # Decode the generated text
            generated_text = self.tokenizer.decode(
                generated_ids[0][model_inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return generated_text

    def calculate_perplexity(self, prompt, answer, temperature=0.0):
        """Calculate perplexity for a prompt and answer pair"""
        import torch

        # Format chat for perplexity calculation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        # Tokenize the text
        encodings = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings.input_ids)

        # Get loss and calculate perplexity
        neg_log_likelihood = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

        return perplexity

    def save_model(self, output_dir, save_method="lora"):
        """Save model to disk using Unsloth's optimized methods"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Use Unsloth's saving methods
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method == "merged_16bit":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_16bit")
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_4bit")
        elif save_method == "gguf":
            self.model.save_pretrained_gguf(output_dir, self.tokenizer, quantization_method="q4_k_m")
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        print(f"Model saved to {output_dir} using method {save_method}")
        return output_dir

    def push_to_hub(self, repo_id, token=None, save_method="lora", private=False):
        """Push model to Hugging Face Hub using Unsloth's optimized methods"""
        if save_method == "lora":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="lora", token=token)
        elif save_method == "merged_16bit":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="merged_16bit", token=token)
        elif save_method == "merged_4bit":
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="merged_4bit", token=token)
        elif save_method == "gguf":
            self.model.push_to_hub_gguf(
                repo_id,
                self.tokenizer,
                quantization_method=["q4_k_m", "q5_k_m"],
                token=token
            )
        else:
            raise ValueError(f"Unknown save method: {save_method}")

        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"
```

### 2. PromptCreator
```python
class PromptCreator:
    """Creates and formats prompts for multiple choice questions"""

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = "teacher"  # Same YAML format but using teacher completions

    def __init__(self, prompt_type=BASIC):
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING
        self.prompt_type = prompt_type
        self.original_type = prompt_type

    def format_choices(self, choices):
        """Format choices as a lettered list"""
        return "\n".join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )

    def get_max_letter(self, choices):
        """Get the maximum letter based on number of choices"""
        return chr(65 + len(choices) - 1)

    def create_inference_prompt(self, question, choices):
        """Create a prompt for inference based on current prompt type"""
        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        if self.prompt_type == self.YAML_REASONING:
            return self._create_yaml_prompt(question, formatted_choices, max_letter)
        else:
            return self._create_basic_prompt(question, formatted_choices, max_letter)

    def _create_basic_prompt(self, question, formatted_choices, max_letter):
        """Create a basic prompt asking for just the answer letter"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Answer with a single letter from A through {max_letter} without any additional explanation or commentary.
"""

    def _create_yaml_prompt(self, question, formatted_choices, max_letter):
        """Create a prompt requesting YAML-formatted reasoning"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Your response MUST be in YAML format as follows:

understanding: |
  <your understanding of what the question is asking>
analysis: |
  <your analysis of each option>
reasoning: |
  <your step-by-step reasoning process>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>

The answer field MUST contain ONLY a single character letter.
"""

    def create_training_prompt(self, question, choices):
        """Create a prompt for training with the current prompt type"""
        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        if self.prompt_type == self.YAML_REASONING:
            return self._create_yaml_training_prompt(
                question, formatted_choices, max_letter
            )
        else:
            return self._create_basic_training_prompt(
                question, formatted_choices, max_letter
            )

    def _create_basic_training_prompt(self, question, formatted_choices, max_letter):
        """Create a basic training prompt"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

The answer is a single letter (A, B, C, etc.). Only provide ONE character as your answer:
"""

    def _create_yaml_training_prompt(self, question, formatted_choices, max_letter):
        """Create a YAML-formatted training prompt"""
        return f"""
QUESTION:
{question}

CHOICES:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Follow the YAML format in your response:

understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>
"""

    def set_prompt_type(self, prompt_type):
        """Set the prompt type"""
        self.original_type = prompt_type
        if prompt_type == self.TEACHER_REASONED:
            pass
        self.prompt_type = prompt_type
        return self

    def is_teacher_mode(self):
        """Check if we're using teacher mode"""
        return self.original_type == self.TEACHER_REASONED
```

### 3. ResponseParser
```python
class ResponseParser:
    """Parser for model responses with support for different formats"""

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"    # Parse YAML formatted response with reasoning

    def __init__(self, parser_mode=BASIC):
        self.parser_mode = parser_mode

    def parse(self, response_text):
        """Parse the model's response according to the current mode"""
        if self.parser_mode == self.YAML:
            return self._parse_yaml_response(response_text)
        else:
            return self._parse_basic_response(response_text)

    def _parse_basic_response(self, response_text):
        """Parse basic response looking for a letter answer"""
        import re

        # Try to extract a single letter answer (A-Z)
        answer_match = re.search(r"(?:^|\s)([A-Z])(?:\s|$|\.)", response_text)
        if answer_match:
            answer = answer_match.group(1)
        else:
            # Take first character if it's a letter
            if response_text and response_text[0].isalpha():
                answer = response_text[0].upper()
            else:
                answer = None

        # For basic mode, we don't extract detailed reasoning
        reasoning = ""

        return answer, reasoning

    def _parse_yaml_response(self, response_text):
        """Parse YAML formatted response extracting answer and reasoning"""
        import re
        import yaml

        # First try to find answer in YAML format
        yaml_match = re.search(r"answer:\s*([A-Z])", response_text)
        if yaml_match:
            answer = yaml_match.group(1)
        else:
            # Fall back to basic extraction if YAML parsing fails
            answer_match = re.search(r"(?:^|\s)([A-Z])(?:\s|$|\.)", response_text)
            if answer_match:
                answer = answer_match.group(1)
            elif response_text and response_text[0].isalpha():
                answer = response_text[0].upper()
            else:
                answer = None

        # Try to parse reasoning from YAML format
        reasoning = ""
        if "reasoning:" in response_text:
            yaml_content = yaml.safe_load("---\n" + response_text)
            if isinstance(yaml_content, dict) and "reasoning" in yaml_content:
                reasoning = yaml_content["reasoning"]

                # Add other YAML fields if available
                if "understanding" in yaml_content:
                    reasoning = f"Understanding: {yaml_content['understanding']}\n\n{reasoning}"
                if "conclusion" in yaml_content:
                    reasoning = f"{reasoning}\n\nConclusion: {yaml_content['conclusion']}"
        else:
            # Use the full response as reasoning if not in YAML format
            reasoning = response_text

        return answer, reasoning

    def set_parser_mode(self, parser_mode):
        """Set the parser mode"""
        self.parser_mode = parser_mode
        return self

    @classmethod
    def from_prompt_type(cls, prompt_type):
        """Create a parser instance with mode matching the prompt type"""
        if prompt_type == PromptCreator.YAML_REASONING or prompt_type == PromptCreator.TEACHER_REASONED:
            return cls(parser_mode=cls.YAML)
        else:
            return cls(parser_mode=cls.BASIC)
```

### 4. MultipleChoiceTester
```python
class MultipleChoiceTester:
    """Framework for testing Qwen models on multiple choice questions"""

    def __init__(self, model_handler, prompt_creator=None):
        self.model_handler = model_handler
        self.prompt_creator = prompt_creator or PromptCreator(PromptCreator.BASIC)
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)

    def infer_example(self, example, temperature=0.7, max_tokens=1024, prompt_type=None, stream=False):
        """Inference on a single example for visualization/demonstration"""
        # Allow temporary override of prompt type
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.set_prompt_type(prompt_type)
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Prepare data
        question = example["question"]

        # Handle different formats of choices
        if isinstance(example["choices"], list):
            choices = example["choices"]
        elif isinstance(example["choices"], str) and example["choices"].startswith("["):
            import ast
            choices = ast.literal_eval(example["choices"]) if "[" in example["choices"] else example["choices"].split(",")
        else:
            choices = str(example["choices"]).split(",")

        # Generate the prompt using prompt creator
        prompt = self.prompt_creator.create_inference_prompt(question, choices)

        # Start timing
        start_time = time.time()

        if stream:
            # Use streaming generation
            streamer = self.model_handler.generate_with_streaming(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            # Collect output from streamer
            raw_response = ""
            print("Model response:")
            for text_chunk in streamer:
                print(text_chunk, end="", flush=True)
                raw_response += text_chunk
            print("\n")
        else:
            # Generate without streaming
            raw_response = self.model_handler.generate_with_streaming(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

        response_time = time.time() - start_time

        # Parse the response using the response parser
        predicted_answer, reasoning = self.response_parser.parse(raw_response)

        # Prepare results
        result = {
            "question": question,
            "choices": choices,
            "predicted_answer": predicted_answer,
            "reasoning": reasoning,
            "response_time": response_time,
            "raw_response": raw_response,
            "prompt_type": self.prompt_creator.prompt_type,
        }

        # Add task_id if available
        if "task_id" in example:
            result["task_id"] = example["task_id"]

        # Calculate metrics if label is provided
        if "answer" in example:
            label = example["answer"]
            result["correct_answer"] = label
            result["is_correct"] = predicted_answer == label

            # Calculate perplexity if requested
            if hasattr(self.model_handler, "calculate_perplexity"):
                perplexity = self.model_handler.calculate_perplexity(prompt, raw_response)
                result["perplexity"] = perplexity

        # Restore original prompt type if it was overridden
        if original_prompt_type is not None:
            self.prompt_creator.set_prompt_type(original_prompt_type)
            self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return result

    def infer_batch(self, examples, temperature=0.7, max_tokens=1024, prompt_type=None, batch_size=4):
        """Inference on a batch of examples"""
        # Allow temporary override of prompt type
        original_prompt_type = None
        if prompt_type is not None:
            original_prompt_type = self.prompt_creator.prompt_type
            self.prompt_creator.set_prompt_type(prompt_type)
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Prepare all prompts
        prompts = []
        metadata = []

        for i, example in enumerate(examples):
            # Extract data
            question = example["question"]

            # Handle different formats of choices
            if isinstance(example["choices"], list):
                choices = example["choices"]
            elif isinstance(example["choices"], str) and example["choices"].startswith("["):
                import ast
                choices = ast.literal_eval(example["choices"]) if "[" in example["choices"] else example["choices"].split(",")
            else:
                choices = str(example["choices"]).split(",")

            # Generate the prompt using prompt creator
            prompt = self.prompt_creator.create_inference_prompt(question, choices)
            prompts.append(prompt)

            # Store metadata for later
            meta = {
                "question": question,
                "choices": choices,
                "index": i,
            }

            # Add label if available
            if "answer" in example:
                meta["label"] = example["answer"]

            if "task_id" in example:
                meta["task_id"] = example["task_id"]

            metadata.append(meta)

        # Process in batches
        results = []
        correct_count = 0
        total_count = 0
        perplexities = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_meta = metadata[i:i+batch_size]

            # Process batch
            start_time = time.time()
            batch_responses = []

            for prompt in batch_prompts:
                response = self.model_handler.generate_with_streaming(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                batch_responses.append(response)

            batch_time = time.time() - start_time

            # Process each response in the batch
            for j, (response, meta) in enumerate(zip(batch_responses, batch_meta)):
                # Parse response
                predicted_answer, reasoning = self.response_parser.parse(response)

                # Create result
                result = {
                    "question": meta["question"],
                    "choices": meta["choices"],
                    "predicted_answer": predicted_answer,
                    "reasoning": reasoning,
                    "raw_response": response,
                    "prompt_type": self.prompt_creator.prompt_type,
                    "response_time": batch_time / len(batch_prompts),
                }

                # Add task_id if available
                if "task_id" in meta:
                    result["task_id"] = meta["task_id"]

                # Add metrics if label available
                if "label" in meta:
                    label = meta["label"]
                    result["correct_answer"] = label
                    result["is_correct"] = predicted_answer == label

                    # Update counts for accuracy
                    total_count += 1
                    if result["is_correct"]:
                        correct_count += 1

                    # Calculate perplexity if possible
                    if hasattr(self.model_handler, "calculate_perplexity"):
                        prompt = batch_prompts[j]
                        perplexity = self.model_handler.calculate_perplexity(prompt, response)
                        result["perplexity"] = perplexity
                        perplexities.append(perplexity)

                results.append(result)

        # Calculate aggregate metrics
        summary_metrics = {}
        if total_count > 0:
            summary_metrics["accuracy"] = correct_count / total_count
            summary_metrics["correct_count"] = correct_count
            summary_metrics["total_count"] = total_count

            if perplexities:
                summary_metrics["avg_perplexity"] = sum(perplexities) / len(perplexities)
                summary_metrics["min_perplexity"] = min(perplexities)
                summary_metrics["max_perplexity"] = max(perplexities)

        # Restore original prompt type if it was overridden
        if original_prompt_type is not None:
            self.prompt_creator.set_prompt_type(original_prompt_type)
            self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return results, summary_metrics

    def evaluate_dataset(self, dataset, temperature=0.7, max_tokens=1024, num_examples=None,
                        verbose=True, prompt_type=None, batch_size=4, log_to_wandb=False):
        """Inference on a whole dataset with metrics calculation"""
        # Allow overriding the prompt type for this evaluation
        original_prompt_type = self.prompt_creator.prompt_type
        if prompt_type is not None:
            self.prompt_creator.set_prompt_type(prompt_type)
            self.response_parser = ResponseParser.from_prompt_type(prompt_type)

        # Select subset if specified
        if num_examples is not None:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        results = []
        correct_count = 0
        total_count = 0
        perplexities = []

        # Process examples in batches
        for i in range(0, len(dataset), batch_size):
            batch_examples = dataset[i:i+batch_size]

            if verbose:
                batch_desc = f"Batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}"
                print(f"\nProcessing {batch_desc} with {len(batch_examples)} examples...")

            # Infer batch
            batch_results, batch_metrics = self.infer_batch(
                examples=batch_examples,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size
            )

            # Update metrics
            results.extend(batch_results)
            if "correct_count" in batch_metrics:
                correct_count += batch_metrics["correct_count"]
                total_count += batch_metrics["total_count"]

                if verbose:
                    batch_accuracy = batch_metrics["accuracy"]
                    overall_accuracy = correct_count / total_count
                    print(f"Batch accuracy: {batch_accuracy:.2%}, Overall: {overall_accuracy:.2%} ({correct_count}/{total_count})")

            # Collect perplexities
            if "avg_perplexity" in batch_metrics:
                for result in batch_results:
                    if "perplexity" in result:
                        perplexities.append(result["perplexity"])

        # Calculate final accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        if verbose:
            prompt_type_str = self.prompt_creator.prompt_type
            print(f"\nFinal accuracy with {prompt_type_str} prompts: {accuracy:.2%} ({correct_count}/{total_count})")
            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                print(f"Average perplexity: {avg_perplexity:.4f}")

        # Prepare comprehensive summary
        summary = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "prompt_type": self.prompt_creator.prompt_type,
            "results": results,
        }

        # Add perplexity metrics if available
        if perplexities:
            summary["avg_perplexity"] = sum(perplexities) / len(perplexities)
            summary["min_perplexity"] = min(perplexities)
            summary["max_perplexity"] = max(perplexities)

        # Log results to wandb if requested
        if log_to_wandb and wandb.run is not None:
            metrics = {
                "test/accuracy": accuracy,
                "test/correct_count": correct_count,
                "test/total_count": total_count,
            }
            if perplexities:
                metrics["test/avg_perplexity"] = summary["avg_perplexity"]
                metrics["test/min_perplexity"] = summary["min_perplexity"]
                metrics["test/max_perplexity"] = summary["max_perplexity"]

            wandb.log(metrics)

            # Create a table of results for visualization if task_id exists
            if "task_id" in dataset.features:
                columns = ["task_id", "question", "correct_answer", "predicted_answer", "is_correct"]
                table = wandb.Table(columns=columns)

                for res in results[:min(100, len(results))]:
                    table.add_data(
                        res.get("task_id", "unknown"),
                        res["question"][:100] + "...",
                        res.get("correct_answer", ""),
                        res.get("predicted_answer", ""),
                        res.get("is_correct", False)
                    )

                wandb.log({"test_samples": table})

        # Restore original prompt type
        self.prompt_creator.set_prompt_type(original_prompt_type)
        self.response_parser = ResponseParser.from_prompt_type(original_prompt_type)

        return summary

    def save_results(self, results, output_dir="./results"):
        """Save evaluation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.json")

        # Create serializable results
        serializable_results = {
            "accuracy": results.get("accuracy", 0.0),
            "correct_count": results.get("correct_count", 0),
            "total_count": results.get("total_count", 0),
            "timestamp": timestamp,
            "prompt_type": results.get("prompt_type", "unknown"),
        }

        # Add perplexity metrics if available
        if "avg_perplexity" in results:
            serializable_results["avg_perplexity"] = results["avg_perplexity"]
            serializable_results["min_perplexity"] = results["min_perplexity"]
            serializable_results["max_perplexity"] = results["max_perplexity"]

        # Process individual results
        serializable_results["individual_results"] = []
        for result in results["results"]:
            # Skip perplexity in individual results to save space
            result_copy = result.copy()
            if "perplexity" in result_copy:
                del result_copy["perplexity"]

            # Convert choices if needed
            choices = result_copy["choices"]
            if not isinstance(choices, list):
                try:
                    import ast
                    result_copy["choices"] = ast.literal_eval(choices)
                except (SyntaxError, ValueError):
                    pass

            serializable_results["individual_results"].append(result_copy)

        # Save to file
        with open(results_file, "w") as f:
            import json
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")
        return results_file
```

</details>

## Quick Start

Here's a simple example of how to use the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_id = "tuandunghcmut/Qwen25_Coder_MultipleChoice"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Example question
question = "What is the correct way to open a file in Python for reading?"
choices = [
    "open('file.txt', 'r')",
    "file.open('file.txt', 'read')",
    "read('file.txt')",
    "File.open('file.txt')"
]

# Format the prompt
prompt = f"""
QUESTION:
{question}

CHOICES:
{chr(65 + i)}. {choice}
for i, choice in enumerate(choices)}

Answer with a single letter from A through {chr(65 + len(choices) - 1)} without any additional explanation or commentary.
"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Model's answer: {response}")
```

## Advanced Usage

### Using the MultipleChoiceTester Framework

For more advanced usage, you can use the provided `MultipleChoiceTester` framework:

```python
from save import QwenModelHandler, MultipleChoiceTester, PromptCreator

# Initialize the model handler
model_handler = QwenModelHandler(
    model_name="tuandunghcmut/Qwen25_Coder_MultipleChoice",
    max_seq_length=2048,
    quantization="4bit",
    device_map="auto"
)

# Create a prompt creator with YAML reasoning format
prompt_creator = PromptCreator(PromptCreator.YAML_REASONING)

# Initialize the tester
tester = MultipleChoiceTester(model_handler, prompt_creator=prompt_creator)

# Example question
example = {
    "question": "What is the correct way to open a file in Python for reading?",
    "choices": [
        "open('file.txt', 'r')",
        "file.open('file.txt', 'read')",
        "read('file.txt')",
        "File.open('file.txt')"
    ],
    "answer": "A"  # Optional ground truth
}

# Get prediction with reasoning
result = tester.infer_example(example, temperature=0.0001, stream=True)
print(f"Predicted answer: {result['predicted_answer']}")
print("Reasoning:")
print(result['reasoning'])
```

### Batch Processing

You can also process multiple questions in batches:

```python
# List of examples
examples = [
    {
        "question": "What is the correct way to open a file in Python for reading?",
        "choices": ["open('file.txt', 'r')", "file.open('file.txt', 'read')", "read('file.txt')", "File.open('file.txt')"],
        "answer": "A"
    },
    # Add more examples...
]

# Process batch
results, metrics = tester.infer_batch(examples, batch_size=4)
print(f"Batch accuracy: {metrics['accuracy']:.2%}")
```

### Streaming Inference

The model supports streaming inference, which provides real-time output as the model generates its response. This is particularly useful for interactive applications and when you want to see the reasoning process in real-time.

#### Basic Streaming Usage

Here's how to use streaming inference:

```python
# Initialize model handler and tester as before
model_handler = QwenModelHandler(
    model_name="tuandunghcmut/Qwen25_Coder_MultipleChoice",
    max_seq_length=2048
)
tester = MultipleChoiceTester(model_handler)

# Example with streaming
example = {
    "question": "Which Python method is used to remove whitespace from both ends of a string?",
    "choices": [
        "strip()",
        "trim()",
        "clean()",
        "remove_whitespace()"
    ],
    "answer": "A"
}

# Enable streaming with stream=True
result = tester.infer_example(
    example,
    temperature=0.0001,
    max_tokens=1024,
    stream=True  # Enable streaming
)

# The output will be printed in real-time as the model generates it
# You can also access the complete response after generation
print("\nFinal result:")
print(f"Predicted answer: {result['predicted_answer']}")
print("Complete reasoning:")
print(result['reasoning'])
```

#### Advanced Streaming Patterns

##### 1. Custom Stream Processing

You can process the streamed output in custom ways:

```python
def process_stream(streamer):
    """Custom stream processing function"""
    collected_text = ""
    for chunk in streamer:
        # Process each chunk as it arrives
        collected_text += chunk
        # You can do custom processing here
        # For example, parse partial YAML, update UI, etc.
        yield chunk, collected_text

# Use custom stream processing
result = tester.infer_example(
    example,
    temperature=0.0001,
    stream=True
)

# Process the stream with custom logic
for chunk, full_text in process_stream(result['stream']):
    # Do something with each chunk
    print(f"Chunk: {chunk}")
    print(f"Full text so far: {full_text}")
```

##### 2. YAML Streaming with Real-time Parsing

When using YAML reasoning format, you can parse the output as it streams:

```python
import yaml
from io import StringIO

def parse_yaml_stream(streamer):
    """Parse YAML content as it streams"""
    buffer = StringIO()
    for chunk in streamer:
        buffer.write(chunk)
        try:
            # Try to parse the current buffer as YAML
            yaml_content = yaml.safe_load(buffer.getvalue())
            if yaml_content:
                yield chunk, yaml_content
        except yaml.YAMLError:
            # Not enough content for valid YAML yet
            continue

# Use YAML streaming with parsing
result = tester.infer_example(
    example,
    temperature=0.0001,
    prompt_type=PromptCreator.YAML_REASONING,
    stream=True
)

# Process YAML content as it streams
for chunk, yaml_content in parse_yaml_stream(result['stream']):
    if isinstance(yaml_content, dict):
        # Access YAML fields as they become available
        if 'understanding' in yaml_content:
            print(f"Understanding: {yaml_content['understanding']}")
        if 'reasoning' in yaml_content:
            print(f"Reasoning: {yaml_content['reasoning']}")
        if 'answer' in yaml_content:
            print(f"Answer: {yaml_content['answer']}")
```

##### 3. Streaming with Progress Tracking

You can track generation progress and timing:

```python
import time

def stream_with_progress(streamer):
    """Stream with progress tracking"""
    start_time = time.time()
    tokens_generated = 0

    for chunk in streamer:
        tokens_generated += len(chunk.split())
        elapsed = time.time() - start_time
        tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0

        yield {
            'chunk': chunk,
            'tokens': tokens_generated,
            'tokens_per_second': tokens_per_second,
            'elapsed': elapsed
        }

# Use streaming with progress tracking
result = tester.infer_example(
    example,
    temperature=0.0001,
    stream=True
)

for progress in stream_with_progress(result['stream']):
    print(f"Generated {progress['tokens']} tokens "
          f"({progress['tokens_per_second']:.2f} tokens/sec)")
    print(f"Chunk: {progress['chunk']}")
```

#### Implementation Details

The streaming implementation uses Unsloth's optimized inference with the following key features:

1. **Efficient Token Generation**
   - Uses Unsloth's `FastLanguageModel` for optimized inference
   - Implements streaming using `TextIteratorStreamer`
   - Supports both greedy and temperature-based sampling

2. **Memory Management**
   - Streams tokens without storing the entire response in memory
   - Efficiently handles long responses
   - Supports batch processing with streaming

3. **Performance Optimizations**
   - Uses `use_cache=True` for faster generation
   - Implements `min_p` sampling for better quality
   - Supports 4-bit quantization for reduced memory usage

4. **Error Handling**
   - Gracefully handles streaming interruptions
   - Provides partial results if generation is interrupted
   - Maintains context for resumed generation

The streaming output will show the model's reasoning process in real-time, including:
- Understanding of the question
- Analysis of each option
- Step-by-step reasoning
- Final conclusion
- Answer selection

This is particularly useful for:
- Debugging model behavior
- Creating interactive demos
- Understanding the model's reasoning process
- Providing immediate feedback to users
- Building real-time applications

## Model Features

- **YAML-Based Reasoning**: The model provides structured reasoning in YAML format
- **Multiple Prompt Types**: Supports both basic and YAML-formatted reasoning prompts
- **Batch Processing**: Efficiently process multiple questions at once
- **Performance Metrics**: Tracks accuracy, perplexity, and response times
- **Streaming Support**: Real-time output streaming for interactive use

## Examples

Check out the [example notebook](https://colab.research.google.com/drive/1YOUR_NOTEBOOK_ID) for more detailed usage examples and demonstrations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# Small-Qwen-Coding-Multiple-Choice
