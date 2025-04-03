import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
import unsloth  # Import unsloth for optimized model loading
import yaml

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from examples import CODING_EXAMPLES, CODING_EXAMPLES_BY_CATEGORY

from src.model.qwen_handler import QwenModelHandler
from src.prompt_processors.prompt_creator import PromptCreator
from src.prompt_processors.response_parser import ResponseParser


class MCQGradioApp:
    """Gradio interface for the multiple choice question answering model"""

    def __init__(self, model_path="tuandunghcmut/Qwen25_Coder_MultipleChoice_v3"):
        """Initialize the application with model"""
        self.model_path = model_path
        self.model_handler = None
        self.prompt_creator = PromptCreator(prompt_type=PromptCreator.YAML_REASONING)
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)
        self.response_cache = {}  # Cache for model responses

        # Initialize the model (will be loaded on first use to save memory)
        self.load_model()

    def load_model(self):
        """Load the model from Hugging Face Hub or local checkpoint"""
        if self.model_handler is None:
            print(f"Loading model from {self.model_path}...")

            try:
                self.model_handler = QwenModelHandler(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    quantization=None,  # Disable quantization
                    device_map="auto",  # Automatically choose best device
                    attn_implementation="flash_attention_2",  # Use flash attention for better performance
                    force_attn_implementation=True,  # Force flash attention even if not optimal
                    model_source="unsloth",  # Use Unsloth's optimized model
                )
                # Set model to float16 after loading
                if self.model_handler.model is not None:
                    self.model_handler.model = self.model_handler.model.to(torch.float16)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    def inference(
        self,
        question,
        choices,
        temperature,
        max_new_tokens,
        top_p,
        top_k,
        repetition_penalty,
        do_sample,
    ):
        """Run inference with the model"""
        try:
            print("\n=== Debug: Inference Process ===")
            print(f"Input Question: {question}")
            print(f"Input Choices: {choices}")

            # Create cache key
            cache_key = f"{question}|{choices}|{temperature}|{max_new_tokens}|{top_p}|{top_k}|{repetition_penalty}|{do_sample}"
            print(f"Cache Key: {cache_key}")

            # Check cache first
            if cache_key in self.response_cache:
                print("Cache hit! Returning cached response")
                return self.response_cache[cache_key]

            # Create the prompt using the standard format from prompt_creator
            print("\nCreating prompt with PromptCreator...")
            prompt = self.prompt_creator.create_inference_prompt(question, choices)
            print(f"Generated Prompt:\n{prompt}")

            # Get model response using streaming generation
            print("\nStarting streaming generation...")
            response_chunks = []
            for chunk in self.model_handler.generate_with_streaming(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_p=0.1,  # Recommended value for better generation
                stream=True,
            ):
                if chunk:  # Only append non-empty chunks
                    response_chunks.append(chunk)
                    # Yield partial response for real-time display
                    partial_response = "".join(response_chunks)
                    yield prompt, f"""Question: {question}

Choices:
{choices}

{partial_response}"""

            # Combine all chunks for final response
            response = "".join(response_chunks)
            print(f"Complete Model Response:\n{response}")

            # Parse the response using the standard format
            print("\nParsing response...")
            try:
                parsed_response = self.response_parser.parse(response)
                print(f"Parsed Response: {parsed_response}")

                if not parsed_response or not isinstance(parsed_response, dict):
                    raise ValueError("Failed to parse response into expected format")

                if "answer" not in parsed_response or not parsed_response["answer"]:
                    raise ValueError("No answer found in parsed response")

                if "full_response" not in parsed_response:
                    parsed_response["full_response"] = response

            except Exception as parse_error:
                print(f"Error parsing response: {parse_error}")
                # Create a fallback response that follows the YAML format
                parsed_response = {
                    "answer": "Error: Could not parse model response",
                    "full_response": f"""understanding: |
  Error occurred while parsing the model response
analysis: |
  Unable to analyze the response due to parsing error
reasoning: |
  The model response could not be properly parsed into the expected YAML format
conclusion: |
  Please try again with different parameters
answer: X

Raw model output:
{response}""",
                }

            # Format the final response
            final_response = f"""Question: {question}

Choices:
{choices}

{parsed_response['full_response']}"""

            print("\nFinal Formatted Response:")
            print(final_response)

            result = (prompt, final_response)

            # Cache the result
            self.response_cache[cache_key] = result
            print("\nCached result for future use")

            # Yield final response
            yield result

        except Exception as e:
            print(f"\nError during inference: {e}")
            # Format error response in YAML format
            error_response = f"""Question: {question}

Choices:
{choices}

understanding: |
  An error occurred during processing
analysis: |
  The system encountered an error while processing the request
reasoning: |
  {str(e)}
conclusion: |
  Please try again or contact support if the error persists
answer: X

Raw model output:
{response if 'response' in locals() else 'No response available'}"""
            yield prompt, error_response

    def process_example(self, example_idx):
        """Process an example from the preset list"""
        if example_idx is None:
            return "", ""

        # Convert string index to integer if needed
        if isinstance(example_idx, str):
            try:
                # Extract the number from the string (e.g., "Example 13: ..." -> 13)
                example_idx = int(example_idx.split(":")[0].split()[-1]) - 1
            except (ValueError, IndexError) as e:
                print(f"Error converting example index: {e}")
                return "", ""

        try:
            if not isinstance(example_idx, int):
                print(f"Invalid example index type: {type(example_idx)}")
                return "", ""

            if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
                print(f"Example index out of range: {example_idx}")
                return "", ""

            example = CODING_EXAMPLES[example_idx]
            question = example["question"]
            choices = "\n".join(example["choices"])

            return question, choices

        except (ValueError, IndexError) as e:
            print(f"Error processing example: {e}")
            return "", ""

    def get_category_examples(self, category_name):
        """Get examples for a specific category"""
        if category_name == "All Categories":
            choices = [f"Example {i+1}: {ex['question']}" for i, ex in enumerate(CODING_EXAMPLES)]
        elif category_name in CODING_EXAMPLES_BY_CATEGORY:
            # Find the starting index for this category in the flattened list
            start_idx = 0
            for cat, examples in CODING_EXAMPLES_BY_CATEGORY.items():
                if cat == category_name:
                    break
                start_idx += len(examples)

            choices = [
                f"Example {start_idx+i+1}: {ex['question']}"
                for i, ex in enumerate(CODING_EXAMPLES_BY_CATEGORY[category_name])
            ]
        else:
            choices = []

        return gr.Dropdown(choices=choices, value=None, interactive=True)

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Coding Multiple Choice Q&A with YAML Reasoning") as interface:
            gr.Markdown("# Coding Multiple Choice Q&A with YAML Reasoning")
            gr.Markdown(
                """
            This app uses a fine-tuned Qwen2.5-Coder-1.5B model to answer multiple-choice coding questions with structured YAML reasoning.

            The model breaks down its thought process in a structured way, providing:
            - Understanding of the question
            - Analysis of all options
            - Detailed reasoning process
            - Clear conclusion
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Examples")

                    # Category selector
                    category_dropdown = gr.Dropdown(
                        choices=["All Categories"] + list(CODING_EXAMPLES_BY_CATEGORY.keys()),
                        value="All Categories",
                        label="Select a category",
                    )

                    # Example selector
                    example_dropdown = gr.Dropdown(
                        choices=[
                            f"Example {i+1}: {q['question']}" for i, q in enumerate(CODING_EXAMPLES)
                        ],
                        label="Select an example question",
                        value=None,
                    )

                    gr.Markdown("### Your Question")

                    # Question and choices inputs
                    question_input = gr.Textbox(
                        label="Question", lines=3, placeholder="Enter your coding question here..."
                    )
                    choices_input = gr.Textbox(
                        label="Choices (one per line)",
                        lines=4,
                        placeholder="Enter each choice on a new line, e.g.:\nOption A\nOption B\nOption C\nOption D",
                    )

                    # Parameters
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                        label="Temperature (higher = more creative, lower = more deterministic)",
                    )

                    # Additional generation parameters
                    max_new_tokens_slider = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        value=2048,
                        step=128,
                        label="Max New Tokens (maximum length of generated response)",
                    )

                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (nucleus sampling probability)",
                    )

                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k (number of highest probability tokens to consider)",
                    )

                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.1,
                        label="Repetition Penalty (higher = less repetition)",
                    )

                    do_sample_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Sampling (unchecked for greedy generation)",
                    )

                    # Submit button
                    submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Model Input")
                    prompt_display = gr.Textbox(
                        label="Prompt sent to model",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                    )

                    gr.Markdown("### Model Response")
                    output = gr.Markdown(label="Response")

            # Set up category selection
            category_dropdown.change(
                fn=self.get_category_examples,
                inputs=[category_dropdown],
                outputs=[example_dropdown],
            )

            # Set up example selection
            example_dropdown.change(
                fn=self.process_example,
                inputs=[example_dropdown],
                outputs=[question_input, choices_input],
            )

            # Update prompt display when question or choices change
            def update_prompt(question, choices):
                print("\n=== Debug: Prompt Update ===")
                print(f"Question Input: {question}")
                print(f"Choices Input: {choices}")

                if not question or not choices:
                    print("Empty question or choices, returning empty prompt")
                    return ""

                try:
                    print("\nCreating prompt with PromptCreator...")
                    prompt = self.prompt_creator.create_inference_prompt(question, choices)
                    print(f"Generated Prompt:\n{prompt}")
                    return prompt
                except Exception as e:
                    print(f"Error creating prompt: {e}")
                    return ""

            # Add prompt update on question/choices change
            question_input.change(
                fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
            )

            choices_input.change(
                fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
            )

            # Set up submission with loading indicator
            submit_btn.click(
                fn=self.inference,
                inputs=[
                    question_input,
                    choices_input,
                    temperature_slider,
                    max_new_tokens_slider,
                    top_p_slider,
                    top_k_slider,
                    repetition_penalty_slider,
                    do_sample_checkbox,
                ],
                outputs=[prompt_display, output],
                show_progress=True,  # Show progress bar
                queue=True,  # Enable queueing for better handling of multiple requests
            )

        return interface


def main():
    """Main function to run the app"""
    app = MCQGradioApp()
    interface = app.create_interface()
    # Enable queueing at the app level
    interface.queue()
    interface.launch(share=True)


if __name__ == "__main__":
    main()
