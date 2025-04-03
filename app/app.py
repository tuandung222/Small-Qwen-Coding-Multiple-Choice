import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
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
                    quantization="4bit",  # Use 4-bit quantization to save memory
                )
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    def inference(self, question, choices_text, temperature=0.1):
        """Run inference on a single example"""
        # Parse choices from text area
        choices = [c.strip() for c in choices_text.split("\n") if c.strip()]

        # Create formatted prompt
        prompt = self.prompt_creator.create_inference_prompt(question, choices)

        # Generate response
        response_text = self.model_handler.generate_with_streaming(
            prompt=prompt, temperature=temperature, max_new_tokens=1024
        )

        # Parse response to extract answer and reasoning
        predicted_answer, reasoning = self.response_parser.parse(response_text)

        # Format the full response including the raw model output
        full_response = f"# Predicted Answer: {predicted_answer}\n\n"

        if reasoning:
            full_response += f"## Reasoning:\n{reasoning}\n\n"

        full_response += f"## Raw Model Output:\n```\n{response_text}\n```"

        return full_response

    def process_example(self, example_idx):
        """Process an example from the preset list"""
        print(f"Debug - example_idx type: {type(example_idx)}")
        print(f"Debug - example_idx value: {example_idx}")

        # Handle case where example_idx is a list
        if isinstance(example_idx, list):
            if not example_idx:
                return "No example selected.", ""
            # Take the first example if it's a list
            example_idx = example_idx[0]
            print(f"Debug - after list handling, example_idx: {example_idx}")

        # Convert string index to integer if needed
        if isinstance(example_idx, str):
            try:
                # Extract the number from "Example X: ..." format
                # Split by ':' and take the first part, then split by space and take the last part
                example_num = example_idx.split(":")[0].split()[-1]
                example_idx = int(example_num) - 1  # Convert to 0-based index
                print(f"Debug - after string conversion, example_idx: {example_idx}")
            except (ValueError, IndexError) as e:
                print(f"Debug - error during conversion: {e}")
                return "Invalid example index.", ""

        if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
            print(f"Debug - index out of range: {example_idx}")
            return "Invalid example index.", ""

        example = CODING_EXAMPLES[example_idx]
        question = example["question"]
        choices = "\n".join(example["choices"])

        return question, choices

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
            This demo showcases a fine-tuned Qwen2.5-Coder-1.5B model answering multiple-choice coding questions with structured YAML reasoning.

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
                        interactive=True,
                    )

                    # Example selector
                    example_dropdown = gr.Dropdown(
                        choices=[
                            f"Example {i+1}: {q['question']}" for i, q in enumerate(CODING_EXAMPLES)
                        ],
                        label="Select an example question",
                        value=None,
                        interactive=True,
                        show_label=True,
                        container=True,
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

                    # Submit button
                    submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column(scale=1):
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

            # Set up submission
            submit_btn.click(
                fn=self.inference,
                inputs=[question_input, choices_input, temperature_slider],
                outputs=[output],
            )

        return interface


def main():
    """Main function to run the app"""
    app = MCQGradioApp()
    interface = app.create_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    main()
