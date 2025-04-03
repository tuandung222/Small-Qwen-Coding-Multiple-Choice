#!/usr/bin/env python
"""
Simple runner script for the Gradio app that defers model loading until needed
"""

import os
import sys
from pathlib import Path

import gradio as gr

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import the examples
from examples import CODING_EXAMPLES, CODING_EXAMPLES_BY_CATEGORY


# Create a simplified version of the app for testing the UI
def create_demo_interface():
    """Create a simplified demo interface that doesn't load the model right away"""

    def mock_inference(question, choices, temperature):
        """Mock inference function that tells the user the model would run here"""
        return """
        # This is a UI demo

        In the actual app, the model would process your question:

        ```
        Question: {question}

        Choices:
        {choices}

        Temperature: {temperature}
        ```

        And provide a YAML-formatted reasoning with understanding, analysis, reasoning, conclusion, and an answer.
        """.format(
            question=question, choices=choices, temperature=temperature
        )

    def process_example(example_idx):
        """Process an example from the preset list"""
        if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
            return "Invalid example index.", ""

        example = CODING_EXAMPLES[example_idx]
        question = example["question"]
        choices = "\n".join(example["choices"])

        return question, choices

    def get_category_examples(category_name):
        """Get examples for a specific category"""
        if category_name == "All Categories":
            return gr.Dropdown.update(
                choices=[
                    f"Example {i+1}: {ex['question']}" for i, ex in enumerate(CODING_EXAMPLES)
                ],
                value=None,
            )
        elif category_name in CODING_EXAMPLES_BY_CATEGORY:
            # Find the starting index for this category in the flattened list
            start_idx = 0
            for cat, examples in CODING_EXAMPLES_BY_CATEGORY.items():
                if cat == category_name:
                    break
                start_idx += len(examples)

            return gr.Dropdown.update(
                choices=[
                    f"Example {start_idx+i+1}: {ex['question']}"
                    for i, ex in enumerate(CODING_EXAMPLES_BY_CATEGORY[category_name])
                ],
                value=None,
            )
        else:
            return gr.Dropdown.update(choices=[], value=None)

    with gr.Blocks(title="Coding Multiple Choice Q&A with YAML Reasoning (Demo)") as interface:
        gr.Markdown("# Coding Multiple Choice Q&A with YAML Reasoning")
        gr.Markdown(
            """
        This demo showcases the interface for a fine-tuned Qwen2.5-Coder-1.5B model answering multiple-choice coding questions with structured YAML reasoning.

        **Note**: This is a UI demonstration only. The actual model is not loaded to save resources. Use `python app.py` to run the full app.

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

                # Submit button
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Model Response")
                output = gr.Markdown(label="Response")

        # Set up category selection
        category_dropdown.change(
            fn=get_category_examples, inputs=[category_dropdown], outputs=[example_dropdown]
        )

        # Set up example selection
        example_dropdown.change(
            fn=process_example, inputs=[example_dropdown], outputs=[question_input, choices_input]
        )

        # Set up submission
        submit_btn.click(
            fn=mock_inference,
            inputs=[question_input, choices_input, temperature_slider],
            outputs=[output],
        )

    return interface


if __name__ == "__main__":
    print("Starting demo interface (without loading the model)...")
    demo = create_demo_interface()
    demo.launch(share=True)
    print("Demo interface stopped.")
