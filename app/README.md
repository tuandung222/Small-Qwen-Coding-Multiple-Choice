# Coding Multiple Choice Q&A Demo

This is a Gradio application that demonstrates the ability of a fine-tuned Qwen2.5-Coder-1.5B model to answer multiple-choice coding questions with structured YAML-based reasoning.

## Features

- Interactive web interface for question answering
- 20 predefined example coding problems
- Structured YAML reasoning format
- Temperature control for generation
- Detailed breakdown of the model's thinking process

## Getting Started

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Open the provided URL in your browser to access the demo.

## How It Works

The application uses a fine-tuned Qwen2.5-Coder-1.5B model specialized in solving coding multiple-choice questions. The model processes questions in a structured way:

1. **Understanding**: Analyzes what the question is asking
2. **Analysis**: Evaluates each option systematically
3. **Reasoning**: Works through the problem step-by-step
4. **Conclusion**: Provides a final answer with justification

## Example Questions

The demo includes 20 example questions covering various coding topics:
- SQL functions and commands
- Python syntax and behavior
- JavaScript features
- Data structures and algorithms
- Web development concepts
- And more...

## Model Details

This demo uses the `tuandunghcmut/Qwen25_Coder_MultipleChoice_v3` model from HuggingFace Hub, which is a fine-tuned version of `unsloth/Qwen2.5-Coder-1.5B-Instruct` specifically trained on multiple-choice coding questions with structured reasoning.

The model was trained using LoRA (Low-Rank Adaptation) with parameters optimized for coding problems.
