# GPT-4o Generated Data Synthesis

This module provides a framework for generating synthetic explanations for multiple-choice questions using OpenAI models. It's designed to create high-quality, teacher-guided explanations that can be used for training or evaluation purposes.

## Features

- **Multiple Model Support**: Works with various OpenAI models (GPT-4o, GPT-4, GPT-3.5-turbo)
- **Configurable Generation**: Adjust temperature, max tokens, and other generation parameters
- **Concurrent Processing**: Process multiple examples simultaneously for faster generation
- **Comprehensive Metrics**: Track success rate, answer preservation, and response times
- **Flexible Output**: Save results in structured formats with detailed metadata
- **User-Friendly Interface**: Command-line arguments for easy configuration

## Usage

### Basic Usage

```bash
python gpt4o_generated.py --model gpt-4o --data-path /path/to/dataset --api-key YOUR_API_KEY
```

### Advanced Usage

```bash
python gpt4o_generated.py \
  --model gpt-4o \
  --data-path /path/to/dataset \
  --sample-size 100 \
  --temperature 0.2 \
  --max-tokens 2048 \
  --concurrent-requests 5 \
  --output-dir ./my_results \
  --system-prompt "Custom system prompt here" \
  --random-seed 42
```

### Command-Line Arguments

#### Model Configuration
- `--model`: OpenAI model to use (choices: gpt-4o, gpt-4, gpt-3.5-turbo, gpt-3.5-turbo-16k)
- `--temperature`: Temperature for model generation (higher = more creative)
- `--max-tokens`: Maximum tokens for model generation
- `--api-key`: OpenAI API key (if not provided, will use environment variable)
- `--system-prompt`: Custom system prompt for the model

#### Dataset Configuration
- `--data-path`: Path to the training dataset
- `--sample-size`: Number of examples to process (None for all)
- `--random-seed`: Random seed for reproducibility

#### Output Configuration
- `--output-dir`: Directory to save outputs

#### Processing Configuration
- `--concurrent-requests`: Number of concurrent API requests
- `--quiet`: Suppress verbose output

## Output Structure

The script generates the following outputs:

```
output_dir/
├── model_name_timestamp/           # Output directory with timestamp
│   ├── 000001_task_id.yaml        # Individual explanation files
│   ├── 000002_task_id.yaml
│   ├── ...
│   ├── synthesis_config.json       # Configuration used for generation
│   ├── synthesis_metrics.json      # Metrics from the generation process
│   └── synthesis_metrics_latest.json # Latest metrics during generation
```

## Example

```python
from gpt4o_generated import run_synthesis

# Run synthesis programmatically
output_dir, metrics = run_synthesis(
    model_name="gpt-4o",
    data_path="/path/to/dataset",
    sample_size=100,
    output_dir="./results",
    api_key="your-api-key",
    temperature=0.2,
    max_tokens=2048,
    concurrent_requests=5,
    random_seed=42,
    verbose=True
)

# Print metrics
print(f"Success rate: {metrics['successful_generations']/metrics['total_examples']:.2%}")
print(f"Answer preservation rate: {metrics['correct_answer_preserved']/metrics['total_examples']:.2%}")
```

## Requirements

- Python 3.8+
- OpenAI Python package
- Datasets library
- PyYAML
- Pandas
- Matplotlib
- Seaborn
- Tqdm

## License

This project is licensed under the MIT License - see the LICENSE file for details.
