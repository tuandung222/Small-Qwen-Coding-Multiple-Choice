#!/bin/bash
# train_with_callbacks.sh
# Script to train with all callbacks enabled (validation, early stopping, WandB)

# Display help information
function show_help {
    echo "Usage: ./train_with_callbacks.sh EXPERIMENT_NAME [OPTIONS]"
    echo ""
    echo "Run a training experiment with all callbacks enabled"
    echo ""
    echo "Positional arguments:"
    echo "  EXPERIMENT_NAME         Name for this experiment (required)"
    echo ""
    echo "Options:"
    echo "  -m, --model NAME        Source model name (default: unsloth/Qwen2.5-Coder-1.5B-Instruct)"
    echo "  -r, --repo NAME         Destination repo name (default: auto-generated)"
    echo "  -e, --epochs NUM        Number of training epochs (default: 5)"
    echo "  -b, --batch-size NUM    Batch size for training (default: 16)"
    echo "  -l, --lr RATE           Learning rate (default: 2e-4)"
    echo "  -g, --grad-accum NUM    Gradient accumulation steps (default: 2)"
    echo "  -d, --dataset NAME      Dataset to use (default: tuandunghcmut/coding-mcq-reasoning)"
    echo "  -s, --seq-len NUM       Maximum sequence length (default: 2048)"
    echo "  -p, --patience NUM      Early stopping patience (default: 3)"
    echo "  -v, --val-split NUM     Validation split ratio (default: 0.15)"
    echo "  -o, --output-root DIR   Root directory for experiment outputs (default: ./experiment_output)"
    echo "  -t, --test-mode         Enable test mode (uses only 2 dataset instances)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  ./train_with_callbacks.sh comprehensive_experiment --batch-size 16 --epochs 5 --patience 5"
    echo "  ./train_with_callbacks.sh quick_test --test-mode --epochs 2"
}

# Check if experiment name is provided
if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
    echo "Error: EXPERIMENT_NAME is required as the first argument"
    show_help
    exit 1
fi

# Set experiment name from first argument
EXP_NAME="$1"
shift

# Default values
MODEL="unsloth/Qwen2.5-Coder-1.5B-Instruct"
REPO=""
EPOCHS=5
BATCH_SIZE=16
LEARNING_RATE=2e-4
GRAD_ACCUM=2
DATASET="tuandunghcmut/coding-mcq-reasoning"
SEQ_LEN=2048
PATIENCE=3
VAL_SPLIT=0.15
OUTPUT_ROOT="./experiment_output"
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -r|--repo)
            REPO="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -g|--grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        -p|--patience)
            PATIENCE="$2"
            shift 2
            ;;
        -v|--val-split)
            VAL_SPLIT="$2"
            shift 2
            ;;
        -o|--output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -t|--test-mode)
            TEST_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Generate timestamp for unique experiment directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}_${TIMESTAMP}"

# Create output directory with parents if needed
mkdir -p "$OUTPUT_DIR"

# Create metadata file with experiment configuration
cat > "${OUTPUT_DIR}/experiment_config.txt" << EOL
Experiment: ${EXP_NAME}
Timestamp: ${TIMESTAMP}
Model: ${MODEL}
Repository: ${REPO:-auto-generated}
Epochs: ${EPOCHS}
Batch size: ${BATCH_SIZE}
Learning rate: ${LEARNING_RATE}
Gradient accumulation: ${GRAD_ACCUM}
Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))
Dataset: ${DATASET}
Sequence length: ${SEQ_LEN}
Early stopping patience: ${PATIENCE}
Validation split: ${VAL_SPLIT}
Test mode: ${TEST_MODE}
EOL

echo "=========================================================="
echo "Starting comprehensive training with all callbacks enabled"
echo "=========================================================="
echo "Experiment name:   $EXP_NAME"
echo "Model:             $MODEL"
echo "Repository:        ${REPO:-auto-generated}"
echo "Epochs:            $EPOCHS"
echo "Batch size:        $BATCH_SIZE"
echo "Learning rate:     $LEARNING_RATE"
echo "Gradient accum:    $GRAD_ACCUM"
echo "Effective batch:   $((BATCH_SIZE * GRAD_ACCUM))"
echo "Dataset:           $DATASET"
echo "Sequence length:   $SEQ_LEN"
echo "Early stopping:    Patience=$PATIENCE"
echo "Validation split:  $VAL_SPLIT (15%)"
echo "Output directory:  $OUTPUT_DIR"
if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE:         Enabled (using only 2 dataset instances)"
fi
echo "=========================================================="

# Create custom Python script for this run with custom callbacks
CUSTOM_SCRIPT="${OUTPUT_DIR}/run_with_callbacks.py"

cat > "$CUSTOM_SCRIPT" << PYTHON
#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.prompt_creator import PromptCreator
from model.qwen_handler import HubConfig, ModelSource, QwenModelHandler
from training.callbacks import EarlyStoppingCallback, ValidationCallback
from training.trainer import QwenTrainer
from utils.auth import setup_authentication
from utils.wandb_logger import WandBCallback, WandBConfig, WandBLogger

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("${OUTPUT_DIR}/training.log")
    ],
)
logger = logging.getLogger(__name__)

# Script configuration - hardcoded values from shell variables
DATASET = "${DATASET}"
MODEL_NAME = "${MODEL}"
REPO_NAME = "${REPO}"
OUTPUT_DIR = "${OUTPUT_DIR}"
EPOCHS = ${EPOCHS}
BATCH_SIZE = ${BATCH_SIZE}
LEARNING_RATE = ${LEARNING_RATE}
GRAD_ACCUM = ${GRAD_ACCUM}
SEQ_LEN = ${SEQ_LEN}
PATIENCE = ${PATIENCE}
VAL_SPLIT = ${VAL_SPLIT}
TEST_MODE = ${TEST_MODE}
EXP_NAME = "${EXP_NAME}"

def main():
    # Setup environment
    try:
        setup_authentication()
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set!")

        # Log GPU information
        if torch.cuda.is_available():
            logger.info("Using GPU: {}".format(torch.cuda.get_device_name(0)))
            logger.info("GPU Memory: {:.2f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
        else:
            logger.warning("CUDA is not available. Training will be slow!")
    except Exception as e:
        logger.error("Authentication setup failed: {}".format(str(e)))
        raise

    # Setup hub configurations
    model_id = MODEL_NAME
    repo_id = REPO_NAME

    source_hub = HubConfig(model_id=model_id, token=hf_token)

    # Handle destination repo
    if repo_id:
        destination_repo_id = repo_id
    else:
        # Get username from HF API
        api = HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            username = user_info.get("name", "user")
            model_name = model_id.split("/")[-1]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            destination_repo_id = "{}/{}_{}_{}".format(username, model_name, "finetuned", timestamp)
        except Exception as e:
            logger.warning("Could not get username from HF API: {}".format(str(e)))
            destination_repo_id = "user/qwen_finetuned_{}".format(time.strftime('%Y%m%d_%H%M%S'))

    # Check if the repository exists
    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=destination_repo_id, repo_type="model")
        logger.info("Repository {} already exists".format(destination_repo_id))
    except Exception as e:
        # If the repo doesn't exist, create it
        logger.info("Repository {} not found, creating it...".format(destination_repo_id))
        try:
            create_repo(
                repo_id=destination_repo_id,
                token=hf_token,
                private=True,
                repo_type="model",
            )
            logger.info("Repository {} created successfully".format(destination_repo_id))
            time.sleep(2)
        except Exception as create_error:
            logger.error("Failed to create repository: {}".format(str(create_error)))
            raise

    destination_hub = HubConfig(
        model_id=destination_repo_id,
        token=hf_token,
        private=True,
        save_method="lora",
    )

    logger.info("Source model: {}".format(source_hub.model_id))
    logger.info("Destination model: {}".format(destination_hub.model_id))

    # Initialize model and trainer
    try:
        # Initialize model handler
        logger.info("Initializing model handler...")
        model_handler = QwenModelHandler(
            model_name=source_hub.model_id,
            max_seq_length=SEQ_LEN,
            quantization="4bit",
            model_source=ModelSource.UNSLOTH,
            device_map="auto",
            source_hub_config=source_hub,
        )

        # Configure LoRA
        logger.info("Setting up LoRA configuration...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
                "gate_proj", "up_proj", "down_proj",     # FFN modules
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = QwenTrainer(
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            prompt_creator=PromptCreator(PromptCreator.YAML_REASONING),
            lora_config=lora_config,
            destination_hub_config=destination_hub,
            debug_samples=5,  # Log 5 samples per epoch for debugging
        )
    except Exception as e:
        logger.error("Error in setup: {}".format(str(e)))
        raise

    # Load dataset
    try:
        logger.info("Loading dataset {} from HuggingFace Hub...".format(DATASET))
        dataset = load_dataset(DATASET, token=hf_token, split="train")
        logger.info("Loaded {} training examples".format(len(dataset)))

        # Apply test mode if enabled
        if TEST_MODE:
            logger.info("TEST MODE ENABLED: Using only 2 dataset instances")
            dataset = dataset.select(range(2))
            logger.info("Dataset reduced to {} examples".format(len(dataset)))

        # Log dataset statistics
        logger.info("Dataset statistics:")
        logger.info("Features: {}".format(list(dataset.features.keys())))
        logger.info("Example:\n{}".format(dataset[0]))
    except Exception as e:
        logger.error("Error loading dataset: {}".format(str(e)))
        raise

    # Setup callbacks
    callbacks = []

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(patience=PATIENCE, min_delta=0.01)
    callbacks.append(early_stopping)
    logger.info("Added early stopping callback with patience={}".format(PATIENCE))

    # Validation callback
    validation_callback = ValidationCallback(trainer_instance=trainer)
    callbacks.append(validation_callback)
    logger.info("Added validation callback for model monitoring")

    # Setup WandB logging
    try:
        # Create WandB configuration
        model_name = MODEL_NAME.split("/")[-1]
        project_name = "{}-Coding-MCQ-Training".format(model_name)
        exp_name = EXP_NAME
        run_name = "{}_batch{}_lr{}_e{}_{}".format(
            exp_name, BATCH_SIZE, LEARNING_RATE, EPOCHS, int(time.time())
        )

        if TEST_MODE:
            run_name = "TEST_{}".format(run_name)

        tags = ["qwen", "coding", "lora", "multiple-choice", "callbacks"]
        if TEST_MODE:
            tags.append("test_mode")

        notes = "Comprehensive training with all callbacks enabled: EarlyStop, Validation, WandB"
        if TEST_MODE:
            notes = "TEST MODE: " + notes

        wandb_config = WandBConfig(
            project_name=project_name,
            run_name=run_name,
            tags=tags,
            notes=notes,
            log_memory=True,
            log_gradients=True,
        )

        # Initialize WandB logger
        wandb_logger = WandBLogger(config=wandb_config)
        wandb_logger.setup()

        # Add WandB callback
        wandb_callback = WandBCallback(logger=wandb_logger)
        callbacks.append(wandb_callback)

        logger.info("WandB logging enabled with project: {}, run: {}".format(project_name, run_name))
    except Exception as e:
        logger.warning("Failed to initialize WandB logging: {}".format(e))
        logger.warning("Continuing without WandB callbacks")

    # Training configuration
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Training outputs will be saved to: {}".format(OUTPUT_DIR))

    # Start training
    logger.info("Starting training with all callbacks enabled...")
    try:
        # In test mode, we use higher frequency logging and saving
        logging_steps = 10 if TEST_MODE else 50
        save_steps = 20 if TEST_MODE else 200

        results = trainer.train(
            train_dataset=dataset,
            val_split=VAL_SPLIT,
            output_dir=OUTPUT_DIR,
            # Training parameters
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            # Validation and checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Hub integration
            push_to_hub_strategy="best",
            # Other settings
            save_total_limit=3,
            random_seed=42,
            # Pass the callbacks
            callbacks=callbacks,
        )

        # Log results
        logger.info("Training completed!")
        if isinstance(results, dict):
            logger.info("Training metrics:")
            for key, value in results.items():
                logger.info("{}: {}".format(key, value))

        # Save metrics to file
        with open(os.path.join(OUTPUT_DIR, "training_metrics.txt"), "w") as f:
            f.write("Experiment: {}\n".format(EXP_NAME))
            f.write("Timestamp: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
            f.write("Test mode: {}\n\n".format(str(TEST_MODE).lower()))
            f.write("Training Metrics:\n")
            for key, value in results.items():
                f.write("{}: {}\n".format(key, value))

        # Create a success marker file
        with open(os.path.join(OUTPUT_DIR, "TRAINING_COMPLETE"), "w") as f:
            f.write("Training completed successfully at " + time.strftime("%Y-%m-%d %H:%M:%S"))

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        with open(os.path.join(OUTPUT_DIR, "TRAINING_INTERRUPTED"), "w") as f:
            f.write("Training was interrupted at " + time.strftime("%Y-%m-%d %H:%M:%S"))
        return 1
    except Exception as e:
        logger.error("Training failed: {}".format(str(e)))
        with open(os.path.join(OUTPUT_DIR, "TRAINING_FAILED"), "w") as f:
            f.write("Training failed at {}: {}".format(time.strftime('%Y-%m-%d %H:%M:%S'), str(e)))
        return 1
    finally:
        # Cleanup wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
PYTHON

# Make the script executable
chmod +x "$CUSTOM_SCRIPT"

# Run the custom script
echo "Running custom training script with all callbacks..."
echo "==========================================================="

python "${CUSTOM_SCRIPT}"

# Check if training was successful
RESULT=$?
echo "==========================================================="
if [ $RESULT -eq 0 ]; then
  echo "Experiment completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"

  # Create a symlink to the latest experiment
  LATEST_LINK="${OUTPUT_ROOT}/latest_callbacks"
  rm -f "$LATEST_LINK" 2>/dev/null
  ln -s "$OUTPUT_DIR" "$LATEST_LINK"
  echo "Created symlink: ${LATEST_LINK} -> ${OUTPUT_DIR}"
else
  echo "Experiment failed or was interrupted! Please check the logs for details."
  echo "Log file: ${OUTPUT_DIR}/training.log"
  exit $RESULT
fi
echo "==========================================================="
