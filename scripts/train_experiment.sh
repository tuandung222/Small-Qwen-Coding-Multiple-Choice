#!/bin/bash
# train_experiment.sh
# Flexible script for running training experiments with different configurations

# Display help information
function show_help {
    echo "Usage: ./train_experiment.sh EXPERIMENT_NAME [OPTIONS]"
    echo ""
    echo "Run a training experiment with custom configuration"
    echo ""
    echo "Positional arguments:"
    echo "  EXPERIMENT_NAME         Name for this experiment run (required)"
    echo ""
    echo "Options:"
    echo "  -m, --model NAME        Source model name (default: unsloth/Qwen2.5-Coder-1.5B-Instruct)"
    echo "  -r, --repo NAME         Destination repo name (default: auto-generated)"
    echo "  -e, --epochs NUM        Number of training epochs (default: 3)"
    echo "  -b, --batch-size NUM    Batch size for training (default: 8)"
    echo "  -l, --lr RATE           Learning rate (default: 2e-4)"
    echo "  -g, --grad-accum NUM    Gradient accumulation steps (default: 4)"
    echo "  -d, --dataset NAME      Dataset to use (default: tuandunghcmut/coding-mcq-reasoning)"
    echo "  -s, --seq-len NUM       Maximum sequence length (default: 2048)"
    echo "  -o, --output-root DIR   Root directory for experiment outputs (default: ./experiment_output)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  ./train_experiment.sh my_large_batch_test --batch-size 32 --epochs 5 --lr 1e-4"
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
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-4
GRAD_ACCUM=4
DATASET="tuandunghcmut/coding-mcq-reasoning"
SEQ_LEN=2048
OUTPUT_ROOT="./experiment_output"

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
        -o|--output-root)
            OUTPUT_ROOT="$2"
            shift 2
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
EOL

echo "=========================================================="
echo "Starting experiment: $EXP_NAME"
echo "=========================================================="
echo "Source model:      $MODEL"
echo "Destination repo:  ${REPO:-auto-generated}"
echo "Epochs:            $EPOCHS"
echo "Batch size:        $BATCH_SIZE"
echo "Learning rate:     $LEARNING_RATE"
echo "Gradient accum:    $GRAD_ACCUM"
echo "Effective batch:   $((BATCH_SIZE * GRAD_ACCUM))"
echo "Dataset:           $DATASET"
echo "Sequence length:   $SEQ_LEN"
echo "Output directory:  $OUTPUT_DIR"
echo "=========================================================="

# Build command with optional arguments
CMD="python src/run.py \
  --source-model \"$MODEL\" \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning-rate $LEARNING_RATE \
  --grad-accum $GRAD_ACCUM \
  --dataset \"$DATASET\" \
  --max-seq-length $SEQ_LEN \
  --output-dir \"$OUTPUT_DIR\""

# Add destination repo if specified
if [ -n "$REPO" ]; then
  CMD="$CMD --destination-repo \"$REPO\""
fi

# Display the command
echo "Executing command:"
echo "$CMD"
echo "=========================================================="

# Execute the command
eval $CMD

# Check if training was successful
RESULT=$?
echo "=========================================================="
if [ $RESULT -eq 0 ]; then
  echo "Experiment completed successfully!"
  echo "Results saved to: $OUTPUT_DIR"

  # Create a symlink to the latest experiment
  LATEST_LINK="${OUTPUT_ROOT}/latest"
  rm -f "$LATEST_LINK" 2>/dev/null
  ln -s "$OUTPUT_DIR" "$LATEST_LINK"
  echo "Created symlink: ${LATEST_LINK} -> ${OUTPUT_DIR}"
else
  echo "Experiment failed! Please check the logs for errors."
  exit $RESULT
fi
echo "=========================================================="
