#!/bin/bash

# Set environment variables
PYTHON_PATH="python"
SCRIPT_PATH="src/tokenizer.py"
LOG_DIR="$HOME/Desktop/SPARSA-LM-Base 0.1/logs"
VOCAB_SIZE=32000
MIN_FREQ=2
DATASET_CONFIG="$HOME/Desktop/SPARSA-LM-Base 0.1/config/dataset_config.yaml"
CACHE_DIR="$HOME/Desktop/SPARSA-LM-Base 0.1/data/processed/cache"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check Python installation
if ! command -v $PYTHON_PATH &> /dev/null; then
    echo "Error: Python not found. Please ensure Python is installed and in PATH."
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Tokenizer script not found at $SCRIPT_PATH."
    exit 1
fi

# Check CUDA availability
echo "Checking hardware configuration..."
$PYTHON_PATH -c "import torch; print('Using ' + ('CUDA' if torch.cuda.is_available() else 'CPU') + ' for tokenizer training')"
if [ $? -ne 0 ]; then
    echo "Error: Unable to determine hardware configuration. Ensure PyTorch is installed correctly."
    exit 1
fi
echo

# Run the tokenizer script with specified parameters
echo "Starting tokenizer training with the following configuration:"
echo "- Vocabulary Size: $VOCAB_SIZE"
echo "- Minimum Frequency: $MIN_FREQ"
echo "- Dataset Config: $DATASET_CONFIG"
echo "- Cache Directory: $CACHE_DIR"
echo "- Log Directory: $LOG_DIR"
echo

# Updated command with proper argument flags and quoted paths
$PYTHON_PATH "$SCRIPT_PATH" \
    --vocab-size "$VOCAB_SIZE" \
    --min-frequency "$MIN_FREQ" \
    --dataset-config "$DATASET_CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --log-dir "$LOG_DIR" > "$LOG_DIR/tokenizer_output.log" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: Tokenizer training failed. Check $LOG_DIR/tokenizer.log for details."
    exit 1
else
    echo "Tokenizer training completed successfully. Logs available in $LOG_DIR"
fi
