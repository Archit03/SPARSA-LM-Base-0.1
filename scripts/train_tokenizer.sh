#!/bin/bash

# Set environment variables
PYTHON_PATH=python
SCRIPT_PATH=tokenizer.py
LOG_FILE=tokenizer.log
VOCAB_SIZE=30000
DATASET_CONFIG=dataset_config.yaml
CACHE_DIR=.cache

# Check Python installation
if ! command -v $PYTHON_PATH &> /dev/null
then
    echo "Error: Python is not installed or not in PATH."
    exit 1
fi

# Check if the tokenizer script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Tokenizer script not found at $SCRIPT_PATH."
    exit 1
fi

# Display Python version
$PYTHON_PATH --version

# Run the tokenizer script
echo "Starting tokenizer training..."
$PYTHON_PATH $SCRIPT_PATH --vocab-size $VOCAB_SIZE --dataset-config $DATASET_CONFIG --cache-dir $CACHE_DIR > $LOG_FILE 2>&1

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error: Tokenizer training failed. Check $LOG_FILE for details."
    exit 1
else
    echo "Tokenizer training completed successfully."
fi
