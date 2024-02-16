#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_path> <result_path>"
    exit 1
fi

# Assign input and result paths to variables
input_path=$1
result_path=$2

# Run the preprocessing and feature generation script
python3 "$(dirname "$0")/Q2/generate_features.py" -g "$input_path" -f "$result_path"

# Check if classify.py is present
if [ -e "$(dirname "$0")/Q2/classify.py" ]; then
    # Run the classification script
    python3 "$(dirname "$0")/Q2/classify.py" -g "$input_path" -f "$result_path"
else
    echo "Error: classify.py script not found."
    echo "Skipping classification step."
fi