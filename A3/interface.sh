#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 <question_number> [sub_part] [dataset_path]"
    echo "  <question_number>: The number of the question to run"
    echo "  [sub_part]: Sub-part of Q2 (if applicable)"
    echo "  [dataset_path]: Path to dataset for Q2 (if applicable)"
    exit 1
}

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
    usage
fi

# Determine which question to run
question_number=$1

case $question_number in
    1)
        # Run code for question 1
       
        python3 "Q1.py"
        ;;
    2)
        # Check if sub-part and dataset path are provided for question 2
        if [ "$#" -lt 3 ]; then
            usage
        fi

        sub_part=$2
        dataset_path=$3

        # Run code for question 2 with provided sub-part and dataset path
        python3 "$(dirname "$0")/Q2.py" -s "$sub_part" -p "$dataset_path"
        ;;
    *)
        echo "Invalid question number. Please provide a valid question number."
        usage
        ;;
esac