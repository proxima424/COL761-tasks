#!/bin/bash
# Main script logic
if [ "$1" = "train" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: bash interface.sh train </path/to/dataset> </path/to/output/model/in>"
        exit 1
    fi
    python3 train.py -d "$2" -m "$3"
elif [ "$1" = "test" ]; then
    if [ "$#" -ne 4 ]; then
        echo "Usage: bash interface.sh test <path/to/model> </path/to/test/dataset> </path/to/output/labels.txt>"
        exit 1
    fi
    python3 evaluate.py -d "$2" -m "$3" -o "$4"
else
    echo "Invalid command. Usage:"
    echo "bash interface.sh train </path/to/dataset> </path/to/output/model/in>"
    echo "bash interface.sh test <path/to/model> </path/to/test/dataset> </path/to/output/labels.txt>"
    exit 1
fi
