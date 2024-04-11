#!/bin/bash

# Usage: ./script_name.sh /path/to/main/directory

# Check if argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DIRECTORY"
    exit 1
fi

# Check if the provided argument is a directory
if [ ! -d "$1" ]; then
    echo "$1 is not a directory."
    exit 1
fi

# Run weight_conversion.sh on each subdirectory starting with 'checkpoint'
for subdir in "$1"/checkpoint*0000/; do
    if [ -d "$subdir" ]; then
        ./trainer/scripts/weight_conversion.sh "$subdir"
    fi
done
