#!/bin/bash

# Define the root directory of your project and output paths
PROJECT_ROOT=$(dirname "$(dirname "$0")")  # Root of the project (one level up from "scripts")
OUTPUT_DOT_FILE="$PROJECT_ROOT/visualizations/structure.dot"
OUTPUT_PNG_FILE="$PROJECT_ROOT/visualizations/structure.png"

# Create visualizations directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/visualizations"

# Start the .dot file with graph definitions
echo "digraph G {" > "$OUTPUT_DOT_FILE"
echo "    node [shape=box];" >> "$OUTPUT_DOT_FILE"

# Generate tree structure, then process each line, filtering out .wav files and files in nested subdirectories
tree -fi "$PROJECT_ROOT" | while read -r line; do
    # Skip lines with .wav files
    if [[ "$line" == *.wav ]]; then
        continue
    fi

    # Only include paths that are directly within the project root directory
    RELATIVE_PATH="${line#$PROJECT_ROOT/}"
    if [[ "$RELATIVE_PATH" == */* ]]; then
        continue
    fi

    # Define node and parent
    NODE="\"$RELATIVE_PATH\""
    PARENT=$(dirname "$RELATIVE_PATH")
    
    # Skip linking root to itself
    if [ "$PARENT" != "$RELATIVE_PATH" ]; then
        echo "    \"$PARENT\" -> $NODE;" >> "$OUTPUT_DOT_FILE"
    fi
done

# End the .dot file
echo "}" >> "$OUTPUT_DOT_FILE"

# Generate the PNG image from the .dot file
dot -Tpng "$OUTPUT_DOT_FILE" -o "$OUTPUT_PNG_FILE"

# Display a completion message
echo "DOT and PNG files have been generated in the 'visualizations' folder:"
echo "  - DOT file: $OUTPUT_DOT_FILE"
echo "  - PNG file: $OUTPUT_PNG_FILE"
