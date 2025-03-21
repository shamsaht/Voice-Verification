#!/bin/bash

# Define the root of your project and target directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR="$SCRIPT_DIR/../srta_vauth/wav/test_set"  # Access srta_vauth/wav/test_set from scripts directory

# Loop through each child directory in the target directory
for dir in "$PARENT_DIR"/*/; do
    # Initialize counters and variables
    count_3s=0
    count_4s=0
    count_5s=0
    total_duration=0
    total_files=0
    non_matching_files=()

    # Loop through each .wav file in the current child directory
    for file in "$dir"*.wav; do
        # Check if the file exists (to handle empty directories)
        if [ -f "$file" ]; then
            # Get the duration of the file in seconds
            duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
            rounded_duration=$(printf "%.0f" "$duration")  # Round to nearest second

            # Accumulate total duration and file count
            total_duration=$(echo "$total_duration + $duration" | bc)
            ((total_files++))

            # Increment counters or log as non-matching based on the rounded duration
            case $rounded_duration in
                3) ((count_3s++)) ;;
                4) ((count_4s++)) ;;
                5) ((count_5s++)) ;;
                *) non_matching_files+=("$file") ;;
            esac
        fi
    done

    # Check if the total files equal the sum of the 3, 4, and 5 second files
    counted_files=$((count_3s + count_4s + count_5s))
    match_status="does not match"
    if [ "$counted_files" -eq "$total_files" ]; then
        match_status="matches"
    fi

    # Print the results for the current directory
    echo "Directory: $(basename "$dir")"
    echo "3-second files: $count_3s"
    echo "4-second files: $count_4s"
    echo "5-second files: $count_5s"
    echo "Total number of files: $total_files"
    echo "Total audio length (in seconds): $total_duration"
    echo "Sum of 3, 4, and 5-second files: $counted_files ($match_status)"
    
    # Print non-matching files, if any
    if [ ${#non_matching_files[@]} -gt 0 ]; then
        echo "Files not 3, 4, or 5 seconds (after rounding):"
        for file in "${non_matching_files[@]}"; do
            echo "  - $(basename "$file")"
        done
    else
        echo "All files are 3, 4, or 5 seconds long (after rounding)."
    fi
    echo ""
done
