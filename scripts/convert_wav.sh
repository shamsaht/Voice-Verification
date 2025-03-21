#!/bin/bash

# Set base directory as the parent of the scripts directory
base_dir="$(dirname "$(dirname "$0")")"

# Define the input and output directories relative to the base directory
input_dir="$base_dir/Dataset/speaker_dataset"
output_dir="$base_dir/srta_vauth/wav/test_set"

# Step 1: Run the Python script to split audio files into segments
echo "Running Python script to split audio files..."
python "$base_dir/Dataset/gen_utterance.py"  # Updated path for Python script

# Check if the Python script executed successfully
if [ $? -ne 0 ]; then
    echo "Python script failed. Exiting."
    exit 1
fi

# Step 2: Convert each .wav segment to 16-bit PCM with 16 kHz sample rate
echo "Converting segmented .wav files to 16-bit PCM with 16 kHz sample rate..."

# Loop through all .wav files in the output directory and its subdirectories
find "$output_dir" -type f -name "*.wav" | while read -r file; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Convert the file using ffmpeg
        ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 -ac 1 "${file%.wav}_temp.wav"
        
        # Replace original file with the converted file
        mv "${file%.wav}_temp.wav" "$file"
        
        echo "Converted $file to 16-bit PCM with 16 kHz sample rate."
    fi
done

# Step 3: Generate a list of .wav files in the srta_vauth directory
find "$base_dir/srta_vauth" -type f -name "*.wav" -print > "$base_dir/srta_vauth/testset.txt"
echo "All files have been processed and the file list has been saved to testset.txt."
