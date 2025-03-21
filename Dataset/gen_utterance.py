import os
import random
import torchaudio

# Set up base directory relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "speaker_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "../srta_vauth/wav/test_set")

# Constants for audio processing
MIN_DURATION_MS = 3000  # 3 seconds in milliseconds
MAX_DURATION_MS = 5000  # 5 seconds in milliseconds
TARGET_SAMPLE_RATE = 16000

def resample_if_needed(waveform, sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    """Resamples the waveform to the target sample rate if necessary."""
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform

def save_audio_segment(segment, sample_rate, output_path):
    """Saves a single audio segment to the specified output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, segment, sample_rate)

def split_audio(file_path, output_dir, min_duration_ms=MIN_DURATION_MS, max_duration_ms=MAX_DURATION_MS):
    """Splits an audio file into random segments between min_duration_ms and max_duration_ms."""
    speaker_id = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(output_dir, speaker_id)
    
    # Check if the directory for this speaker already exists
    if os.path.exists(save_dir):
        print(f"Skipping {file_path}: segments already exist in {save_dir}")
        return
    
    # Load and resample audio if necessary
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = resample_if_needed(waveform, sample_rate)
    audio_length = waveform.shape[1]

    os.makedirs(save_dir, exist_ok=True)
    start = 0
    segment_count = 0

    while start < audio_length:
        segment_duration = random.randint(min_duration_ms, max_duration_ms) * TARGET_SAMPLE_RATE // 1000
        end = min(start + segment_duration, audio_length)
        segment = waveform[:, start:end]
        
        segment_file = os.path.join(save_dir, f"{speaker_id}_seg{segment_count}.wav")
        save_audio_segment(segment, TARGET_SAMPLE_RATE, segment_file)
        
        print(f"Created segment: {segment_file} ({start / TARGET_SAMPLE_RATE:.2f} to {end / TARGET_SAMPLE_RATE:.2f} seconds)")

        start += segment_duration
        segment_count += 1

def process_directory(input_dir, output_dir):
    """Processes all .wav files in the input directory and splits them into segments."""
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(input_dir, file_name)
            split_audio(file_path, output_dir)

if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
