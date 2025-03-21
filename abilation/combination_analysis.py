import os
import torch
import numpy as np
import pickle
import time
import torchaudio
import threading
from itertools import count
import concurrent.futures
from itertools import combinations
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from pathlib import Path

# Caching embeddings
embedding_cache = {}
# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent

# Update LOG_DIR and other paths relative to BASE_DIR
LOG_DIR = BASE_DIR / "logs"
SPEAKER_MAP_FILE = LOG_DIR / "speakermap.txt"
COMBINATIONS_DIR = LOG_DIR / "combinations"
STATS_DIR = LOG_DIR / "stats"
COMBINATIONS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

def cache_embedding(file, model):
    if file not in embedding_cache:
        embedding_cache[file] = process_audio(model, file)
    return embedding_cache[file]

def load_model():
    model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
    model.eval()
    return model

# Generate speaker ID mapping
def generate_speaker_map(speakers):
    speaker_map = {i + 1: speaker for i, speaker in enumerate(speakers)}
    with open(SPEAKER_MAP_FILE, "w") as f:
        for idx, name in speaker_map.items():
            f.write(f"{idx} - {name}\n")
    return speaker_map

# Save combination and stats data
def save_data(filepath, data):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

# Generate combinations of 3 files for a speaker and store
def generate_combinations(speaker_id, files):
    comb_filepath = COMBINATIONS_DIR / f"{speaker_id}_combinations.pkl"
    combinations_list = list(combinations(files, 3))
    save_data(comb_filepath, combinations_list)
    return combinations_list


def calculate_stats_for_combinations(model, speaker_id, speaker_combinations, test_files, speaker_map):
    stat_filepath = STATS_DIR / f"{speaker_id}_stats.pkl"
    all_stats = []
    total_combinations = len(speaker_combinations)

    # Cache test file embeddings for faster access
    test_embeddings = {file: cache_embedding(file, model) for file in test_files}

    # Initialize a thread-safe counter for processed combinations
    combination_counter = count(1)
    start_time = time.time()
    lock = threading.Lock()
    def log_progress(processed_count):
        """Logs progress and estimates remaining time."""
        elapsed = time.time() - start_time
        total_processed = processed_count
        time_per_comb = elapsed / total_processed
        remaining_combinations = total_combinations - total_processed
        estimated_time_remaining = time_per_comb * remaining_combinations
        print(
            f"Speaker {speaker_id}: Processed {total_processed}/{total_combinations} combinations. "
            f"Elapsed: {elapsed:.2f}s, Estimated Remaining: {estimated_time_remaining:.2f}s"
        )
    def process_combination(comb_index, comb_files):
        # Get reference embeddings and compute mean and std
        reference_embeddings = [cache_embedding(file, model) for file in comb_files]
        reference_embedding = np.mean(reference_embeddings, axis=0)
        reference_std = np.std(reference_embeddings, axis=0)

        # Gather embeddings for remaining test files (exclude current combination files)
        remaining_test_files = [file for file in test_files if file not in comb_files]
        remaining_test_embeddings = np.array([test_embeddings[file] for file in remaining_test_files])

        # Calculate cosine similarities using cdist in one operation
        reference_embedding_reshaped = reference_embedding.reshape(1, -1)
        cosine_distances = cdist(reference_embedding_reshaped, remaining_test_embeddings, metric="cosine").flatten()
        similarities = 1 - cosine_distances  # Convert cosine distances to similarities

        # Collect statistics for each speaker in test set
        similarity_stats = {}
        for other_speaker_id, other_speaker_name in speaker_map.items():
            # Filter similarities for files belonging to the current test speaker
            speaker_similarities = [
                similarities[i] for i, file in enumerate(remaining_test_files) if other_speaker_name in file
            ]
            if speaker_similarities:
                similarity_stats[other_speaker_id] = {
                    "mean": np.mean(speaker_similarities),
                    "std": np.std(speaker_similarities),
                    "min": np.min(speaker_similarities),
                    "max": np.max(speaker_similarities),
                }
            else:
                similarity_stats[other_speaker_id] = {"mean": 0, "std": 0, "min": 0, "max": 0}

        # Store statistics for the current combination
        result = {
            "combination_number": comb_index,
            "reference_embedding": reference_embedding,
            "reference_std": reference_std,
            "similarity_stats": similarity_stats,
        }
    
        # Update the progress counter and print progress every 10 combinations
        with lock:
            current_count = next(combination_counter)
            if current_count % 5000 == 0:
                log_progress(current_count)

        return result

    # Process combinations in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_combination, comb_index, comb_files)
            for comb_index, comb_files in enumerate(speaker_combinations, 1)
        ]
        for future in concurrent.futures.as_completed(futures):
            all_stats.append(future.result())

    # Save results for the speaker
    save_data(stat_filepath, all_stats)

# Process audio file and generate embedding
def process_audio(model, file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    with torch.no_grad():
        embedding = model(waveform).squeeze().cpu().numpy().flatten()
    return embedding

# Main function to run the experiment

def load_speaker_files_from_txt(file_path, level = -2):
    speakers_files = {}
    with open(file_path, "r") as f:
        for line in f:
            file_path = line.strip()
            speaker_name = file_path.split('/')[level]  # Get the speaker name from the parent directory
            if speaker_name not in speakers_files :
                speakers_files[speaker_name] = []
            speakers_files[speaker_name].append(file_path)
    return speakers_files

def run_experiment(model, selected_speaker=None, experiment_name="abs"):
    # Path to testset.txt, relative to the current script
    test_file = "srta_vauth/testset.txt"
    level = -2
    if experiment_name == "vox" :
        test_file = "srta_vauth/voxset.txt"
        level = -3
    testset_file_path = Path(__file__).resolve().parent.parent / test_file
    
    # Load speaker files from testset.txt
    speakers_files = load_speaker_files_from_txt(testset_file_path, level=level)
    
    # Create a speaker map
    speaker_map = generate_speaker_map(list(speakers_files.keys()))

    # Check if a specific speaker is selected
    if selected_speaker:
        if selected_speaker in speaker_map.values():
            # Get the ID for the selected speaker
            speaker_id = next(sid for sid, name in speaker_map.items() if name == selected_speaker)
            print(f"Processing selected speaker {speaker_id}: {selected_speaker}")

            # Run experiment for the selected speaker only
            files = speakers_files[selected_speaker]
            test_files = [f for name, f_list in speakers_files.items() for f in f_list]
            speaker_combinations = generate_combinations(speaker_id, files)
            calculate_stats_for_combinations(model, speaker_id, speaker_combinations, test_files, speaker_map)
            print(f"Completed processing for Speaker {speaker_id}: {selected_speaker}")
        else:
            print(f"Speaker '{selected_speaker}' not found in the dataset.")
    else:
        # Run experiment for each speaker
        for speaker_id, speaker_name in speaker_map.items():
            print(f"Processing Speaker {speaker_id}: {speaker_name}")
            files = speakers_files[speaker_name]
            test_files = [f for name, f_list in speakers_files.items() for f in f_list]
            speaker_combinations = generate_combinations(speaker_id, files)
            calculate_stats_for_combinations(model, speaker_id, speaker_combinations, test_files, speaker_map)
            print(f"Completed processing for Speaker {speaker_id}: {speaker_name}")

if __name__ == "__main__":
    model = load_model()  # Assumes load_model function is defined
    start_time = time.time()
    # Specify the selected speaker name if needed, or set to None for all speakers
    selected_speaker = "Selina"  # Replace with None to run for all speakers
    run_experiment(model)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Program runtime: {elapsed_time:.2f} seconds")

