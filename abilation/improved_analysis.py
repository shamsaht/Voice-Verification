import os
import torch
import numpy as np
import pickle
import threading
import time
import torchaudio
from itertools import count
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from scipy.spatial.distance import cdist
from pathlib import Path
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
SPEAKER_MAP_FILE = LOG_DIR / "speakermap.txt"
COMBINATIONS_DIR = LOG_DIR / "combinations"
STATS_DIR = LOG_DIR / "stats"
COMBINATIONS_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Caching
embedding_cache = {}
cache_lock = threading.Lock()


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

def cache_embedding(file, model, device="cpu", max_cache_size=5000):
    """Caches embeddings with memory management."""
    global embedding_cache

    with cache_lock:
        # Manage cache size
        if len(embedding_cache) >= max_cache_size:
            embedding_cache.clear()  # Clear cache if max size exceeded
        
        # Retrieve or compute embedding
        if file not in embedding_cache:
            embedding = process_audio(file, model, device)
            embedding_cache[file] = embedding
        return embedding_cache[file]

def process_audio(file_path, model, device="cpu"):
    """Processes audio and generates an embedding."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    with torch.no_grad():
        embedding = model(waveform.to(device)).squeeze().cpu().numpy()
    return embedding

def precompute_pairwise_similarities(test_files, model, device="cpu"):
    """Precomputes pairwise cosine similarities for all test files."""
    embeddings = np.array([cache_embedding(file, model, device) for file in test_files])
    pairwise_similarities = 1 - cdist(embeddings, embeddings, metric="cosine")
    return pairwise_similarities, embeddings

def load_model():
    """Loads the pretrained model."""
    model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
    model.eval()
    return model

def generate_speaker_map(speakers):
    """Generates a mapping of speaker IDs to names."""
    speaker_map = {i + 1: speaker for i, speaker in enumerate(speakers)}
    with open(SPEAKER_MAP_FILE, "w") as f:
        for idx, name in speaker_map.items():
            f.write(f"{idx} - {name}\n")
    return speaker_map

def generate_combinations(speaker_id, files):
    """Generates all combinations of files for a speaker."""
    comb_filepath = COMBINATIONS_DIR / f"{speaker_id}_combinations.pkl"
    combinations_list = list(combinations(files, 3))
    save_data(comb_filepath, combinations_list)
    return combinations_list

def save_data(filepath, data):
    """Saves data to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def calculate_stats_for_combinations_optimized(
    model, speaker_id, speaker_combinations, test_files, speaker_map, pairwise_similarities
):
    """Calculates statistics using precomputed pairwise similarities."""
    stat_filepath = STATS_DIR / f"{speaker_id}_stats.pkl"
    all_stats = []
    total_combinations = len(speaker_combinations)

    # Cache test embeddings
    test_file_indices = {file: idx for idx, file in enumerate(test_files)}

    def process_combination(comb_index, comb_files):
        """Processes a single combination."""
        reference_indices = [test_file_indices[file] for file in comb_files]
        reference_embeddings = pairwise_similarities[reference_indices, :]
        reference_mean = np.mean(reference_embeddings, axis=0)

        remaining_indices = [
            idx for idx, file in enumerate(test_files) if file not in comb_files
        ]
        remaining_similarities = pairwise_similarities[reference_indices][:, remaining_indices]

        similarity_stats = {}
        for other_speaker_id, other_speaker_name in speaker_map.items():
            speaker_similarities = [
                remaining_similarities[i] for i, file in enumerate(test_files) if other_speaker_name in file
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

        return {
            "combination_number": comb_index,
            "reference_embedding": reference_mean,
            "similarity_stats": similarity_stats,
        }

    # Process combinations in parallel
    combination_counter = count(1)
    start_time = time.time()

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

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_combination, comb_index, comb_files): comb_index
            for comb_index, comb_files in enumerate(speaker_combinations, 1)
        }
        for i, future in enumerate(as_completed(futures)):
            all_stats.append(future.result())
            processed_count = next(combination_counter)
            if processed_count % 1000 == 0:
                log_progress(processed_count)

    # Save results
    save_data(stat_filepath, all_stats)

def run_experiment_optimized(model, test_files, speakers_files, speaker_map):
    """Runs the experiment using optimized methods."""
    print("Precomputing pairwise similarities...")
    pairwise_similarities, embeddings = precompute_pairwise_similarities(test_files, model)

    for speaker_id, speaker_name in speaker_map.items():
        print(f"Processing Speaker {speaker_id}: {speaker_name}")
        files = speakers_files[speaker_name]
        speaker_combinations = generate_combinations(speaker_id, files)
        calculate_stats_for_combinations_optimized(
            model, speaker_id, speaker_combinations, test_files, speaker_map, pairwise_similarities
        )
        print(f"Completed processing for Speaker {speaker_id}: {speaker_name}")

def main():
    """Main function to run the optimized experiment."""
    model = load_model()
    testset_file_path = BASE_DIR / "srta_vauth/voxset.txt"
    speakers_files = load_speaker_files_from_txt(testset_file_path, level=-3)
    test_files = [f for files in speakers_files.values() for f in files]
    speaker_map = generate_speaker_map(list(speakers_files.keys()))

    run_experiment_optimized(model, test_files, speakers_files, speaker_map)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
