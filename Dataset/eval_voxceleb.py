import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from pathlib import Path
from dependencies import cdist
from core import load_model, process_audio

# Constants
PARAMS = {
    "preprocessing": {
        "audio_norm_target_dBFS": -30,
    },
}

embedding_cache = {}

def cache_embedding(file_path, model, device = "cpu"):
    try :
        if file_path not in embedding_cache:
            embedding = process_audio(model=model, audio_path=file_path)
            embedding_cache[file_path] = embedding
        return embedding_cache[file_path].reshape(1, -1)
    except Exception as e:
        print(f"Error reading {file_path} : {e}")
        return None 

# Function to process a single audio pair
def process_pair(model, file1, file2, test_dir, device="cpu"):
    try:
        # Retrieve cached embeddings
        features_1 = cache_embedding(test_dir / file1, model)
        features_2 = cache_embedding(test_dir / file2, model)
        # Calculate similarity
        similarity = 1 - cdist(features_1, features_2, metric="cosine").flatten()
        return similarity
    except Exception as e:
        print(f"Error processing pair {file1} and {file2}: {e}")
        return None
    #TODO : implement GPU support

# Main function
def main():
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    datasets_dir = base_dir / "Datasets/voxceleb"
    result_csv = datasets_dir / "result.csv"
    test_csv_path = datasets_dir / "test.csv"
    test_dir = datasets_dir / "vox1_test_wav/wav"

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model()
    model.to(device)
    model.eval()

    # Load test data
    test_csv = pd.read_csv(test_csv_path)
    predict = []

    # Process in parallel
    print("Processing audio pairs...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_row = {
            executor.submit(process_pair, model, row['audio_1'], row['audio_2'], test_dir, device): idx
            for idx, row in test_csv.iterrows()
        }

        for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            idx = future_to_row[future]
            try:
                similarity = future.result()
                predict.append((idx, similarity))
            except Exception as e:
                print(f"Error in processing row {idx}: {e}")
                predict.append((idx, None))

    # Assign predictions back to the dataframe
    for idx, similarity in predict:
        test_csv.loc[idx, 'output'] = similarity

    # Save results
    test_csv.to_csv(result_csv, index=False)
    print(f"Results saved to {result_csv}")

if __name__ == "__main__":
    main()
