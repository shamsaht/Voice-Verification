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
from visualizations import silhouette_analysis, create_interactive_umap


embedding_cache = {}
ID_cache = {}

def cache_embedding(test_dir, file_path, model, device = "cpu"):
    def extract_speaker_id(file_name):
        return file_name.split('/')[0][-3:]
    try :
        if file_path not in embedding_cache:
            embedding = process_audio(model=model, audio_path=test_dir / file_path)
            embedding_cache[file_path] = embedding
            ID_cache[file_path] = extract_speaker_id(file_path)
        return embedding_cache[file_path].reshape(1, -1)
    except Exception as e:
        print(f"Error reading {file_path} : {e}")
        return None

def dic_2_arr() :
    global embedding_cache, ID_cache
    # Initialize empty lists for embeddings and labels
    embeddings = []
    labels = []

    # Iterate through filenames in the embedding dictionary
    for filename, embedding in embedding_cache.items():
        if filename in ID_cache:  # Ensure the file exists in both dictionaries
            embeddings.append(embedding)
            labels.append(ID_cache[filename])

    # Convert to numpy arrays if necessary
    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)
    embedding_cache.clear()
    ID_cache.clear()
    return embeddings_array, labels_array

# Function to process a single audio pair
def process_pair(model, file1, file2, test_dir, device="cpu"):
    try:
        # Retrieve cached embeddings
        cache_embedding(test_dir, file1, model)
        cache_embedding(test_dir, file2, model)
        # Calculate similarity
    except Exception as e:
        print(f"Error processing pair {file1} and {file2}: {e}")
        return None
    #TODO : implement GPU support

# Main function
def main():
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    datasets_dir = base_dir / "Datasets/voxceleb"
    test_csv_path = datasets_dir / "test.csv"
    test_dir = datasets_dir / "vox1_test_wav/wav"
    output_file = Path(__file__).resolve().parent.parent / "visualizations/voxsil.png"
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model()
    model.to(device)
    model.eval()

    # Load test data
    test_csv = pd.read_csv(test_csv_path)

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
                future.result()
            except Exception as e:
                print(f"Error in processing row {idx}: {e}")

    embeds, labels = dic_2_arr()
    silhouette_analysis(embeds, labels, "voxsil.png")
    labels = pd.DataFrame(labels, columns=['speaker_name'])
    embeds = pd.DataFrame(embeds)
    create_interactive_umap(embeds, labels, "voxceleb_umap")
    

if __name__ == "__main__":
    main()
