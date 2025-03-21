import pickle
import numpy as np
from pathlib import Path

# Define paths
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
COMBINATIONS_DIR = LOG_DIR / "combinations"
STATS_DIR = LOG_DIR / "stats"
CSV_DIR = LOG_DIR / "csv"
REFERENCE_EMBED_DIR = LOG_DIR.parent / "reference_embed"
REFERENCE_EMBED_DIR.mkdir(parents=True, exist_ok=True)

def load_speaker_map(file_path):
    """Load the speaker map from the speakermap.txt file."""
    speaker_map = {}
    with open(file_path, "r") as file:
        for line in file:
            # Parse the line to get the speaker ID and name
            speaker_id, speaker_name = line.strip().split(" - ")
            speaker_map[int(speaker_id)] = speaker_name
    return speaker_map

def retrieve_embed(speaker_id, combination_number):
    """Retrieve a specific combination for a given speaker and combination number."""
    combination_file = COMBINATIONS_DIR / f"{speaker_id}_combinations.pkl"
    
    # Load the combinations for the specified speaker
    with open(combination_file, 'rb') as f:
        combinations_data = pickle.load(f)

    # Retrieve the specified combination number (1-based index)
    if 0 < combination_number <= len(combinations_data):
        selected_combination = combinations_data[combination_number - 1]
        print(f"Speaker {speaker_id}, Combination {combination_number}: {selected_combination}")
        return selected_combination
    else:
        print("Combination not found.")
        return None

# Load reference embeddings from a pickle file
def load_references(file_path='reference_embed/references.pkl'):
    """
    The file contains a dictionary
    speaker_id : [Combination Files, Reference Embedding]
    """
    with open(file_path, 'rb') as f:
        references = pickle.load(f)
    return references

# Generate reference embedding for a given speaker ID
def retrieve_reference_embedding(speaker_id,references):
    # Retrieve the reference embedding for the given speaker ID
    if speaker_id in references:
        reference_embedding = references[speaker_id]["Reference Embedding"]
        return reference_embedding
    else :
        raise ValueError(f"Speaker ID {speaker_id} not found in references.")
    
# Generate reference embedding for a given speaker ID
def retrieve_reference_files(speaker_id,references):
    # Retrieve the reference embedding for the given speaker ID
    if speaker_id in references:
        reference_files = references[speaker_id]["Combination Files"]
        return reference_files
    else :
        raise ValueError(f"Speaker ID {speaker_id} not found in references.")
    
def load_speaker_files_from_txt(file_path):
    speakers_files = {}
    with open(file_path, "r") as f:
        for line in f:
            file_path = line.strip()
            speaker_name = file_path.split('/')[-2]  # Get the speaker name from the parent directory
            if speaker_name not in speakers_files:
                speakers_files[speaker_name] = []
            speakers_files[speaker_name].append(file_path)
    return speakers_files