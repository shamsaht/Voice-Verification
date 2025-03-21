import torch
import torchaudio
import numpy as np
import os
import random
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import sounddevice as sd
from pynput import keyboard
import pickle
sd.default.device = 0
torchaudio.set_audio_backend("soundfile")
# Load the ReDimNet model
def load_model():
    model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
    model.eval()
    return model

# Process a single audio file into a spectrogram and then to an embedding
def process_audio(model, audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary (e.g., to 16kHz)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
    # Generate embedding
    with torch.no_grad():
        embedding = model(waveform)  # Assuming model takes waveform input directly
    
    # Flatten to ensure a 1-D array and confirm shape
    embedding = embedding.squeeze().cpu().numpy().flatten()
    return embedding

# Load reference embeddings from a pickle file
def load_references(file_path='references.pkl'):
    with open(file_path, 'rb') as f:
        references = pickle.load(f)
    return references

# Generate reference embedding for a given speaker ID
def generate_reference_embedding(speaker_id, references_file_path='reference_embed/references.pkl'):
    # Load references from the pickle file
    references = load_references(references_file_path)
    
    # Retrieve the reference embedding for the given speaker ID
    if speaker_id in references:
        reference_embedding = references[speaker_id]["Reference Embedding"]
        return reference_embedding
    else :
        raise ValueError(f"Speaker ID {speaker_id} not found in references.")

def evaluate(model, reference_embedding, test_files, speaker_id):
    similarities = []
    labels = []
    
    for file in test_files:
        test_embedding = process_audio(model, file)
        similarity = 1 - cosine(reference_embedding, test_embedding)
        similarities.append(similarity)
        labels.append(1 if speaker_id in file else 0)  # Label as 1 if it belongs to speaker, else 0
    
    return np.array(similarities), np.array(labels)

def load_speaker_map(file_path):
    """Load the speaker map from the speakermap.txt file."""
    speaker_map = {}
    with open(file_path, "r") as file:
        for line in file:
            # Parse the line to get the speaker ID and name
            speaker_id, speaker_name = line.strip().split(" - ")
            speaker_map[int(speaker_id)] = speaker_name
    return speaker_map

def load_test_files(file_path, base_dir):
    """Load all test files from testset.txt."""
    with open(file_path, 'r') as f:
        files = [os.path.join(base_dir, line.strip()) for line in f]
    return files

def sample_enrollment_files(test_files, speaker, num_samples=3):
    """Randomly sample files for enrollment for the specified speaker without replacement and update test files."""
    # Filter files by speaker
    speaker_files = [f for f in test_files if speaker in os.path.basename(f)]
    
    # Sample 3-5 files for enrollment
    enrollment_files = random.sample(speaker_files, num_samples)
    
    # Remove the selected enrollment files from test_files
    test_files = [f for f in test_files if f not in enrollment_files]
    
    return enrollment_files, test_files

def experiment_for_each_speaker(model, testset_file, base_dir, log_file="logs.txt"):
    test_files = load_test_files(testset_file, base_dir)
    speakers = set(os.path.basename(os.path.dirname(file)) for file in test_files)
    parent_dir = os.path.dirname('logs/csv')
    speaker_map_file = os.path.join(parent_dir, "speakermap.txt")
    speaker_map = load_speaker_map(speaker_map_file)
    results = {}

    with open(log_file, "w") as log:
        for speaker in speakers:
            print(f"\nRunning experiment for speaker: {speaker}")
            log.write(f"Speaker: {speaker}\n")
            speaker_id = next((k for k, v in speaker_map.items() if v == speaker), None)
            # Sample enrollment files for the current speaker
            
            remaining_files = [f for f in test_files]
            
            # Log the enrollment files
            log.write("Enrollment files:\n")
            
            
            # Generate reference embedding
            reference_embedding = generate_reference_embedding(speaker_id)
            
            # Evaluate on remaining test files
            similarities, labels = evaluate(model, reference_embedding, remaining_files, speaker)

# Example usage
if __name__ == "__main__":
    model = load_model()
    base_folder = "/Users/abdulrahmanbanabila/Documents/TII/codebases/ReDim"
    testset_file = os.path.join(base_folder, "testset.txt")
    # Run experiment for each speaker and collect results
    experiment_for_each_speaker(model, testset_file, base_folder)