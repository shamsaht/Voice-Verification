"""
retrieve_combination.py : Mike Banabila
This file will run the first experiment which
aims to generate the best embeddings for a given speaker.
You can select the starting phase of the program
Phase I : It will generate all combinations based on the 
available dataset and store how each performs in a pkl file
Phase II : Process the pkl files into CSV files for easier analysis
Phase III : Computes the best performing combination and stores the 
best embeddings in a pkl file.
"""
import os
import pickle
import numpy as np
from pathlib import Path

# Import necessary functions
from .combination_analysis import run_experiment, load_model
from .pickle_csv import process_pickle_files
from .analysis_embed import process_files_in_directory

# Define paths
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
COMBINATIONS_DIR = LOG_DIR / "combinations"
STATS_DIR = LOG_DIR / "stats"
CSV_DIR = LOG_DIR / "csv"
REFERENCE_EMBED_DIR = LOG_DIR.parent / "reference_embed"
REFERENCE_EMBED_DIR.mkdir(parents=True, exist_ok=True)

def string_to_array(string):
    # Remove the square brackets and convert to a NumPy array
    return np.fromstring(string.strip("[]"), sep=' ')

# Function to retrieve a specific combination for a given speaker and combination number
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

# Main execution
def main(speaker_id=None, combination_number=None, phase=0, verbose=0, exp_name="abs"):
    # If specific speaker and combination number are provided, retrieve the combination
    if speaker_id and combination_number:
        retrieve_embed(speaker_id, combination_number)
    else:
        # Phase control: Execute functions based on the phase argument
        if phase == 0:
            # Run all steps
            model = load_model()
            run_experiment(model, experiment_name=exp_name)
            process_pickle_files(STATS_DIR, CSV_DIR)
            results = process_files_in_directory(CSV_DIR)

        elif phase == 1:
            # Skip run_experiment
            model = load_model()
            process_pickle_files(STATS_DIR, CSV_DIR)
            results = process_files_in_directory(CSV_DIR)

        else:
            # Skip run_experiment and process_pickle_files
            results = process_files_in_directory(CSV_DIR)
        
        # Create dictionary for storing reference embeddings and file names
        reference_dict = {}
        
        for spkr_id, data in results.items():
            # Retrieve the combination files for each speaker from the combinations pickle file
            combination_files = retrieve_embed(spkr_id, data["Combination Number"])
            
            if combination_files is not None:
                # Store the combination files and reference embedding as a numpy array
                reference_embeds = string_to_array(data["Reference Embedding"])
                # print(type(reference_embeds))
                reference_dict[spkr_id] = {
                    "Combination Files": combination_files,
                    "Reference Embedding": reference_embeds
                }
                

        # Save the reference dictionary as a pickle file
        with open(REFERENCE_EMBED_DIR / "references.pkl", "wb") as f:
            pickle.dump(reference_dict, f)

        # Print reference_dict if verbose is set to non-zero
        if verbose:
            print("\nReference Dictionary Contents:")
            for spkr_id, info in reference_dict.items():
                print(f"Speaker {spkr_id}:")
                print(f"  Combination Files: {info['Combination Files']}")
                # Display first 5 elements for brevity
                print()

        print("Reference embeddings saved to references.pkl.")

# Run main function
if __name__ == "__main__":
    # Call with specific speaker, combination, phase, and verbosity level
    main(phase=0, verbose=1)  # Adjust as needed
