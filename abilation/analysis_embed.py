import os
import numpy as np
import pandas as pd

def load_csv(file_path):
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def find_row_with_highest_min(distances):
    """Find the row with the highest minimum value in the distances array."""
    row_mins = np.min(distances, axis=1)
    max_min_index = np.argmax(row_mins)
    highest_min_row = distances[max_min_index]
    highest_min_value = row_mins[max_min_index]
    return max_min_index, highest_min_row, highest_min_value

def find_row_with_lowest_min(distances):
    """Find the row with the lowest minimum value in the distances array."""
    row_mins = np.max(distances, axis=1)
    min_min_index = np.argmin(row_mins)
    lowest_min_row = distances[min_min_index]
    lowest_min_value = row_mins[min_min_index]
    return min_min_index, lowest_min_row, lowest_min_value

def calculate_adjusted_differences(data, target_speaker_id):
    """Calculate differences between (mean - 2*std) of the target speaker and (mean + 2*std) of other speakers."""
    target_speaker_mean_col = f"Speaker {target_speaker_id} Mean"
    target_speaker_std_col = f"Speaker {target_speaker_id} Std"

    other_speaker_mean_cols = [
        col for col in data.columns if col.startswith("Speaker") and col.endswith("Mean") and f"{target_speaker_id} " not in col
    ]
    other_speaker_std_cols = [
        col for col in data.columns if col.startswith("Speaker") and col.endswith("Std") and f"{target_speaker_id} " not in col
    ]
    
    target_speaker_means = data[[target_speaker_mean_col]].values
    target_speaker_stds = data[[target_speaker_std_col]].values
    target_adjusted_values = target_speaker_means - 2 * target_speaker_stds

    other_speaker_means = data[other_speaker_mean_cols].values
    other_speaker_stds = data[other_speaker_std_cols].values
    other_speaker_adjusted = other_speaker_means + 2 * other_speaker_stds

    differences = target_adjusted_values - other_speaker_adjusted
    made_up = np.where(target_speaker_means > 0.65, 0.65, target_speaker_means)
    #print(f"{target_speaker_means.mean}:{made_up.mean}")
    criteria = (target_speaker_means - other_speaker_means) / (target_speaker_stds + other_speaker_stds)
    criteria = np.where(criteria < 0, 400, criteria)
    return criteria, differences


def load_speaker_map(file_path):
    """Load the speaker map from the speakermap.txt file."""
    speaker_map = {}
    with open(file_path, "r") as file:
        for line in file:
            # Parse the line to get the speaker ID and name
            speaker_id, speaker_name = line.strip().split(" - ")
            speaker_map[int(speaker_id)] = speaker_name
    return speaker_map


def process_files_in_directory(directory_path, verbose=0):
    """Process each .csv file in the directory, finding rows with highest min distance, and storing specific data."""
    parent_dir = os.path.dirname(directory_path)
    speaker_map_file = os.path.join(parent_dir, "speakermap.txt")
    speaker_map = load_speaker_map(speaker_map_file)
    column_names_printed = False
    speaker_data = {}  # Dictionary to store results for each speaker

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            speaker_id = int(filename.split('.')[0])

            if verbose == 0:
                print(f"\nProcessing Speaker {speaker_id} from file: {filename}")
            
            data = load_csv(file_path)
            criteria, distances = calculate_adjusted_differences(data, speaker_id)

            # Print column names once if verbose is enabled
            if verbose == 0 and not column_names_printed:
                print("Column names:", data.columns.to_numpy())
                column_names_printed = True

            # Find row with highest minimum value
            max_min_index, max_min_row, n_value = find_row_with_highest_min(criteria)
            min_min_index, min_min_row, min_min_value = find_row_with_lowest_min(criteria)
            if verbose == 0 :
                # 6 & 58 & 0.73±0.04 & 0.10±0.06 & 4 & 0.42
                print("Speaker & combination & speaker mu + std & worst match & W mu + std & distance")
                # row where we found the max n
                max_data = data.iloc[max_min_index].to_numpy()
                # our speaker
                speaker_idx = 3 + 4 * (speaker_id - 1)
                # the guy in the row with the smallest n
                worst_match = np.argmin(max_min_row) + 1
                worst_match = worst_match if worst_match < speaker_id else worst_match + 1
                worst_idx = 3 + 4 * (worst_match - 1)
                print(f"{speaker_map[speaker_id]} & {max_min_index} & {max_data[speaker_idx]:.2f} \pm {max_data[speaker_idx + 1]:.2f} & {speaker_map[worst_match]} & {max_data[worst_idx]:.2f} \pm {max_data[worst_idx + 1]:.2f} & {n_value:.2f} \\\\")
                #print("Original data row at max_min_index:")
                #print(data.iloc[max_min_index].to_numpy())
                min_data = data.iloc[min_min_index].to_numpy()
                speaker_idx = 3 + 4 * (speaker_id - 1)
                worst_match = np.argmin(min_min_row) + 1
                worst_match = worst_match if worst_match < speaker_id else worst_match + 1
                worst_idx = 3 + 4 * (worst_match - 1) 
                print(f"{speaker_map[speaker_id]} & {min_min_index} & {min_data[speaker_idx]:.2f} \pm {min_data[speaker_idx + 1]:.2f} & {speaker_map[worst_match]} & {min_data[worst_idx]:.2f} \pm {min_data[worst_idx + 1]:.2f} & {min_min_value:.2f} \\\\")
                print(max_min_row)
                #print("Original data row at min_min_index:")
                #print(data.iloc[min_min_index].to_numpy())
            #Reference Embedding
            # Store data in the dictionary
            speaker_data[speaker_id] = {
                "Combination Number": data.iloc[max_min_index]["Combination Number"],
                "Reference Embedding": data.iloc[max_min_index]["Reference Embedding"]
            }

    return speaker_data

def main():
    directory_path = 'logs/csv'  # Update path as needed
    process_files_in_directory(directory_path)

if __name__ == "__main__":
    main()
