import os
import pickle
import pandas as pd

def load_pickle(file_path):
    """Loads data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def transform_data(data):
    """Transforms the data into a tabular format for analysis."""
    table_data = []
    for entry in data:
        row = {
            "Combination Number": entry.get("combination_number"),
            "Reference Embedding": entry.get("reference_embedding", []),
            "Reference Std": entry.get("reference_std", []),
        }

        # Extract similarity statistics for each speaker
        similarity_stats = entry.get("similarity_stats", {})
        for speaker_id, stats in similarity_stats.items():
            row[f"Speaker {speaker_id} Mean"] = stats.get("mean")
            row[f"Speaker {speaker_id} Std"] = stats.get("std")
            row[f"Speaker {speaker_id} Min"] = stats.get("min")
            row[f"Speaker {speaker_id} Max"] = stats.get("max")

        table_data.append(row)
    
    return pd.DataFrame(table_data)

def save_to_csv(df, output_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def process_pickle_files(input_dir, output_dir):
    """Processes all pickle files in the input directory and saves each as a CSV in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pkl'):
            speaker_id = file_name.split('_')[0]  # Extract speaker ID from filename (e.g., '6_stats.pkl' -> '6')
            file_path = os.path.join(input_dir, file_name)
            output_csv_path = os.path.join(output_dir, f"{speaker_id}.csv")

            # Load, transform, and save data
            data = load_pickle(file_path)
            df = transform_data(data)
            save_to_csv(df, output_csv_path)

def main():
    input_dir = 'logs/stats'   # Directory containing .pkl files
    output_dir = 'logs/csv'    # Directory to store generated .csv files
    process_pickle_files(input_dir, output_dir)

if __name__ == "__main__":
    main()