import pandas as pd
from pathlib import Path

def load_test_file(test_path):
    """
    Load the test CSV file into a DataFrame.

    Parameters:
        test_path (Path): Path to the test.csv file.

    Returns:
        DataFrame: Loaded DataFrame from the test.csv file.
    """
    return pd.read_csv(test_path)

def extract_unique_file_paths(test_df, parent_dir):
    """
    Extract unique filenames from the 'audio_1' and 'audio_2' columns
    and create their absolute paths.

    Parameters:
        test_df (DataFrame): DataFrame containing the test data.
        parent_dir (Path): Parent directory of the test.csv file.

    Returns:
        List[str]: List of unique absolute file paths.
    """
    unique_files = set(test_df['audio_1']).union(set(test_df['audio_2']))
    return [str(parent_dir / file) for file in unique_files]

def save_to_file(file_paths, output_file):
    """
    Save the list of file paths to a text file.

    Parameters:
        file_paths (List[str]): List of file paths to save.
        output_file (Path): Path to the output text file.
    """
    with open(output_file, "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")
    print(f"Unique file paths saved to {output_file}")

def main():
    # Define the path to the test.csv file
    current_dir = Path(__file__).resolve().parent
    test_path = current_dir.parent.parent / "Datasets/voxceleb/test.csv"  # Replace with the actual path to your test.csv

    # Resolve the parent directory of the test.csv file
    parent_dir = test_path.parent.resolve()
    test_dir = parent_dir / "vox1_test_wav/wav"
    # Load the test.csv file
    test_df = load_test_file(test_path)

    # Extract unique file paths
    file_paths = extract_unique_file_paths(test_df, test_dir)

    # Define the output file path
    output_file = current_dir.parent / "srta_vauth/voxset.txt"

    # Save the unique file paths to a text file
    save_to_file(file_paths, output_file)

if __name__ == "__main__":
    main()
