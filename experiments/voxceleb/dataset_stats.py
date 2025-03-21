import numpy as np
import pandas as pd
from pathlib import Path

def generate_latex(test_csv):
    # Create a function to extract speaker ID from the file path
    def extract_speaker_id(file_path):
        return file_path.split('/')[0]

    # Extract speaker IDs for both columns
    test_csv['audio_1_speaker'] = test_csv['audio_1'].apply(extract_speaker_id)
    test_csv['audio_2_speaker'] = test_csv['audio_2'].apply(extract_speaker_id)

    # Count unique files for each speaker in audio_1 and audio_2
    audio_1_stats = test_csv.groupby('audio_1_speaker')['audio_1'].nunique().reset_index()
    audio_1_stats.columns = ['speaker_id', 'unique_files_audio_1']

    audio_2_stats = test_csv.groupby('audio_2_speaker')['audio_2'].nunique().reset_index()
    audio_2_stats.columns = ['speaker_id', 'unique_files_audio_2']

    # Merge the stats into a single DataFrame
    combined_stats = pd.merge(
        audio_1_stats,
        audio_2_stats,
        on='speaker_id',
        how='outer'
    ).fillna(0)  # Fill NaN with 0 for speakers not present in both columns

    # Sort by speaker ID for consistency
    combined_stats = combined_stats.sort_values('speaker_id')

    # Prepare the LaTeX table rows
    latex_rows = []
    for idx, row in enumerate(combined_stats.itertuples(), 1):
        latex_rows.append(f"{idx} & {row.speaker_id} & {int(row.unique_files_audio_1)} & {int(row.unique_files_audio_2)} \\\\ \\hline")

    # Format the LaTeX table
    latex_table = (
        "\\begin{table}[htp]\n"
        "\\centering\n"
        "\\begin{tabular}{|c|c|c|c|}\n"
        "\\hline\n"
        "Number & Speaker ID & Unique Files in Column 1 & Unique Files in Column 2 \\\\ \\hline\n"
    )
    latex_table += "\n".join(latex_rows)
    latex_table += (
        "\n\\end{tabular}\n"
        "\\caption{Speaker statistics for unique files in both columns.}\n"
        "\\label{tab:combined_speaker_stats}\n"
        "\\end{table}"
    )

    return latex_table

# Main function
def main():
    # Paths
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    datasets_dir = base_dir / "Datasets/voxceleb"
    test_csv_path = datasets_dir / "test.csv"

    # Load test data
    test_csv = pd.read_csv(test_csv_path)
    print(generate_latex(test_csv))

if __name__ == "__main__":
    main()
