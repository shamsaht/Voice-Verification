from core import all_metrics
import numpy as np
import pandas as pd
from pathlib import Path
from dependencies import plt

def main():
    # File paths
    current_dir = Path(__file__).parent.resolve()
    csv_file = current_dir.parent.parent.parent / "Datasets/voxceleb/result.csv"
    output_dir = current_dir.parent.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_file)
    similarities = df.iloc[:, -1].values.astype(float)  # Second to last column
    labels = df.iloc[:, -2].values.astype(int)  # Last column

    # Compute metrics
    stats = all_metrics(similarities, labels)
    
    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[htp]
\\centering
\\begin{{tabular}}{{|c|c|c|}}
\\hline
Metric & Value & Threshold (More, Less, Wst) \\\\
\\hline
EER & {stats["eer"]:.4f} & ({stats["eer_threshold_more"]:.4f}, {stats["eer_threshold_less"]:.4f}, {stats["eer_threshold_wst"]:.4f}) \\\\
MinDCF & {stats["min_dcf"]:.4f} & ({stats["min_dcf_threshold_more"]:.4f}, {stats["min_dcf_threshold_less"]:.4f}, {stats["min_dcf_threshold_wst"]:.4f}) \\\\
\\hline
\\end{{tabular}}
\\caption{{Evaluation Metrics}}
\\label{{tab:metrics}}
\\end{{table}}
"""
    print("LaTeX Table:\n")
    print(latex_table)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(stats["fpr"], stats["tpr"], label=f"ROC Curve (AUC = {np.trapz(stats['tpr'], stats['fpr']):.4f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save the plot
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    print(f"ROC curve saved to {roc_path}")

if __name__ == "__main__":
    main()