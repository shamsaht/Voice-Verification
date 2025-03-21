# main.py
import warnings
# Warnings to be ignored 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from core.model_utils import load_model
from core.data_utils import load_test_files
from experiments.batch_experiment import run_batch_experiment
from experiments.live_experiment import run_live_experiment

def main():
    verification_model, asr_model = load_model()
    base_folder = "/Users/abdulrahmanbanabila/Documents/TII/codebases/ReDim"
    testset_file = "srta_vauth/testset.txt"
    test_files = load_test_files(testset_file, base_folder)
    is_live = input("Run live experiment? (y/n): ").strip().lower() == 'y'
    if is_live:
        speaker_id = input("Enter the speaker ID: ").strip()
        trascript = run_live_experiment(verification_model, asr_model, test_files, speaker_id)
        return trascript
    else :
        speaker = input("Enter speaker to evaluate: ")
        run_batch_experiment(verification_model, test_files, speaker)

if __name__ == "__main__":
    main()
