# core/data_utils.py

from dependencies import os, random

def load_test_files(file_path, base_dir):
    """
    Load all test files from testset.txt.
    Args : 
        - file_path : string, a full path to where the testfiles.txt is
        - base_dir : relative path, between whats stored in the txt file
        and where the data is stored. 
    Return : 
        - [] : list of filenames with absolute paths.
    """
    with open(file_path, 'r') as f:
        return [os.path.join(base_dir, line.strip()) for line in f]

def sample_enrollment_files(test_files, speaker, num_samples=3):
    """Randomly sample files for enrollment for the specified speaker."""
    speaker_files = [f for f in test_files if speaker in os.path.basename(f)]
    enrollment_files = random.sample(speaker_files, num_samples)
    remaining_files = [f for f in test_files if f not in enrollment_files]
    return enrollment_files, remaining_files
