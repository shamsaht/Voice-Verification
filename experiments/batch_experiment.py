# experiments/batch_experiment.py

from core import generate_reference_embedding, process_audio
from core import compute_metrics, sample_enrollment_files
from dependencies import cosine

def run_batch_experiment(model, test_files, speaker):
    enrollment_files, remaining_files = sample_enrollment_files(test_files, speaker)
    reference_embedding = generate_reference_embedding(model, enrollment_files)

    similarities, labels = [], []
    for file in remaining_files:
        test_embedding = process_audio(model, file)
        similarity = 1 - cosine(reference_embedding, test_embedding)
        similarities.append(similarity)
        labels.append(1 if speaker in file else 0)
    
    eer, min_dcf, eer_threshold, min_dcf_threshold = compute_metrics(similarities, labels)
    print(f"Speaker {speaker}: EER = {eer:.4f}, minDCF = {min_dcf:.4f}")
    print(f"EER Threshold: {eer_threshold:.4f}, minDCF Threshold: {min_dcf_threshold:.4f}")
