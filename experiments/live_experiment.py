# experiments/live_experiment.py

from core import generate_reference_embedding, record_audio, evaluate_live 
from core import sample_enrollment_files, load_test_files
from config import THRESHOLD

def run_live_experiment(v_model, asr_model, test_files, speaker_id):
    enrollment_files, _ = sample_enrollment_files(test_files, speaker_id)
    reference_embedding = generate_reference_embedding(v_model, enrollment_files)
    print(f"Running live verification for {speaker_id}.")

    waveform, _ = record_audio()
    if waveform is None:
        print("Failed to record audio.")
        return

    test_embedding = v_model(waveform)
    transcript = ""
    if evaluate_live(v_model, reference_embedding, test_embedding, THRESHOLD):
        waveform = waveform.detach().cpu().numpy()
        transcript = asr_model(waveform.copy())
        print(f"This is {speaker_id} and he says {transcript}")
    else:
        print(f"This is not {speaker_id}, so we don't care what they say")
    return transcript
