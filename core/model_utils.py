# model_utils.py

from dependencies import torch, torchaudio, np
from config import SAMPLE_RATE
from models import Wav2Vec2ASR

torchaudio.set_audio_backend("soundfile")

def load_model():
    """Load the ReDimNet model with predefined configuration."""
    model = torch.hub.load('IDRnD/ReDimNet', 'b0', pretrained=True, finetuned=True)
    model.eval()
    asr_model = Wav2Vec2ASR().eval()
    return model, asr_model

def process_audio(model, audio_path):
    """Load an audio file, resample if needed, and generate an embedding."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
    
    with torch.no_grad():
        embedding = model(waveform)
    return embedding.squeeze().cpu().numpy().flatten()

def generate_reference_embedding(model, enrollment_files):
    """Generate a reference embedding by averaging embeddings from enrollment files."""
    embeddings = [process_audio(model, file) for file in enrollment_files]
    return np.mean(embeddings, axis=0)
