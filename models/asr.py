from dependencies import torch, torchaudio, pipeline, nn

class Wav2Vec2ASR(nn.Module):
    def __init__(self):
        super(Wav2Vec2ASR, self).__init__()
        # Load pretrained Wav2Vec 2.0 model bundle
        model_name = "facebook/wav2vec2-base-960h"  # Choose a lightweight model variant
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-base", device=device)
        self.sample_rate = 16000

    def forward(self, waveform, sample_rate=16000):
        """Forward method for transcribing audio."""
        # Resample the audio if it doesn't match the model's sample rate
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform).detach().cpu().numpy()

        if waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform.flatten()
        # Inference
        with torch.no_grad():
            transcription = self.model(waveform, return_timestamps=True)
        return transcription['text']