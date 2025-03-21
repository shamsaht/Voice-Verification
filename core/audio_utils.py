# core/audio_utils.py

from dependencies import torch, np, sd, keyboard
from config import SAMPLE_RATE, BUFFER_SIZE

def record_audio():
    sd.default.device = 0  # Set the default audio device if needed

    print("Press Enter to start recording.")
    input("Press Enter to start recording...")  # Wait for Enter to start recording
    print("Recording... Press Enter again to stop.")

    audio_chunks = []
    stop_recording = False

    def on_press(key):
        nonlocal stop_recording
        if key == keyboard.Key.enter:
            print("Stopping recording...")
            stop_recording = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            while not stop_recording:
                chunk, _ = stream.read(BUFFER_SIZE)
                audio_chunks.append(chunk)

        waveform = torch.from_numpy(np.concatenate(audio_chunks).T)
        return waveform, SAMPLE_RATE

    except Exception as e:
        print(f"Error during recording: {e}")
        return None, None
    finally:
        listener.stop()
