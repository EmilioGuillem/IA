import sounddevice as sd
import numpy as np
import wave

# Parameters
DURATION = 5  # seconds
SAMPLE_RATE = 44100
OUTPUT_FILENAME = "output.wav"

print("Recording...")

# Record audio
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished

print("Finished recording.")

# Save the recorded data as a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 2 bytes per sample
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data.tobytes())