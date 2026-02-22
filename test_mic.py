import sounddevice as sd
from scipy.io.wavfile import write
import yaml

# opening the config.yaml file for defining some config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DEVICE = config['Input_device']

def record_audio(filename, duration=5, samplerate=48000, device=DEVICE):
    """Record audio and save to WAV file"""

    # Get device info to know how many channels it supports
    device_info = sd.query_devices(device)
    channels = device_info['max_input_channels']

    print(f"Recording for {duration} seconds...")

    # Record audio from microphone
    audio_data = sd.rec(
        frames=int(samplerate * duration),
        samplerate=samplerate,
        channels=channels,
        dtype="int16",
        device=device
    )

    # Wait until recording finishes
    sd.wait()

    # Save to WAV file
    write(filename, samplerate, audio_data)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    record_audio("test.wav", duration=5, samplerate=48000, device=6)
