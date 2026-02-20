import sounddevice as sd

INPUT_DEVICE = 6       # Your microphone
OUTPUT_DEVICE = 6   # Default speakers
SAMPLERATE = 48000

def callback(indata, outdata, frames, time, status):
    # Copy microphone input directly to speaker output
    outdata[:] = indata

# Open a stream that reads from mic and writes to speakers simultaneously
with sd.Stream(samplerate=SAMPLERATE,
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        channels=1,
        callback=callback):
    print("Listening... press Enter to stop")
    input()

print("Stopped.")
