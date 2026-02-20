# oww is open wake word

print("📃 Starting up 📃")

print("📃 Importing packages 📃")
import os
import openwakeword
from openwakeword.model import Model
import yaml
#from wyoming.tts import Synthesize
#from wyoming.audio import AudioChunk
#from wyoming.audio import AudioChunk, AudioStop
import asyncio
import sounddevice as sd
import numpy as np


print("📃 packages imported 📃")

# defining microphone variables for sounddevice
samplerate=16000
channels=1


# opening the config.yaml file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# defining the variables from yaml file
whisper_model_py = config['whisper_model']
wake_word_confidence = config['wake_word_confidence']
wake_word_models = config['wake_word_models']
wyoming_STT_IP = config['whisper_STT_IP']
wyoming_TTS_IP = config['piper_TTS_IP']
device = config['Input&output_device']

# Loading the Openwake word models
print("📃 Loading up the Openwake word models 📃")
oww_model = Model(wake_word_models)
print("📃 Open wake word models loaded 📃")

# getting the models prediction from the microphone audio and say detected if it is detected
print("📃 Recording and predicting 📃")
while True:   
    audio_data = sd.rec(
    samplerate=samplerate,
    channels=channels,
    dtype="int16",
    device=device,
    frames=1280
    )
    sd.wait()

##    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    prediction = oww_model.predict(audio_data) # Model predicitng the wake word

    print(f"Scores: {prediction}") # this line is for debugging

    # this block says if the model has predicted the wake word
    print("Listening for wake word")
    for wake_word, score in prediction.items():
        if score > 0.3:
            print("wake word detected '{wake_word}' detected ")
