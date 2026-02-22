# oww is open wake word

print("📃 Starting up 📃")

print("📃 Importing packages 📃")
import os
import openwakeword
from openwakeword.model import Model
import yaml
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk
from wyoming.audio import AudioChunk, AudioStop
import asyncio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
print("📃 packages imported 📃")

# opening the config.yaml file for defining some config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# defining the variables from yaml file. The yaml file is config.yaml
whisper_model_py = config['whisper_model']
wake_word_confidence = config['wake_word_confidence']
wake_word_models = config['wake_word_models']
wyoming_STT_IP = config['whisper_STT_IP']
wyoming_TTS_IP = config['piper_TTS_IP']
input_device = config['Input_device']
output_device = config['Output_device']
notification_sound = config['notification_sound']

# Loading the Openwake word models
print("📃 Loading up the Openwake word models 📃")
oww_model = Model(wake_word_models)
print("📃 Open wake word models loaded 📃")

# getting the models prediction from the microphone audio and say detected if it is detected
print("📃 Recording and predicting 📃")

with sd.InputStream(samplerate=16000, device=input_device, channels=1, dtype="int16",) as stream:  # This line here opens up the microphone 
    while True:
        prediction_audio_data, overflowed = stream.read(frames=1280)
        prediction = oww_model.predict(prediction_audio_data.squeeze()) # Model predicitng the wake word

        # this block says if the model has predicted the wake word
        print("📃 Listening for wake word 📃")
        for wake_word, score in prediction.items():
            if score > wake_word_confidence: # The wake_word_confidence is a number in the config.yaml
                print("📃 wake word detected and playing notification sound 📃")

                # playing the notification sound
                samplerate_noti, noti_data = wav.read(notification_sound) # Reading the .wav file
                
                sd.play(noti_data, samplerate_noti, device=output_device)
                sd.wait()
