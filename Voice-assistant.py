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
import webrtcvad
import time

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
silence_time = config['silence_time']

# Loading the Openwake word models and webrtcvad
print("📃 Loading up the Openwake word models and webrtcvad 📃")
oww_model = Model(wake_word_models)
vad = webrtcvad.Vad(3)
print("📃 Open wake word models loaded and webrtcvad 📃")

print("📃 Everything loaded, starting in 3 seconds")
time.sleep(3)

# getting the models prediction from the microphone audio and say detected if it is detected
print("📃 Recording and predicting 📃")

# this little block is calculating how many 30 millisconds chunks are in the silence_time string. Then it rounds them up, becasue python might not like it.
silence_chunks = silence_time / 0.030
rounded_silence_chunks = round(silence_chunks)

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
                print("📃 Playing notification sound 📃")
                samplerate_noti, noti_data = wav.read(notification_sound) # Reading the .wav file
                
                # Playing the audio to the speakers
                sd.play(noti_data, samplerate_noti, device=output_device)
                sd.wait()

                print("📃 Listening the command and then processing it with STT 📃")


                # Converting the audio data from an numpy array to bytes for webrtcvad
                silence_counter = 0
                while silence_counter < 100:
                    
                    # Converting the audio data from an numpy array to bytes for webrtcvad
                    whisper_audio_data, whisper_overflowed = stream.read(frames=1280)
                    webrtcvad_audio_data = whisper_audio_data.tobytes()

                    if vad.is_speech(webrtcvad_audio_data, sample_rate=16000) is False:
                        silence_counter =+ 1
                    
                    elif vad.is_speech(webrtcvad_audio_data, sample_rate=16000) is True:
                        silence_counter = 0
                
                print("3 Seconds of silence")