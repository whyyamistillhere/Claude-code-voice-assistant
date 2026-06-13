# This is the client and it's purpose is to detect the wake word then record the speech
# until some time of seconds has been passed, then it sends it to the server
# then it waits for the response from the server and then plays it.

# The clients code logic: wait's for a wakeword --> when detected --> Starts recording for when X seconds of silence
# has passed --> sends the sound to the server --> wait's until the server sends audio back --> when audio has
# arrived it plays it.

# what == what is it doing?
# why == why is it there?

# oww == open wake word
# vad == webrtcvad

print("📃 Starting up 📃")

print("📃 Importing packages 📃")
from openwakeword.model import Model
import yaml
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from math import gcd
import webrtcvad
import requests
import numpy as np
import io

print("📃 packages imported 📃")

# what: it will see the microphones and speakers capabilities. Like the samplerate, is it mono or stereo and etc
# why: to ensure compatipility across all devices
def default_mic_and_speaker():
    # getting the default microphone and speaker
    # then getting the samplerate and channels
    print("📃 Searching what's the default microphone and speaker 📃")
    default_mic = sd.default.device[0]
    default_speaker = sd.default.device[1]
    mic_capabilites = sd.query_devices(default_mic)
    speaker_capabilites = sd.query_devices(default_speaker)

    # defining all of the microphones samplerate and input channels
    mic_channels = mic_capabilites["max_input_channels"]
    mic_samplerate = mic_capabilites["default_samplerate"]
    return default_mic, mic_samplerate, mic_channels, default_speaker

# what: it is converting the samplerate from the microphone to 16 kHz for oww
# and converting stereo to mono if the microphone is stereo
# why: since oww and vad needs 16000 kHz and mono audio
def samplerate_and_channel_conversion(audio_data, mic_input_samplerate):
    if audio_data.ndim > 1:
        mono_audio = np.mean(audio_data, axis=1).astype(np.int16)
    else:
        mono_audio = audio_data
    
    # working out the resampling ratio
    common_divisor = gcd(int(mic_input_samplerate), 16000)
    up = 16000 // common_divisor
    down = int(mic_input_samplerate) // common_divisor

    processed_audio = resample_poly(mono_audio, up=up, down=down)

    processed_audio = processed_audio.astype(np.int16)
    return processed_audio


# what: it is predicitng is this piece of audio a wake word if it is then 
# then it return true which then can continue the loop 
# why: to detect if you need to record the command or not
def wake_word_detecting(stream, oww_model, mic_samplerate, wake_word_confidence):
        frames = int(mic_samplerate * 0.08)
        audio_data, overflowed = stream.read(frames=frames)
        oww_audio = samplerate_and_channel_conversion(audio_data=audio_data, mic_input_samplerate=mic_samplerate)
        wake_word_prediction = oww_model.predict(oww_audio)

        for wake_word, score in wake_word_prediction.items():
            if score > wake_word_confidence:
                print("wake word has been detected", wake_word)
                return True

# what: just plays a sound through the speakers
# why: so the user know that the wake word has been detected
def play_notification_sound(notification_sound, default_speaker):
    print("📃 Playing notification sound 📃")
    samplerate_noti, noti_data = wav.read(notification_sound) # Reading the .wav file

    # Playing the audio to the speakers
    sd.play(noti_data, samplerate_noti, device=default_speaker)

# what: it is recording the command until some amount of time has passed using webrtcvad
def webrtcvad_and_command_recording(stream, vad, mic_samplerate, silence_time):
    # Definening a counter and audio chunks for while loop
    silence_counter = 0 # just a variable
    audio_chunks = [] # holds the actual audio
    silence_chunks = silence_time / 0.030 # why: since the vad reads 30 millisecond chunks at a time
    frames = int(mic_samplerate * 0.03) # why: every microphone has diffrent samplerate and you need to calculate for each one the exact amount of frames
    
    print("📃 recording the command 📃")
    while silence_counter < silence_chunks:
        # Converting the samplerate and channels to the prober ones, aka 16 kHz samplerate and mono audio
        audio_data, overflowed = stream.read(frames=frames)
        vad_audio = samplerate_and_channel_conversion(audio_data=audio_data, mic_input_samplerate=mic_samplerate)
        vad_bytes = vad_audio.tobytes()

        # what this does is that it checks a audio chunk then it tells if it contains speech or not.
        # If not then it puts +1 to the silence counter until it breaks from it's loop
        # If it did detect speech then it resets the counter
        if vad.is_speech(vad_bytes, sample_rate=16000) is False:
            silence_counter += 1
        elif vad.is_speech(vad_bytes, sample_rate=16000) is True:
            silence_counter = 0

        audio_chunks.append(vad_audio)
    
    full_audio_chunks = np.concatenate(audio_chunks)
    full_audio_chunks = full_audio_chunks.tobytes()
    print("📃 command has been recorded now going to the server")
    return full_audio_chunks

def server_audio(server_IP, server_port, full_audio_chunks):
    print("📃 sending the command to the server📃")
    server_request = requests.post(f"http://{server_IP}:{server_port}/process", data=full_audio_chunks)
    server_response = server_request.content
    server_response_mem = io.BytesIO(server_response)
    print("📃 Got the reponse back now playing it 📃")
    server_response_audio_samplerate, server_response_audio_data = wav.read(server_response_mem)
    return server_response_audio_samplerate, server_response_audio_data


default_mic, mic_samplerate, mic_channels, default_speaker = default_mic_and_speaker()

# ============================================
# opening the config.yaml file for defining some config
with open('client-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# defining the variables from yaml file. The yaml file is config.yaml
wake_word_confidence = config['wake_word_confidence']
wake_word_models = config['wake_word_models']
server_IP = config['server_IP']
server_port = config['server_port']
# what: This line will check, is the yaml variable not empty AKA None, if
# it is not empty then it sets the value from the yaml file
# why: because when the user want's to select the device him self
# and even if there is no device put there, then it still overrides the default.
if config['Input_device'] is not None:
    default_mic = config['Input_device']
if config['Output_device'] is not None:
    default_speaker = config['Output_device']

notification_sound = config['notification_sound']
silence_time = config['silence_time']
# ============================================

print("📃 Loading up the Openwake word models and webrtcvad 📃")
oww_model = Model(wake_word_models)
vad = webrtcvad.Vad(3)
print("📃 Openwake word and webrtcvad has been loaded 📃")
print("📃 microphone selected", default_mic, "📃 speaker selected", default_speaker)
print("📃 Listening for wake word 📃")

# Main loop
with sd.InputStream(samplerate=mic_samplerate,device=default_mic, channels=mic_channels, dtype="int16") as stream:
    while True:
        # detecting the wake word        
        if wake_word_detecting(stream=stream, oww_model=oww_model, mic_samplerate=mic_samplerate, wake_word_confidence=wake_word_confidence, ) is True:
            print("📃 wake word has been detected")
            play_notification_sound(notification_sound=notification_sound, default_speaker=default_speaker)
            
            # recording the command and for X time of silence
            full_audio_chunks = webrtcvad_and_command_recording(stream=stream, vad=vad, mic_samplerate=mic_samplerate, silence_time=silence_time)
            
            # sending the audio to the server to let it convert the audio to text, then sending it to the AI agent
            audio_back_samplerate, audio_back_data = server_audio(server_IP=server_IP,server_port=server_port, full_audio_chunks=full_audio_chunks)
            
            # playing the response from the server
            sd.play(audio_back_data, samplerate=audio_back_samplerate, device=default_speaker)
            sd.wait()
            oww_model.reset() # the wake word model has a little buffer of audio and if you don't reset it then it will misfire 