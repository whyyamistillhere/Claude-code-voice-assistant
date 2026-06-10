print("📃 starting up 📃")
print("📃 Importing packages")
from flask import Flask, request, send_file
import whisper
import yaml
from piper import PiperVoice
from time import sleep
import subprocess
import io
import numpy as np
import wave

# opening the config.yaml file for defining some config
with open('server-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# definening config from server-config.yaml
piper_model = config['piper_model_path']
whisper_model = config['whisper_model']
eleven_labs_API_key = config['eleven_labs_API_key']
openai_API_key = config['openai_API_key']

print("📃 Loading up whisper and piper 📃")
stt_model = whisper.load_model(whisper_model)
sleep(3)
tts_voice = PiperVoice.load(piper_model)

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():

    
    # This code block recieves the audio and converts the int16 to float32 for whisper
    audio_bytes = request.data   # the audio arrives here
    print("📃 Audio has arrived and is now being processed 📃")
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    whisper_audio = audio_array.astype(np.float32) / 32768.0

    # making audio to text
    result = stt_model.transcribe(whisper_audio)
    print("📃 Audio now going to the AI agent 📃")

    # this code block run's the AI coding tool command
    response = subprocess.run(['codex', 'exec', result["text"]],
    capture_output=True,
    text=True,
    )
    AI_response = response.stdout.strip() # This cleans up the response from the subprocess command, becuase it also gives some uneeded data

    print("Here's the AI agents response:", AI_response) # The output from this command also contains some other info, not just the claude's response

    
    print("📃 Converting the text to voice 📃")
    # This code block will make the text to speech
    voice_file = io.BytesIO()
    with wave.open(voice_file, "wb") as wav_writer:
        tts_voice.synthesize_wav(AI_response, wav_writer)
    print("📃 Voice has been converted and now it is being send back to the server")

    voice_file.seek(0)
    return send_file(voice_file, mimetype="audio/wav") # Send's the audio data


app.run(host="0.0.0.0", port=5000)