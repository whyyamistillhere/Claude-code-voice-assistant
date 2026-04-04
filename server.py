print("📃 starting up 📃")
print("📃 Importing packages")
from flask import Flask, request, send_file
import whisper
import yaml
from piper import PiperVoice
from time import sleep
import subprocess
import io

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
    audio_bytes = request.data   # the audio arrives here
        
    # making audio to text
    result = whisper.transcribe(audio_bytes)
    print("Audio arrived and it has been processed, now going to claude")

    # this code block run's the "claude -p (prompt)" command
    response = subprocess.run(['claude', '-p', result],
    capture_output=True,
    text=True,
    )
    print("Here's claude's response:", response) # The output from this command also contains some other info, not just the claude's response

    claudes_response = response.stdout.strip() # This cleans up the response from the subprocess command, becuase it also gives some uneeded data
    
    # This code block will make the text to speech
    with io.BytesIO.write("voice.wav", "wb") as voice_file:
        PiperVoice.synthesize_wav(claudes_response, voice_file)

    return send_file(io.BytesIO.seek)


app.run(host="0.0.0.0", port=5000)

process()