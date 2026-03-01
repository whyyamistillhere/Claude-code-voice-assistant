print("📃 starting up 📃")
print("📃 Importing packages")
from flask import Flask, request, send_file
import whisper
import yaml

# opening the config.yaml file for defining some config
with open('server-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# definening config from server-config.yaml
piper_model = config['piper_model_path']
whisper_model = config['whisper_model']
eleven_labs_API_key = config['eleven_labs_API_key']
openai_API_key = config['openai_API_key']

app = Flask(__name__)

while True:
    @app.route('/process', methods=['POST'])
    def process():
        audio_bytes = request.data   # the audio arrives here
        # ... whisper, claude, piper ...
        return send_file("response.wav")

    app.run(host="0.0.0.0", port=5000)