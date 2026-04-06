# Claude-code-voice-assistant
Use your voice to ask claude code to do stuff. For example connect it to your home assistant through mcp and there
it can turn off lights, check sensor's readings and more.

The voice assistant work's like this:
You have the client.py which should be something small like a raspberry pi.
Then you have the server.py which houses the STT and TTS.

Client is listening for the wakeword --> when it has been detected --> it starts listening for the command until 5 seconds of silence has been detected(this can be configured) --> Then it sends the audio to the server --> it wait's for the response from the server and then it plays the response.

Server is listening for the audio data --> when it arrives, it then starts converting the speech to text using openai's whisper --> The text will be sent to claude --> claude's response will be converted to voice using piper --> the voice will be sent back to the client which then plays the audio.

# Features
|Feature|Implemented|
|---|---|
|Yaml config file|✅|
|Openwake word for the wake word engine|✅|
|Fully local STT and TTS and having a way to offload it to a diffrent server|✅|
|Optional OpenAI STT API key support and elevenlabs TTS|❌
|Claude code for the AI agent|✅|
|Having a one liner to setup everything|❌|
|Running the client on a microcontroller or similar, but this is a maybe|❔|
|Openclaw as the AI agent, but this is a maybe|❔|

# Installation

### Step 1 Clone the repo

```
git clone https://github.com/whyyamistillhere/Claude-code-voice-assistant.git && cd Claude-code-voice-assistant
```

### Step 2 creating the python enviorment and installing packages

Install python venv
```
sudo apt install python3-venv
```

Make python venv and activate it

```
python3 -m venv vvv
```
```
source vvv/bin/activate
```
If you see (vvv) next to your user name, that means you done it right.

## Now run one of these commands to install all of the packages

### Client packages
```
pip install openwakeword pyyaml sounddevice numpy flask
```
### Server packages
```
pip install openai-whisper piper-tts flask pyyaml
```

### Step 3 running up the server.py and client.py

