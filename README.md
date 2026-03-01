# Claude-code-voice-assistant
Use your voice to ask claude code to do stuff

# THIS IS INCOMPLETE !!!

I am a beginner programmer and so this might take a while.

# Features
|Feature|Implemented|
|---|---|
|Yaml config file|✅|
|Openwake word for the wake word engine|✅|
|Fully local STT and TTS and having a way to offload it to a diffrent server|❌|
|Optional OpenAI STT API key support and elevenlabs TTS|❌
|Claude code for the AI agent|❌|
|Openclaw as the AI agent, but this is a maybe|❔|

# Packages

### Client packages
```
pip install openwakeword pyyaml sounddevice numpy flask
```
### Server packages
```
pip install openai-whisper piper-tts flask pyyaml
```