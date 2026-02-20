#!/usr/bin/env python3
"""
RPi Voice Assistant for Claude Code with Whisper STT and openWakeWord
Supports multiple custom wake words simultaneously
Optimized for Raspberry Pi 4 with USB mic/speaker
"""

import os
import sys
import subprocess
import tempfile
import wave
import whisper
import pyaudio
import time
import numpy as np
from datetime import datetime, timedelta
from openwakeword.model import Model
import webrtcvad

# Configuration
WHISPER_MODEL = "base"
PIPER_VOICE_PATH = os.path.expanduser("~/piper-voices/en_GB-alan-medium.onnx")
NOTIFICATION_SOUND = os.path.expanduser("~/voice-assistant/formula-1-radio-notification.wav")
SAMPLE_RATE = 16000
WAKE_WORD_THRESHOLD = 0.50  # 0.0 to 1.0, higher = more confident required
CONVERSATION_TIMEOUT = 900  # 15 minutes in seconds
COMMAND_DURATION = 10  # Maximum seconds to record (will stop early if silence detected)
VAD_SILENCE_DURATION = 1.5  # Seconds of silence before stopping recording

# Wake word models to load
# Add your custom .onnx models here!
WAKE_WORD_MODELS = [
    'Sudo.onnx',      # Pre-trained community model
    'Suto.onnx',           # Pre-trained model
    # 'hey_sudo.onnx',      # Your custom trained model (add when ready!)
]

class VoiceAssistant:
    def __init__(self):
        print("Initializing Sudo Voice Assistant with openWakeWord...")

        # Initialize openWakeWord
        print("Loading wake word detector...")
        try:
            # openWakeWord can load multiple models simultaneously
            # VAD (Voice Activity Detection) filters out non-speech sounds
            self.oww_model = Model(
                wakeword_models=WAKE_WORD_MODELS,
                inference_framework='onnx',  # Use ONNX runtime for RPi
                vad_threshold=0.5  # Enable VAD to filter ambient noise
            )

            print(f"✓ Wake word detector loaded!")
            print(f"Active wake words: {list(self.oww_model.models.keys())}")

        except Exception as e:
            print(f"Error loading openWakeWord: {e}")
            print("Make sure you've installed: pip install openwakeword")
            sys.exit(1)

        # Load Whisper model
        print(f"Loading Whisper {WHISPER_MODEL} model...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        print("✓ Whisper loaded!")

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Initialize WebRTC VAD for silence detection
        # Mode 3 = most aggressive filtering (best for noisy environments)
        self.vad = webrtcvad.Vad(3)
        print("✓ Voice Activity Detection initialized!")

        # Load Piper voice
        print("Loading text-to-speech voice...")
        from piper import PiperVoice
        self.piper_voice = PiperVoice.load(PIPER_VOICE_PATH)
        print("✓ Piper voice loaded!")

        # Conversation state
        self.conversation_id = None
        self.last_interaction_time = None
        self.conversation_start_time = None

        print("✓ Voice Assistant initialized successfully!")
    
    def _get_conversation_id(self):
        """Get or create conversation ID, resetting after timeout"""
        current_time = datetime.now()
        
        if (self.conversation_id is None or 
            self.last_interaction_time is None or
            (current_time - self.last_interaction_time).total_seconds() > CONVERSATION_TIMEOUT):
            
            old_id = self.conversation_id
            self.conversation_id = f"voice-{int(time.time())}"
            self.conversation_start_time = current_time
            
            if old_id:
                print(f"\n⏱️  Conversation timeout - starting new conversation")
                print(f"New: {self.conversation_id}\n")
            else:
                print(f"\n🆕 Starting conversation: {self.conversation_id}\n")
        
        self.last_interaction_time = current_time
        return self.conversation_id
    
    def _get_conversation_duration(self):
        """Get how long the current conversation has been active"""
        if self.conversation_start_time:
            duration = datetime.now() - self.conversation_start_time
            minutes = int(duration.total_seconds() // 60)
            return f"{minutes} min"
        return "0 min"

    def _play_notification(self):
        """Play the F1 radio notification sound"""
        try:
            print("🔔 Playing notification...")
            # Play pre-converted WAV file directly with aplay
            subprocess.run(['aplay', '-q', NOTIFICATION_SOUND], check=True, timeout=3)
        except subprocess.TimeoutExpired:
            print("⚠️  Notification timed out")
        except Exception as e:
            print(f"⚠️  Could not play notification: {e}")
            # Don't fail the whole flow if notification doesn't play

    def listen_for_wake_word(self):
        """
        Continuously listen for wake words using openWakeWord
        Returns: tuple of (detected: bool, wake_word_name: str)
        """
        try:
            # CRITICAL: Reset model state to clear previous detection buffers
            # This prevents the model from "remembering" the last wake word
            self.oww_model.reset()
            print("🔄 Model state reset")
            
            # openWakeWord expects 1280 samples per chunk (80ms at 16kHz)
            CHUNK_SIZE = 1280
            
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            # CRITICAL: Flush audio buffers to clear residual/buffered audio
            # Read and discard 20 chunks (~1.6 seconds) before starting detection
            print("🔄 Flushing audio buffers...")
            for _ in range(20):
                stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            print("🎧 Listening for wake word...")
            
            while True:
                # Read audio chunk
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Convert to numpy array (int16)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Get predictions for all loaded models
                prediction = self.oww_model.predict(audio_array)
                
                # Check if any wake word exceeded threshold
                for wake_word, score in prediction.items():
                    if score >= WAKE_WORD_THRESHOLD:
                        # Extract clean wake word name (remove .onnx extension)
                        clean_name = wake_word.replace('.onnx', '').replace('_', ' ')
                        print(f"\n✓ Wake word detected: '{clean_name}' (confidence: {score:.2f})")

                        stream.stop_stream()
                        stream.close()

                        # Play F1 notification sound
                        self._play_notification()

                        return True, clean_name
                    
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            return False, None
    
    def record_command(self, max_duration=COMMAND_DURATION):
        """Record audio for command after wake word detected, stops on silence"""
        print(f"🎤 Recording command (max {max_duration}s, auto-stop on silence)...")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # WebRTC VAD works best with specific frame sizes
            # 30ms frames at 16kHz = 480 samples
            FRAME_DURATION_MS = 30
            CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples

            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            frames = []
            num_silent_chunks = 0
            num_chunks_for_silence = int(VAD_SILENCE_DURATION * 1000 / FRAME_DURATION_MS)
            max_chunks = int(max_duration * 1000 / FRAME_DURATION_MS)

            print("🎙️  Listening...", end="", flush=True)

            for i in range(max_chunks):
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)

                    # Check if this chunk contains speech
                    is_speech = self.vad.is_speech(data, SAMPLE_RATE)

                    if is_speech:
                        num_silent_chunks = 0
                        print(".", end="", flush=True)
                    else:
                        num_silent_chunks += 1
                        print("-", end="", flush=True)

                    # Stop if we've had enough silence
                    if num_silent_chunks >= num_chunks_for_silence:
                        print(f"\n✓ Silence detected after {len(frames) * FRAME_DURATION_MS / 1000:.1f}s")
                        break

                except Exception as e:
                    print(f"\nAudio capture error: {e}")
                    break

            if num_silent_chunks < num_chunks_for_silence:
                print(f"\n✓ Max duration reached ({max_duration}s)")

            stream.stop_stream()
            stream.close()

            # Save to WAV file
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))

            # Transcribe with Whisper
            print("🔄 Processing with Whisper...")
            result = self.whisper_model.transcribe(
                tmp_path,
                language="en",
                fp16=False
            )

            text = result["text"].strip()
            return text

        except Exception as e:
            print(f"Error during recording: {e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def speak(self, text):
        """Convert text to speech using Piper and play"""
        print(f"🔊 Speaking: {text[:50]}...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            import wave as wave_module
            with wave_module.open(tmp_path, "wb") as wav_file:
                self.piper_voice.synthesize_wav(text, wav_file)
            
            subprocess.run(['aplay', '-q', tmp_path], check=True)
            
            # CRITICAL: Wait for acoustic settling after TTS playback
            # Prevents microphone from picking up residual speaker audio
            print("⏳ Waiting for audio to settle...")
            time.sleep(1.5)
            
        except Exception as e:
            print(f"Error during speak: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def query_claude(self, prompt):
        """Send query to Claude Code and get response with conversation continuity"""
        print(f"💭 Processing with Claude Code...")
        
        conversation_id = self._get_conversation_id()
        duration = self._get_conversation_duration()
        print(f"📝 Conversation: {conversation_id} ({duration})")
        
        try:
            result = subprocess.run(
                ['claude', '--continue', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            response = result.stdout.strip()
            
            if result.returncode != 0:
                error = result.stderr.strip()
                print(f"❌ Error: {error}")
                return f"I encountered an error: {error}"
            
            return response if response else "I processed your request but have no output to share."
            
        except subprocess.TimeoutExpired:
            return "I apologize, but the request took too long to process."
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return f"I encountered an error: {str(e)}"
    
    def run(self):
        """Main loop - wait for wake word, then record and process command"""
        print("\n" + "="*50)
        print("Sudo Voice Assistant Ready")
        print(f"Active wake words: {list(self.oww_model.models.keys())}")
        print(f"Detection threshold: {WAKE_WORD_THRESHOLD}")
        print(f"Conversation timeout: {CONVERSATION_TIMEOUT//60} minutes")
        print("Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        try:
            while True:
                # Wait for wake word (continuous listening)
                wake_detected, wake_word = self.listen_for_wake_word()
                
                if wake_detected:
                    # Record command after wake word
                    command = self.record_command()
                    
                    if command:
                        print(f"Command: {command}")
                        
                        # Query Claude Code
                        response = self.query_claude(command)
                        
                        # Speak response
                        self.speak(response)
                    else:
                        print("⚠️  No command detected")
                    
                    print()  # Blank line for readability
                
        except KeyboardInterrupt:
            print("\n\nShutting down Sudo Voice Assistant...")
            if self.conversation_id:
                print(f"Final conversation: {self.conversation_id}")
                print(f"Duration: {self._get_conversation_duration()}")
            self.audio.terminate()
            print("Goodbye, sir!")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
