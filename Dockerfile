FROM python:3.13

WORKDIR /client

RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install openwakeword pyyaml sounddevice numpy requests webrtcvad scipy

COPY client.py /client/
COPY *.onnx /client/
COPY *.tflite /client/
COPY *.wav /client/

CMD ["python", "client.py" ]