FROM python:3.13

WORKDIR /client

RUN apt-get update && apt-get install -y portaudio19-dev
RUN pip install setuptools
RUN pip install openwakeword pyyaml sounddevice numpy requests webrtcvad-wheels scipy

COPY client.py /client/
COPY *.onnx /client/
COPY *.tflite /client/
COPY *.wav /client/
COPY client-config.yaml /client/

CMD ["python", "client.py" ]