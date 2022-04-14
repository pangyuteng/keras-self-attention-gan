FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

WORKDIR /opt
COPY requirements.txt .
RUN pip install -r requirements.txt
