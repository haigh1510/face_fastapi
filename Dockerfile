FROM python:3.8-slim

WORKDIR /facerec

COPY facerec_module facerec_module
COPY requirements.txt requirements.txt

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y cmake && \
    apt-get install -y libgl1 libglib2.0-0

RUN pip3 install -r requirements.txt
