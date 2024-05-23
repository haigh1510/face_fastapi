FROM python:3.8-slim

WORKDIR /facerec

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y cmake && \
    apt-get install -y libgl1 libglib2.0-0

RUN mkdir facerec_module

COPY facerec_module/requirements.txt facerec_module/requirements.txt
COPY requirements.txt requirements.txt

RUN pip3 install -r facerec_module/requirements.txt
RUN pip3 install -r requirements.txt

COPY facerec_module facerec_module
RUN pip3 install ./facerec_module
