FROM ubuntu:latest
FROM python:latest

COPY requirements.txt /
COPY imagelabels.mat /
COPY setid.mat /
COPY model.h5 /
# test images are in testdata folder
COPY testx.npy /
COPY testy.npy / 
# Weights are in flowers_model folder
COPY  weights.h5 /
# Installing dependancies
RUN pip install -r /requirements.txt
# Copy inference program inside docker container
COPY inference.py /
# Giving permission to inference program
RUN chmod u+x ./inference.py
# Running inference program
CMD ./inference.py
