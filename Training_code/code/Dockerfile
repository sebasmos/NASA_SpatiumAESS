FROM python:3.6-slim-buster


# Install all the necessary dependencies and folders
RUN apt-get update -y\
    && apt-get install -y \ 
    ffmpeg \
    libsm6 \
    libxext6  -y \
    wget \
    curl \
    cmake \
    unzip \
    && python3 -m pip install --upgrade pip 

RUN pip install gdown

WORKDIR /nasa/
COPY ./requirements.txt /nasa/

RUN pip install -r requirements.txt \
    && apt-get purge -y python3-dev libpq-dev build-essential

COPY . /nasa/
#RUN sh train.sh

#RUN sh test.sh
