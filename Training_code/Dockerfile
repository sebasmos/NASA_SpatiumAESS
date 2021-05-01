FROM python:3.6-slim-buster

# # Working directory
# ARG MAIN_DIR=/opt/frontend/

# Set working directory
WORKDIR /nasa

#RUN wget https://drive.google.com/file/d/1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh/view?usp=sharing
#RUN wget https://drive.google.com/file/d/11Zf8BLSr3v7dMeug5g2WMdyMl3IBeB_V/view?usp=sharing
#RUN mv train/ code/train 
#RUN mv test/ code/test 

# Copy the code
COPY ./requirements.txt /nasa/

# Install all the necessary dependencies and folders
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && python3 -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y python3-dev libpq-dev build-essential

# Run command


#CMD ["sh","code/main.sh"]
#CMD while true; do sleep 3600; done
