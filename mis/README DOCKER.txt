$ docker build -t nasa.
$ docker run -v <local_data_path>:/data:ro -v <path to solution folder >:/wdata -it nasa


EXAMPLE : ~\Desktop\Nasa\Model_1_short_segmentation\code
# Debe ir la ruta absoluta completa segun el sistema opertivo

EXAMPLE
$ docker run -v "C:\Users\Christian\Desktop\Nasa\Model_1_short_segmentation\Training_code:/nasa/" nasa/challenge

docker run -v <local_data_path>:/data:ro -v <local_writable_area_path>:/wdata -it <id>





# # Download datasets
# #https://drive.google.com/uc?id=1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh/
#https://drive.google.com/u/0/uc?export=download&confirm=cDYI&id=1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh
# #https://drive.google.com/file/d/11Zf8BLSr3v7dMeug5g2WMdyMl3IBeB_V/
# # download train 

WORKDIR /nasa/data
#RUN gdown --no-cookies https://.google.com/uc?id=1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh/  \
RUN wget https://drive.google.com/u/0/uc?export=download&confirm=cDYI&id=1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh

RUN curl -o train.zip https://drive.google.com/u/0/uc?export=download&confirm=cDYI&id=1rs7x3-e8eswLeQljxfdwK4MVbhxlyhyh 
RUN ls
RUN unzip train.zip 
RUN rm train.zip 

RUN rm train.zip

# # download test
# RUN curl -sS http://foo.bar/filename.zip > file.zip \
#     && unzip file.zip \
#     && rm file.zip


#
#CMD while true; do sleep 3600; done
