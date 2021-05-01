$ docker build -t nasa .
$ docker run -v "[RUTA A training_code]:/nasa/DATA" nasa

EXAMPLE : /home/sebasmos/Documentos/NASA_Spacesuit/Model_1_short_segmentation/Training_code
# Debe ir la ruta absoluta completa segun el sistema opertivo

EXAMPLE

$ docker run -v "...\Desktop\Nasa\Model_1_short_segmentation\Training_code:/nasa/" nasa/challenge

# docker run -v "/home/sebasmos/Documentos/NASA_Spacesuit/Model_1_short_segmentation/Training_code:/nasa/" nasa


