$ docker build -t nasa .
$docker run -v <local_data_path>:/nasa/data:ro -v <path to solution folder >:/nasa/wdata -t nasa sh train
$docker run -v <local_data_path>:/nasa/data:ro -v <path to solution folder >:/nasa/wdata -t nasa /bin/bash train.sh

$ docker run -v <local_data_path>:/nasa/data:ro -v <path to solution folder >:/nasa/wdata -it nasa bash

EJEMPLO 1
docker run -v C:\Users\Christian\Desktop\Nasa\DATA:/nasa/data:ro -v C:\Users\Christian\Desktop\Nasa\Model_1_short_segmentation\Training_code\solution:/nasa/wdata -it nasa bash

EJEMPLO 2
docker run -v C:\Users\Christian\Desktop\Nasa\DATA:/nasa/data:ro -v C:\Users\Christian\Desktop\Nasa\Model_1_short_segmentation\Training_code\solution:/nasa/wdata nasa/challenge sh train.sh