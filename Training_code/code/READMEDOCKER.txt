$ docker build -t nasa .
$ docker run -v <local_data_path>:/data:ro -v <path to solution folder >:/wdata -it nasa
