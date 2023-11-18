CC = gcc
OPENCL_INCLUDE_PATH = /home/paasio/Documents/parallel_programming/OpenCL-Headers
OPENCL_LIB_PATH = /home/paasio/Documents/parallel_programming/OpenCL-Headers

canny: canny.c
	$(CC) -o canny util.c opencl_util.c canny.c -lm -O3 -ftree-vectorize -mavx2 -ffast-math -I${OPENCL_INCLUDE_PATH} -L${OPENCL_LIB_PATH} -lOpenCL

