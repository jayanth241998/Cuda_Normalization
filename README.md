### CS451 Assignment 4

there are 2 files

1. matcuda.cu is the source code and its executable matcuda, along with its library files.
2. matrixnorm.c 


the program was written and run in the following environment

1. compute capability of 8.6 , with NVIDIA GeForce RTX 3050 Ti.
2. CUDA version used is 11.8


**how to compile CUDA code**
use the following command

**compile with following command**
nvcc -o matcuda matcuda.cu

**execute with following command**
matcuda 8000 0 4 256


**arguments to be passed**

matcuda arg1 arg2 arg3 arg4

arg1 -> integer value for matrix size,e.g. for 4 * 4 matrix arg1 should be 4.

arg2 -> seed value for random number generation.

arg3 -> number of blocks per grid to be used.

arg4 -> number of threads per block to be used.

**note**
arg3 and arg4 are optional , if not provided they are set by default to 4 blocks and 256 threads.


**executing and compiling C serial matrix normalization program**

**compile**
gcc matrixnorm.c -o matserial

**execution**
matserial 4000 0


matserial arg1 arg2
where,
arg1-> matrix size.
arg2-> seed value.





 
 
