#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include <time.h>
#include <math.h>


int N;
float *A,*B;


__global__ void MatrixNorm(float *matA,float *matB,int N)
{
   int row, col;
   float mu, sigma;
   col = blockDim.x * blockIdx.x + threadIdx.x;
   if(col < N)
   {
    mu = 0.0;
        for (row=0; row < N; row++)
            mu += matA[row*N + col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(matA[row*N + col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                matB[row*N + col] = 0.0;
            else
                matB[row*N + col] = (matA[row*N + col] - mu) / sigma;
        }
   }

}

void Initialize_Matrix()
{

  int row, col;
    A = (float *)malloc(sizeof(float)* N * N);
    B = (float *)malloc(sizeof(float)* N * N);
    
    for (row = 0; row < N; row++) 
    { 
        for (col = 0; col < N; col++) {
           
            A[row*N + col] = (float)rand() / 32768.0;
            B[row*N + col] = 0.0;
        }
    }
}


void Initialize_parameters(int argc,char **argv)
{
  N = atoi(argv[1]);
  int seed = atoi(argv[2]);
  srand(seed);

}

void print_inputs()
{
  int row,col;
   for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        printf("%f ",A[row*N + col]);
     }
     printf("\n");
   }
}

void print_output()
{
  int row,col;
   for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        printf("%f ",B[row*N + col]);
     }
     printf("\n");
   }
}


int main(int argc,char **argv)
{
   int row;
   cudaError_t error = cudaSuccess;

   Initialize_parameters(argc,argv);
   Initialize_Matrix();
   print_inputs();

   float *d_A = NULL;
   
   error = cudaMalloc((void **)&d_A, sizeof(float)*N*N);

  
  if (error != cudaSuccess)
   {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }
   

   error = cudaMemcpy(d_A,A,sizeof(float)*N*N,cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   float *d_B = NULL;

    error = cudaMalloc((void **)&d_B, sizeof(float)*N*N);

  
  if (error != cudaSuccess)
   {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }
   

   error = cudaMemcpy(d_B,B,sizeof(float)*N*N,cudaMemcpyHostToDevice);

   if (error != cudaSuccess)
   {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   

   int threadsPerBlock = 256;
   int blocksPerGrid  = ( N + threadsPerBlock - 1)/threadsPerBlock;
   MatrixNorm<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,N);
   error = cudaGetLastError();

   if (error != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

    printf("Copy output data from the CUDA device to the host memory\n");
  error = cudaMemcpy(B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

   if (error != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy matrix B from device to host (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  print_output();


  error = cudaFree(d_A);

   if (error != cudaSuccess) {
    fprintf(stderr, "Failed to free device matrix A (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   error = cudaFree(d_B);

   if (error != cudaSuccess) {
    fprintf(stderr, "Failed to free device matrix A (error code %s)!\n",
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   free(A);
   free(B);
   
printf("done\n");
 return 0;
}

