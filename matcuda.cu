#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include <time.h>
#include <math.h>

int N;
float *A,*B,*test_B;
bool correct = true;
int threadsPerBlock = 256;
int blocksPerGrid = 4;

//cuda kernel function that runs on GPU cores
__global__ void MatrixNorm(float *matA,float *matB,int N,int divider)
{
   int row, col;
   float mu, sigma;
   //calculating the portion of rows each thread should normalize

   int startCol = (blockDim.x * blockIdx.x + threadIdx.x)*divider;
   int endCol = (startCol + divider) > N ? N : (startCol + divider);

   //performing normalization
   if(startCol< N)
   {
   for(col=startCol; col < endCol;col++)
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

}

//serial function to run on host and to verify cuda output
void Normalize_serial()
{
    int row, col;
    float mu, sigma; // Mean and Standard Deviation
    
    for (col=0; col < N; col++) {
         mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row*N + col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row*N + col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                test_B[row*N + col] = 0.0;
            else
                test_B[row*N + col] = (A[row*N + col] - mu) / sigma;
        }
    }
}

//function to verify output of CUDA program with that of serial function
void verify()
{
  printf("\n");
  printf("verifying \n");
  Normalize_serial();
int row, col;
 for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        if(fabs(B[row*N + col] - test_B[row*N + col]) > 1e-5 )
        {
          printf("%f and %f in %d and %d\n",B[row*N + col],test_B[row*N + col],row,col);
                correct = false;
                return;
        }
     }
   }
}

//function to initialize matrix
void Initialize_Matrix()
{

  int row, col;
    A = (float *)malloc(sizeof(float)* N * N);
    B = (float *)malloc(sizeof(float)* N * N);
    test_B = (float *)malloc(sizeof(float)* N * N);
    for (row = 0; row < N; row++) 
    { 
        for (col = 0; col < N; col++) {
           
            A[row*N + col] = (float)rand() / 32768.0;
            B[row*N + col] = 0.0;
            test_B[row*N + col] = 0.0;
        }
    }
}


//function to initialize program parameters
void Initialize_parameters(int argc,char **argv)
{
  if(argc < 3)
  {
   printf("please enter seed and matrix size: program [matrix] [seed] \n");
   exit(0);
  }
  N = atoi(argv[1]);
  int seed = atoi(argv[2]);
  srand(seed);

  if(argc == 5)
  {
    blocksPerGrid = atoi(argv[3]);
    threadsPerBlock = atoi(argv[4]);
  }

}

//function to print matrix
void print_matrix(float *mat)
{
  int row,col;
   printf("\n");
  if(N < 20)
  {
   for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        printf("%f ",mat[row*N + col]);
     }
     printf("\n");
   }
  }

}


int main(int argc,char **argv)
{
   
   cudaError_t error = cudaSuccess;
   clock_t start;
   clock_t stop;  /* Elapsed times using gettimeofday() */
   double runtime;

   Initialize_parameters(argc,argv);
   Initialize_Matrix();

   if(N<20)
   {
   printf("input matrix \n");
   print_matrix(A);
   }

   printf("Copying input data from the host memory to the CUDA device. \n");
   float *d_A = NULL;
   
   //allocating space for device matrix
   error = cudaMalloc((void **)&d_A, sizeof(float)*N*N);

  
   if(error != cudaSuccess)
   {
    fprintf(stderr,"Failed to copy vector A from host to device (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }
   
   //copying matrix data from host to device
   error = cudaMemcpy(d_A,A,sizeof(float)*N*N,cudaMemcpyHostToDevice);

   if(error != cudaSuccess)
   {
    fprintf(stderr,"Failed to copy vector A from host to device (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   float *d_B = NULL;

    error = cudaMalloc((void **)&d_B, sizeof(float)*N*N);

  
   if(error != cudaSuccess)
   {
    fprintf(stderr,"Failed to copy vector A from host to device (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }
   

   error = cudaMemcpy(d_B,B,sizeof(float)*N*N,cudaMemcpyHostToDevice);

   if(error != cudaSuccess)
   {
    fprintf(stderr,"Failed to copy vector A from host to device (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   
   start = clock();
   
   
   //calculating the no of columns each thread should normalize
   int divider = (N/(threadsPerBlock*blocksPerGrid)) + 1;
  
  // setting minimum columns to zero if total threads are more than columns
   if(divider == 0)
   {
    divider = 2;
   }

   printf("\n");
   printf("performing matrix normalization with %d threads, and %d blocks. \n",threadsPerBlock,blocksPerGrid);
   printf("No. of columns per thread is %d\n",divider);

   //cuda kernel call

   MatrixNorm<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,N,divider);
   error = cudaGetLastError();

   if (error != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

  printf("Copying output data from the CUDA device to the host memory.\n");
  error = cudaMemcpy(B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

   if (error != cudaSuccess) 
  {
    fprintf(stderr,"Failed to copy matrix B from device to host (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  //stopping clock
  stop = clock();
  

  //verifying ouput
  verify();
  
  if(correct == true)
  {
        printf("output is correct\n");
  }
  else if(correct == false)
  {
        printf("output is incorrect\n");
  }

  //calculating runtime
  runtime = (double)(stop - start)/CLOCKS_PER_SEC;
  
  if(N<20)
   {
  printf("output matrix \n");
  print_matrix(B);
   }
  //freeing allocated memory
  error = cudaFree(d_A);

   if (error != cudaSuccess) 
   {
    fprintf(stderr, "Failed to free device matrix A (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   error = cudaFree(d_B);

   if (error != cudaSuccess) {
    fprintf(stderr, "Failed to free device matrix A (error code %s)!\n",cudaGetErrorString(error));
    exit(EXIT_FAILURE);
   }

   free(A);
   free(B);
   printf("\n");
   printf("done. \n");
   printf("Runtime = %f s.\n", runtime);
   

 return 0;
}

