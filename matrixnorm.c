/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Program Parameters */
int N = 6000; /* Matrix size */

/* Matrices */
volatile float A[10000][10000], B[10000][10000];


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    //srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }
    
}


/* Kernel function */

void matrixNorm() {
    int row, col;
    float mu, sigma; // Mean and Standard Deviation
    
    printf("Computing Serially.\n");
    
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += A[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(A[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                B[row][col] = 0.0;
            else
                B[row][col] = (A[row][col] - mu) / sigma;
        }
    }
    
}

void print_inputs(){
int row,col;
if(N<20){
   for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        printf("%f ",A[row][col]);
     }
     printf("\n");
   }
}
}

void print_output(){
int row,col;
if(N<20){
   for (row = 0; row < N ; row++)
   {
     for(col = 0; col < N ; col++)
     {
        printf("%f ",B[row][col]);
     }
     printf("\n");
   }
}
}



int main(int argc, char **argv) {
    /* Timing variables */
    clock_t start;
   clock_t stop;  /* Elapsed times using gettimeofday() */
   double runtime;
    
    N =atoi(argv[1]);
    int seed = atoi(argv[2]);
    srand(seed);
    /* Initialize A and B */
    initialize_inputs();
    
    printf("input matrix\n");
    print_inputs();
    
    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    start = clock();
    
    
    /* Matrix Normalization */
    matrixNorm();
    printf("normalized matrix\n");
    print_output();
    /* Stop Clock */
    stop = clock();
    runtime = (double)(stop - start)/CLOCKS_PER_SEC;
    
    
    /* Display timing results */
     printf("Runtime = %f s\n", runtime);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}