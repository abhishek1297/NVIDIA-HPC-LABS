#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TILE_DIM 32

//Input size
#define SIZE 16384

//2d index to 1d index
#define idx(x,y,z) x*y + z

//TOTAL size of a matrix
size_t TOTAL_SIZE = SIZE * SIZE;

//Host and Device allocated matrices
double *restrict mat;//Original
double *restrict matT;//Transposed
double *restrict matSym;//Symmetric

void printMatrix(double *restrict M) {

    int i, j;
    if (SIZE > 16) {
        printf("Too big of an input to be printed!\n");
        return;
    }
    printf("MAT:\n");
    for (i=0; i<SIZE; ++i) {
        for (j=0; j<SIZE; ++j) {
            printf("%.2f ", M[idx(i,SIZE,j)]);
        }
        printf("\n");
    }
}

void allocate() {

    mat = (double *) malloc(sizeof(double) * TOTAL_SIZE);
    matT = (double *) malloc(sizeof(double) * TOTAL_SIZE);
    matSym = (double *) malloc(sizeof(double) * TOTAL_SIZE);
}

void initialize() {

    int i, j;
    //Parallelizing the following loop by distributing the workload
    //among TILE_DIM x TILE_DIM threads for better locality.
    //Data is already present on the device.
    #pragma acc parallel loop tile(TILE_DIM,TILE_DIM) present(mat[:TOTAL_SIZE])
    for (i=0; i<SIZE; ++i) {
        for (j=0; j<SIZE; ++j) {
            //loading the index itself
            mat[idx(i,SIZE,j)] = idx(i,SIZE,j);
        }
    }
}

void transpose() {

    int i, j;
    //Parallelizing the following loop by distributing the workload
    //among TILE_DIM x TILE_DIM threads for better locality.
    //Data is already present on the device.
    #pragma acc parallel loop tile(TILE_DIM,TILE_DIM) present(mat[:TOTAL_SIZE], matT[:TOTAL_SIZE])
    for (i=0; i<SIZE; ++i) {
        for (j=0; j<SIZE; ++j) {
            matT[idx(j,SIZE,i)] = mat[idx(i,SIZE,j)];
        }
    }
}

void matrixMultiply() {

    int i, j, k;
    double accum;
    //Parallelizing the following loop by distributing the workload
    //among TILE_DIM x TILE_DIM threads for better locality.
    //Data is already present on the device.
    #pragma acc parallel loop tile(TILE_DIM,TILE_DIM) present(mat[:TOTAL_SIZE], matT[:TOTAL_SIZE], matSym[:TOTAL_SIZE])
    for (i=0; i<SIZE; ++i) {
        for (j=0; j<SIZE; ++j) {
            accum = 0.0;
            for (k=0; k<SIZE; ++k) {
                accum += mat[idx(i,SIZE,k)] * matT[idx(k,SIZE,j)];
            }
            matSym[idx(i,SIZE,j)] = accum;
        }
    }
}

void calculateSymmetricMatrix() {

    //This will generate a symmetric matrix where mat(i,j) = mat(j,i)
    transpose();
    matrixMultiply();

}

void deallocate() {

    free(mat);
    free(matT);
    free(matSym);
}

int main(int argc, char **argv) {

    int i, N = 1;
    struct timeval start, stop;
    double execTime = 0.0;
    if (argc > 1) { // Number of iterations
        N = atoi(argv[1]);
    }
    
    allocate();
     
    printf("\nExecution times(sec)\n");
    //Allocate memory on the GPU prior performing any operations.
    //Note that instead of copyout you can use create if you don't want to print the data.
    #pragma acc data copyout(mat[:TOTAL_SIZE], matT[:TOTAL_SIZE], matSym[:TOTAL_SIZE])
    {
        for (i=0; i<N; ++i) {
            gettimeofday(&start, NULL);
            
            initialize();
            
            calculateSymmetricMatrix();
            
            gettimeofday(&stop, NULL);
            execTime += (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
            printf("At %d\t%.8f s\n", i, execTime);
        }
    }
    
    deallocate();
    return 0;
}
