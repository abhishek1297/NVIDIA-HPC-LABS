#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>

//Input size
#define SIZE 16384

//2d index to 1d index
#define idx(x,y,z) x*y + z

//TOTAL size of a matrix
size_t TOTAL_SIZE = SIZE * SIZE;

//Device allocated matrices
double *d_mat, *d_matT, *d_matSym;

//Host allocated matrix
double *h_mat;

void printMatrix(double *d_mat, int num) {

    if (SIZE > 16) {
        printf("Too big of an input to be printed!\n");
        return;
    }
    cudaMemcpy(h_mat, d_mat, sizeof(double) * TOTAL_SIZE, cudaMemcpyDeviceToHost);
    int i, j;
    printf("MAT:\n");
    for (i=0; i<SIZE; ++i) {
        for (j=0; j<SIZE; ++j) {
            printf("%f ", h_mat[i * SIZE + j]);
        }
    }
}

void allocate() {

    cudaMalloc((void **)&d_mat, sizeof(double) * TOTAL_SIZE);
    cudaMalloc((void **)&d_matT, sizeof(double) * TOTAL_SIZE);
    cudaMalloc((void **)&d_matSym, sizeof(double) * TOTAL_SIZE);
    h_mat  = (double *) malloc(sizeof(double) * TOTAL_SIZE);
}
__global__ void initialize(double *d_mat) {

    //Global indices
    int tx = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < SIZE && ty < SIZE) {
        //loading the index itself
        d_mat[idx(tx,SIZE,ty)] = idx(tx,SIZE,ty);
    }
}
__global__ void transpose(double *d_mat, double *d_matT) {

    //Global indices
    int tx = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < SIZE && ty < SIZE) {
        //Transposing the matrix
        d_matT[idx(ty,SIZE,tx)] = d_mat[idx(tx,SIZE,ty)];
    }
}

__global__ void matrixMultiply(double *d_mat, double *d_matT, double *d_matSym) {

    //Global inidices
    int tx = blockIdx.y * blockDim.y + threadIdx.y;
    int ty = blockIdx.x * blockDim.x + threadIdx.x;
    int k;

    if (tx < SIZE && ty < SIZE) {
        double accum = 0.0;
        //Accumulation for (tx,ty) position
        for (k=0; k<SIZE; ++k) {
            accum += d_mat[idx(tx,SIZE,k)] * d_matT[idx(k,SIZE,ty)];
        }
        d_matSym[idx(tx,SIZE,ty)] = accum;
    }

}

void calculateSymmetricMatrix(int TILE_DIM) {

    //Configuring the dimensions for thread launch
    dim3 grid_dim(SIZE/TILE_DIM, SIZE/TILE_DIM, 1);
    dim3 blk_dim(TILE_DIM, TILE_DIM, 1);

    //This will generate a symmetric matrix where mat(i,j) = mat(j,i)
    initialize<<<grid_dim, blk_dim>>>(d_mat);
    cudaDeviceSynchronize();
    
    transpose<<<grid_dim, blk_dim>>>(d_mat, d_matT);
    cudaDeviceSynchronize();
    
    matrixMultiply<<<grid_dim, blk_dim>>>(d_mat, d_matT, d_matSym);
    cudaDeviceSynchronize();
}

void deallocate() {

    cudaFree(d_mat);
    cudaFree(d_matT);
    cudaFree(d_matSym);
    free(h_mat);
}

int main (int argc, char **argv) {

    int i, N = 1, TILE_DIM = 32;
    struct timeval start, stop;
    
    double execTime = 0.0;
    
    if (argc > 1) { // Number of iterations
        N = atoi(argv[1]);
    }

    printf("\n%d x %d Matrix\n\n", SIZE, SIZE);
    
    allocate();

    printf("\nExecution times(sec)\n");
    
    for (i=0; i<N; ++i) {
        gettimeofday(&start, NULL);
        
        calculateSymmetricMatrix(TILE_DIM);
        
        gettimeofday(&stop, NULL);
        execTime += (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
        printf("At %d\t%.8f s\n", i, execTime);
    }
    
    deallocate();
    return 0;
}
