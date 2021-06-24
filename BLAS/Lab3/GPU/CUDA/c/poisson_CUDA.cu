/*
    The following code is solving 2D Discrete poisson equation using Systems of linear equations.
*/

//General Header files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//CUDA Header files
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define N 100
#define NRHS 1
#define LDA ((N-1) * (N-1))
#define LDB ((N-1) * (N-1))
#define DEL_N (1.0 / (double) (N - 1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define check(stat) \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        printf ("\nError: %d at %d\n", stat, __LINE__); \
        exit(1);\
    }

/*

https://en.wikipedia.org/wiki/Discrete_Poisson_equation

In Poisson's equation, Unknowns at the boundaries are set to 0. So we don't need to calculate those variables.

Size of sub-matrix D without boundaries is N-1 x N-1.

The coefficient Matrix A, is a pentadiagonal matrix (5 diagonals) that is constructed using
Matrix D. Size of A is (N-1)^2 x (N-1)^2.

LDA is the leading dimension of A (N-1)^2.

NRHS is the number of columns on the RHS. Here, it is 1 because RHS is a vector.

LDB is the leading dimension of RHS (N-1)^2.

DEL_N is first derivative wrt (N-1). Simply, it is the distance between two grid points of Matrix D.

IDX2C is a conversion of 2D column-major index to a 1D index.

*/

//Coefficient Matrix
double *d_A, *A;

//RHS and Solution Vector
double *d_B, *B;

//Storing output values
int *d_info, *info, *d_ipiv, *ipiv;

//Workspace used by cusolver
double *d_work_buffer;

//Timing Variables
cudaEvent_t start, end;
float time_taken, total_time;

void allocate() {

    /*
        A: Coefficient Matrix
        B: RHS Vector as well as Solution vector
        ipiv: Stores the pivot indices that define permutation matrix
        info: Value returned by the linear solver function

        NOTE: Host pointers are not needed
        I have kept them for printing the matrices for verification.
    */

    //Host pointers
    A = (double *) malloc(sizeof(double) * LDA * LDA);
    B = (double *) malloc(sizeof(double) * NRHS * LDB);
    ipiv = (int *) malloc(sizeof(int) * LDA);
    info = (int *) malloc(sizeof(int));
    
    //Device Pointers
    cudaMalloc((void **) &d_A, sizeof(double) * LDA * LDA);
    cudaMalloc((void **) &d_B, sizeof(double) * NRHS * LDB);
    cudaMalloc((void **) &d_ipiv, sizeof(int) * LDA);
    cudaMalloc((void **) &d_info, sizeof(int));

    //Event creation for time variables
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void print_matrix(double *d_x, double *h_x, int n, int m) {

    /*
        Prints the Matrix
        Columns are printed horizontally
    */

    int i, j;

    if (d_x != NULL)
        cudaMemcpy(h_x, d_x, sizeof(double) * n * m, cudaMemcpyDeviceToHost);

    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
             printf("%.3f ", h_x[IDX2C(i,j,m)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

extern "C" __device__ __host__ int valid_index(int i, int j, int n, int m) {

    /*
        Checking if the index is within range
    */
    return IDX2C(i,j,n) >= 0 && IDX2C(i,j,n) < (n * m);
}

extern "C" __global__ void initialize_coeffs(double *d_A) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < LDA && j < LDA) {

        int k, index_diff;

        index_diff = abs(i - j);
        if (i == j) {

            k = i % (N-1);
            
            
            //Fill the Main Diagonal
            d_A[IDX2C(i,i,LDA)] = -4.0 / (double)(DEL_N * DEL_N);

            //Fill the Lower Diagonal
            if (valid_index(i, i+1, LDA, LDA))
                d_A[IDX2C(i,i+1,LDA)] = (k != (N-2)) ? 1.0 / (double)(DEL_N * DEL_N): 0.0;
            
            //Fill the Upper Diagonal
            if (valid_index(i+1,i,LDA,LDA))
                d_A[IDX2C(i+1,i,LDA)] = (k != (N-2)) ? 1.0 / (double)(DEL_N * DEL_N): 0.0;

            //Fill the (N-1)th Diagonal from the center
            if (i+(N-1) < LDA && valid_index(i+(N-1),i,LDA, LDA))
                d_A[IDX2C(i+(N-1),i,LDA)] = 1.0 / (double)(DEL_N * DEL_N);
            
            //Fill the -(N-1)th Diagonal from the center
            if (i-(N-1) >= 0 && valid_index(i-(N-1),i,LDA, LDA))
                d_A[IDX2C(i-(N-1),i,LDA)] = 1.0 / (double)(DEL_N * DEL_N);
        }
        else
        if (index_diff != 1 && index_diff != N-1) {
            /*
                When i == j, All 5 diagonals are getting filled
                so we don't need to fill them again by another thread
                To avoid this, I am using index_diff if it is either 1 or N-1
                Then do not fill those indices with 0 because those values are filled at i == j
            */            
            d_A[IDX2C(i,j,LDA)] = 0.0;
        }
    }
}

void create_coeff_matrix() {

    /*

            |D      |
        A = |   D   |
            |      D|

        For N = 4, we get,

        D =
        -36.00 9.00 0.00 
        9.00 -36.00 9.00 
        0.00 9.00 -36.00 
        
        A =
        -36.00 9.00 0.00 9.00 0.00 0.00 0.00 0.00 0.00 
        9.00 -36.00 9.00 0.00 9.00 0.00 0.00 0.00 0.00 
        0.00 9.00 -36.00 0.00 0.00 9.00 0.00 0.00 0.00 
        9.00 0.00 0.00 -36.00 9.00 0.00 9.00 0.00 0.00 
        0.00 9.00 0.00 9.00 -36.00 9.00 0.00 9.00 0.00 
        0.00 0.00 9.00 0.00 9.00 -36.00 0.00 0.00 9.00 
        0.00 0.00 0.00 9.00 0.00 0.00 -36.00 9.00 0.00 
        0.00 0.00 0.00 0.00 9.00 0.00 9.00 -36.00 9.00 
        0.00 0.00 0.00 0.00 0.00 9.00 0.00 9.00 -36.00
    */

    int DIM = ceil(LDA/(double) 32);
    
    //Creating blocks of size 32 x 32
    dim3 BLK_DIM(32, 32, 1), GRID_DIM(DIM, DIM, 1);

    printf("\nIntializing Coeffcient Matrix...\n");

    cudaEventRecord(start);
    
    //Initialize the coefficient Matrix A
    initialize_coeffs<<<GRID_DIM, BLK_DIM>>>(d_A);
    cudaDeviceSynchronize();
    
    cudaEventRecord(end);

    cudaEventElapsedTime(&time_taken, start, end);
    printf("\nCOEFF TIME = %f sec\n", time_taken/1000);
    total_time += time_taken;
}

extern "C" __global__ void initialize_rhs(double *d_B) {

    /*
        These values are not exactly application specific
        so creating a vector of sinusoidal values.
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N-1 && j < N-1) {

        double a = (double) (j + 1) * DEL_N;
        double b = (double) (i + 1) * DEL_N;
        d_B[IDX2C(i,j,(N-1))] = sin(a) * cos(b);
    }
}


void create_rhs() {

    /*
        Let b be a vector,

            |v00    |
            |v10    |
        b = |.      |
            |.      |
            |v(N-1)0|
        
        Then,

            |b1     |
            |b2     |
        B = |.      |
            |.      |
            |b(N-1) |

        Therefore RHS B vector is of size (N-1) * (N-1)
    */

    int DIM = ceil((N-1)/(double) 32);

    //Creating blocks of size 32 x 32
    dim3 BLK_DIM(32, 32, 1), GRID_DIM(DIM, DIM, 1);

    printf("\nInitializing RHS Vector...\n");

    cudaEventRecord(start);    

    initialize_rhs<<<GRID_DIM, BLK_DIM>>>(d_B);
    cudaDeviceSynchronize();

    cudaEventRecord(end);

    cudaEventElapsedTime(&time_taken, start, end);
    printf("\nRHS TIME = %f sec\n",time_taken/1000);
    total_time += time_taken;
}

void solve_linear_equations() {

    /*
        cuSOLVER provides functions to solve system of linear equations.
    */
    int workspace_bytes, workspace_length;

    cusolverDnHandle_t handle = NULL;
    cusolverDnParams_t params;
    cusolverStatus_t status;

    //Creating cuSOLVER handle
    status = cusolverDnCreate(&handle);
    check(status);
    
    //Initializing a structure used by cuSOLVER
    status = cusolverDnCreateParams(&params);
    check(status);

    //The following function calculates buffer space
    //required for LU factorization of the given matrix
    status = cusolverDnDgetrf_bufferSize(handle,
                                (N-1) * (N-1),
                                (N-1) * (N-1),
                                d_A,
                                LDA,
                                &workspace_length);
    check(status);

    //Allocate the space for the buffer
    workspace_bytes = workspace_length * sizeof(double);   
    cudaMalloc((void **)&d_work_buffer, workspace_bytes);

    printf("\nSolving Linear Equations.");
    
    cudaEventRecord(start);

    //Performs LU factorization of Matrix A
    //P * A = L * U
    //P is a permutation matrix, L is a lower triangular matrix with unit diagonal,
    //and U is an upper triangular matrix.
    status = cusolverDnDgetrf(handle, 
                            (N-1) * (N-1),
                            (N-1) * (N-1), 
                            d_A, 
                            LDA, 
                            d_work_buffer,
                            d_ipiv, 
                            d_info);
    check(status);

    //Solves the system of linear equations.
    status = cusolverDnDgetrs(handle,
                            CUBLAS_OP_N,
                            (N-1) * (N-1),
                            NRHS,
                            d_A,
                            LDA,
                            d_ipiv,
                            d_B,
                            LDB,
                            d_info);

    cudaDeviceSynchronize();
    check(status);
    
    cudaEventRecord(end);

    //Releasing the resources
    cudaFree(d_work_buffer);
    cusolverDnDestroy(handle);
    cusolverDnDestroyParams(params);

    cudaEventElapsedTime(&time_taken, start, end);
    printf("\nLS TIME = %f sec", time_taken/1000);
    total_time += time_taken;
}


void print_sum() {

    int i;
    double sum = 0.0;
    
    cudaMemcpy(B, d_B, sizeof(double) * NRHS * LDB, cudaMemcpyDeviceToHost);

    for (i=0; i<NRHS*LDB; ++i)
        sum += B[i];

    printf("\n\nB SUM = %e", sum);
}

void deallocate() {

    free(A);
    free(B);
    free(ipiv);
    free(info);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_ipiv);
    cudaFree(d_info);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

int main() {

    int i, j;
    
    allocate();
    
    create_coeff_matrix();

    create_rhs();

    // for (i=0; i<10; ++i)
        solve_linear_equations();
    
    print_sum();

    deallocate();
    
    printf("\n\nTOTAL TIME = %f sec\n", total_time/1000);
    return 0;
}
