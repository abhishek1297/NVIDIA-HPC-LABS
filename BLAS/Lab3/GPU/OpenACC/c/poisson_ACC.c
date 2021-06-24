/*
    The following code is solving 2D Discrete poisson equation using Systems of linear equations.
*/

//General Header files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

//CUDA Header files
#include <openacc.h>
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
double *A;

//RHS and Solution Vector
double *B;

//Storing output values
int *info, *ipiv;

//Workspace used by cusolver
double *work_buffer;

//Timing Variables
struct timeval start, end;
double time_taken, total_time;

void allocate() {

    /*
        A: Coefficient Matrix
        B: RHS Vector as well as Solution vector
        ipiv: Stores the pivot indices that define permutation matrix
        info: Value returned by the linear solver function

        NOTE: Host pointers are not needed
        I have kept them for printing the matrices for verification.
    */

    A = (double *) malloc(sizeof(double) * LDA * LDA);
    B = (double *) malloc(sizeof(double) * NRHS * LDB);
    ipiv = (int *) malloc(sizeof(int) * LDA);
    info = (int *) malloc(sizeof(int));
}

void print_matrix(double *x, int n, int m) {

    /*
        Prints the Matrix
        Columns are printed horizontally
    */

    int i, j;
    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
             printf("%.3f ", x[IDX2C(i,j,m)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int valid_index(int i, int j, int n, int m) {
    
    /*
        Checking if the index is within range
    */
    return IDX2C(i,j,n) >= 0 && IDX2C(i,j,n) < (n * m);
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

    int i, j, k, index_diff;

    printf("\nInitializing Coefficient Matrix...\n");

    gettimeofday(&start, NULL);
    
    //Set the main coeff matrix;
    #pragma acc parallel loop collapse(2) pcreate(A[LDA*LDA])
    for (j=0; j<LDA; ++j) {
        for (i=0; i<LDA; ++i) {
        
            index_diff = abs(i - j);
            if (i == j) {

                k = i % (N-1);
                
                //Fill the Main Diagonal
                A[IDX2C(i,i,LDA)] = -4.0 / (double)(DEL_N * DEL_N);

                //Fill the Lower Diagonal
                if (valid_index(i, i+1, LDA, LDA))
                    A[IDX2C(i,i+1,LDA)] = (k != (N-2)) ? 1.0 / (double)(DEL_N * DEL_N): 0.0;
                
                //Fill the Upper Diagonal
                if (valid_index(i+1,i,LDA,LDA))
                    A[IDX2C(i+1,i,LDA)] = (k != (N-2)) ? 1.0 / (double)(DEL_N * DEL_N): 0.0;

                //Fill the (N-1)th Diagonal from the center
                if (i+(N-1) < LDA && valid_index(i+(N-1),i,LDA, LDA))
                    A[IDX2C(i+(N-1),i,LDA)] = 1.0 / (double)(DEL_N * DEL_N);
                
                //Fill the -(N-1)th Diagonal from the center
                if (i-(N-1) >= 0 && valid_index(i-(N-1),i,LDA, LDA))
                    A[IDX2C(i-(N-1),i,LDA)] = 1.0 / (double)(DEL_N * DEL_N);
            }
            else
            if (index_diff != 1 && index_diff != N-1) {
                /*
                    When i == j, All 5 diagonals are getting filled
                    so we don't need to fill them again at another iteration
                    To avoid this, I am using index_diff if it is either 1 or N-1
                    Then do not fill those indices with 0 because those values are filled at i == j
                */
                A[IDX2C(i,j,LDA)] = 0.0;
            }
        }
    }

    gettimeofday(&end, NULL);

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                              start.tv_usec)) * 1e-6;
    printf("\nCOEFF TIME = %f sec\n", time_taken);
    total_time += time_taken;
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

        These values are not exactly application specific
        so creating a vector of sinusoidal values.
    */

    double a, b;
    int i, j;

    printf("\nInitializing RHS Vector...\n");
   
    gettimeofday(&start, NULL);
   
    #pragma acc parallel loop collapse(2) pcreate(B[NRHS*LDB])
    for (j=0; j<N-1; ++j) {
        for (i=0; i<N-1; ++i) {

            a = (double) (j + 1) * DEL_N;
            b = (double) (i + 1) * DEL_N;
            B[IDX2C(i,j,(N-1))] = sin(a) * cos(b);
        }
    }

    gettimeofday(&end, NULL);

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                              start.tv_usec)) * 1e-6;
    printf("\nRHS TIME = %f sec\n", time_taken);
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
    
    printf("\nSolving Linear Equations.");    

    #pragma acc host_data use_device(A)
    {
        //The following function calculates buffer space
        //required for LU factorization of the given matrix
        status = cusolverDnDgetrf_bufferSize(handle,
                                    (N-1) * (N-1),
                                    (N-1) * (N-1),
                                    A,
                                    LDA,
                                    &workspace_length);
        check(status);
    }

    //Allocate the space for the buffer
    workspace_bytes = workspace_length * sizeof(double);
    work_buffer = (double *) acc_malloc(workspace_bytes);
     
    gettimeofday(&start, NULL);
    
    #pragma acc data present(A[LDA*LDA], B[NRHS*LDB], ipiv[LDA], info) 
    {
        #pragma acc host_data use_device(A, B, info, ipiv)
        {
            //Performs LU factorization of Matrix A
            //P * A = L * U
            //P is a permutation matrix, L is a lower triangular matrix with unit diagonal,
            //and U is an upper triangular matrix.
            status = cusolverDnDgetrf(handle, 
                                    (N-1) * (N-1),
                                    (N-1) * (N-1), 
                                    A, 
                                    LDA, 
                                    work_buffer,
                                    ipiv, 
                                    info);
            check(status);

            //Solves the system of linear equations.
            status = cusolverDnDgetrs(handle,
                                    CUBLAS_OP_N,
                                    (N-1) * (N-1),
                                    NRHS,
                                    A,
                                    LDA,
                                    ipiv,
                                    B,
                                    LDB,
                                    info);
        }
        check(status);
    }

    gettimeofday(&end, NULL);
    
    //Releasing the resources
    acc_free(work_buffer);
    cusolverDnDestroy(handle);
    cusolverDnDestroyParams(params);

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                              start.tv_usec)) * 1e-6;
    printf("\nLS TIME = %f sec", time_taken);
    total_time += time_taken;
}

void print_sum() {

    int i;
    double sum = 0.0;

    for (i=0; i<NRHS*LDB; ++i)
        sum += B[i];

    printf("\n\nB SUM = %e", sum);
}

void deallocate() {

    free(A);
    free(B);
    free(ipiv);
    free(info);
}

int main() {

    int i, j;
    
    allocate();

    #pragma acc data create(A[LDA*LDA], info[1], ipiv[LDA]) copyout(B[NRHS*LDB])
    {
        create_coeff_matrix();

        create_rhs();
        
        // for (i=0; i<10; ++i)
            solve_linear_equations();
    }

    print_sum();

    deallocate();

    printf("\n\nTOTAL TIME = %f sec\n",total_time);    
    return 0;
}
