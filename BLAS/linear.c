#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define N 3
#define NRHS 2
#define LDA N
#define LDB N

double A[LDA*N] =  { 6.8, -2.11, 5.66,
                        -6.05, -3.30, 5.36,
                        -0.45, 2.58, -2.7};
double A_ref[LDA*N];
double b[LDB*NRHS] =  { 4.02, 6.19, -8.22,
                        -1.56, 4.0, -8.67};
double b_ref[LDB*NRHS];

int ipiv[N], info;

double *d_A, *d_b;
int *d_ipiv, *d_info;
void *work_buffer;


void print_matrix() {
        int i, j;
    /*
        Actual matrix not in memory
         6.80    -6.05   -0.45
        -2.11    -3.30    2.58
         5.66     5.36   -2.70
        
        4.02     -1.56
        6.19      4.00 
       -8.22     -8.67
    */
    for (j=0; j<N; ++j) {
        for (i=0; i<N; ++i) {
            printf("%.2f ", A[IDX2C(i,j,LDA)]);
        }
    }
    printf("\n\n");
    for (j=0; j<NRHS; ++j) {
        for (i=0; i<N; ++i) {
            printf("%.2f ", b[IDX2C(i,j,LDB)]);
        }
    }
    printf("\n\n");
}

void do_cpu() {

    if ((info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A, LDA, ipiv, b, LDB)) != 0) {
        printf("\nLAPACKE Error %d\n\n", info);
        exit(1);
    }
}

void do_gpu() {

    int workspace_bytes_device, workspace_bytes_host;
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t params;
    cusolverStatus_t status;

    status = cusolverDnCreate(&cusolverH);
    
    status = cusolverDnCreateParams(&params);
    
    status = cusolverDnDgetrf_bufferSize(cusolverH,
                                N, N,
                                d_A,
                                LDA,
                                &workspace_bytes_device);

    cudaMalloc((void **)&work_buffer, workspace_bytes_device);
    
    status = cusolverDnDgetrf(cusolverH, 
                            N, N, 
                            d_A, 
                            LDA, 
                            (double *) work_buffer, 
                            d_ipiv, 
                            d_info);
    
    printf("\nWORKSPACE = %d %d\n", workspace_bytes_device, workspace_bytes_host);
    printf("\nSgetrf = %d", status);

    status = cusolverDnDgetrs(cusolverH,
                            CUBLAS_OP_N,
                            N,
                            NRHS,
                            d_A,
                            LDA,
                            d_ipiv,
                            d_b,
                            LDB,
                            d_info);

    printf("\nSgetrs = %d", status);
    cudaFree(work_buffer);
    cusolverDnDestroy(cusolverH);
    cusolverDnDestroyParams(params);
}

void l2Norm() {

    double sum_sq = 0.0;
    double diff;
    double sum_ref = 0.0;
    int i;
    for (i=0; i<N*NRHS; ++i) {
       diff = b_ref[i] - b[i];
       sum_ref += b_ref[i] * b_ref[i];
       sum_sq += diff * diff;
    }

    printf("\n\nL2 NORM: %.10f", sum_sq / sum_ref);
}


int main() {

    


    cudaMalloc((void **) &d_A, sizeof(A));
    cudaMalloc((void **) &d_b, sizeof(b));
    cudaMalloc((void **)&d_ipiv, sizeof(ipiv));
    cudaMalloc((void **)&d_info, sizeof(info));

    cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);


    do_cpu();
    do_gpu();

    cudaMemcpy(b_ref, d_b, sizeof(b), cudaMemcpyDeviceToHost);
    l2Norm();


    printf("\n");
    
    
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    return 0;
}