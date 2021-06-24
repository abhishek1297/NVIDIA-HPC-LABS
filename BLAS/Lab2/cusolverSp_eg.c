#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
//cusolver for sparse matrices
#include <cusolverSp.h>

#define N 5
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

//Coefficient non-zero Vector
double *csrValA;
//index and offset vectors
int *csrRowPtrA, *csrColIdxA, *csrEndPtrA;
//Diagonal Matrix used for constructing A
double *diag_block;
//RHS Vector that stores the Constants
double *B;
//Solution vector
double *X;
//Number of non-zero values in A
int A_NNZ;


static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_SUCCESS";

    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";

    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";

    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";

    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";

    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";

    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";

    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

void print_matrix(double *x, int n, int m) {

    int i, j;
    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
             printf("%.3f ", x[IDX2C(i,j,m)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

void allocate() {

    //Number of Non zeros values in the matrix.
    A_NNZ = (int) (LDA + 4 * (LDA - sqrt(LDA)));
    csrValA = (double *) malloc(sizeof(double) * A_NNZ);
    csrRowPtrA = (int *) malloc(sizeof(int) * (LDA + 1));
    csrEndPtrA = (int *) malloc(sizeof(int) * (LDA + 1));
    csrColIdxA = (int *) malloc(sizeof(int) * A_NNZ);
    B = (double *) malloc(sizeof(double) * NRHS * LDB);
    X = (double *) malloc(sizeof(double) * NRHS * LDB);
}

void create_coeff_matrix() {


    /*
    To fill the CSR attributes
    Iterate over the matrix row by row and to fill the data 
    of non zero vectors and their position information.
    */

    int i, j, k;
    int cnt = 0;

    for (i=0; i<LDA; ++i) {
        csrRowPtrA[i] = cnt;
        for (j=0; j<LDA; ++j) {
    
            if (i == j) {
    
                k = i % (N-1);
                if (j-(N-1) >= 0) {
                    csrValA[cnt] = 1.0 / (double)(DEL_N * DEL_N);
                    csrColIdxA[cnt] = j - (N-1);
                    ++cnt;
                }
                if (j-1 >= 0 && k != 0) {
                    csrValA[cnt] = 1.0 / (double)(DEL_N * DEL_N);
                    csrColIdxA[cnt] = j - 1;
                    ++cnt;
                }
                
                csrValA[cnt] = -4.0 / (double)(DEL_N * DEL_N);
                csrColIdxA[cnt] = j;
                ++cnt;

                if (j+1 < LDA && k != (N-2)) {
                    csrValA[cnt] = 1.0 / (double)(DEL_N * DEL_N);
                    csrColIdxA[cnt] = j + 1;
                    ++cnt;
                }

                if (j+(N-1) < LDA) {
                    csrValA[cnt] = 1.0 / (double)(DEL_N * DEL_N);
                    csrColIdxA[cnt] = j + (N-1);
                    ++cnt;
                }
            }
            csrEndPtrA[i] = cnt - 1;
        }
    }
    csrRowPtrA[i] = cnt;
    csrEndPtrA[i] = cnt + 1;

    printf("\nCNT = %d\n", cnt);
}

void create_rhs() {

    double a, b;
    int i, j;
    /*
    The solution is a vector but below I am performing a 2D traversal because
    2 different values are required to calculate sin and cos
    Note that, NRHS * LDB == (N-1) * (N-1)
    */

    for (j=0; j<N-1; ++j) {
        for (i=0; i<N-1; ++i) {

            a = (double) (j + 1) * DEL_N;
            b = (double) (i + 1) * DEL_N;
            B[IDX2C(i,j,(N-1))] = sin(a) * cos(b);
        }
    }
}

void solve_linear_equations() {

    cusolverStatus_t status;
    cusolverSpHandle_t handle = NULL;
    cusparseMatDescr_t descrA;
    float tol = 0.0;
    int reorder = 0;
    int singularity;
    int issym;


    cusolverSpCreate(&handle);

    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);

    status = cusolverSpXcsrissymHost(handle,
                        LDA,
                        A_NNZ,
                        descrA,
                        csrRowPtrA,
                        csrEndPtrA,
                        csrColIdxA,
                        &issym);

    printf("\nISSYM = %s\n", issym?"YES":"NO");
    
    /*
    status = cusolverSpDcsrlsvluHost(handle,
                            LDA,
                            A_NNZ,
                            descrA,
                            csrValA,
                            csrRowPtrA,
                            csrColIdxA,
                            B,
                            tol,
                            reorder,
                            X,
                            &singularity);
    */

    status = cusolverSpDcsrlsvqrHost(handle,
                            LDA,
                            A_NNZ,
                            descrA,
                            csrValA,
                            csrRowPtrA,
                            csrColIdxA,
                            B,
                            tol,
                            reorder,
                            X,
                            &singularity);
    
    printf("\n\n%s\n", _cusolverGetErrorEnum(status));
    check(status);
    cusparseDestroyMatDescr(descrA);
    cusolverDnDestroy(handle);
}

void deallocate() {

    free(csrValA);
    free(csrRowPtrA);
    free(csrEndPtrA);
    free(csrColIdxA);
    free(X);
    free(B);
}
int main() {

    int i, j;
    allocate();

    create_coeff_matrix();
    create_rhs();
    // for (i=0; i<LDA; ++i) {
        
    //     for (j=csrRowPtrA[i]; j<csrRowPtrA[i+1]; ++j) {
    //         printf("%.1f ",csrValA[j]);
    //     }
    //     printf("\n");
    // }
    
    solve_linear_equations();
    // print_matrix(X, NRHS, LDB);
    deallocate();

    return 0;
}

