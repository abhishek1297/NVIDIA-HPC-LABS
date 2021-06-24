#include <lapacke.h>
#include <stdio.h>
#define N 3
#define NRHS 2
#define LDA N
#define LDB N




int main() {

    int ipiv[N], info;
    int i, j;

    double A[LDA*N] =  { 6.8, -2.11, 5.66,
                         -6.05, -3.30, 5.36,
                         -0.45, 2.58, -2.7};

    double b[LDB*NRHS] =  { 4.02, 6.19, -8.22,
                         -1.56, 4.0, -8.67};

    printf("\nINFO = %d\n\n",LAPACKE_dgesv(LAPACK_COL_MAJOR, N, NRHS, A, LDA, ipiv, b, LDB));


    for (i=0; i<N*NRHS; ++i) {
        printf("%.5f ", b[i]);
    }

    printf("\n");
    return 0;
}