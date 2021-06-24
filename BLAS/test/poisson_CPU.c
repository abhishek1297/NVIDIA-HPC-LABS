/*
    The following code is solving 2D Discrete poisson equation using Systems of linear equations.
*/

//General Header files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
//BLAS header files
#include <lapacke.h>
/*
N is the total size of the matrix

In Poisson the boundary values are not taken into consideration so the size of the matrix
becomes (N-1) x (N-1).

(N-1) x (N-1) are the dimensions of a sub matrix which forms the main coeff matrix of size (N-1)^2 * (N-1)^2

NRHS is the number of columns on the RHS. Here, it is 1 because RHS is a vector.

LDA is the leading dimension of the coeff array i.e length of each column i.e (N-1)^2

LDB is the leading dimension of the constant array/vector i.e length of each column i.e (N-1)^2

DELTA_N is first derivative wrt N-1 i.e dimension of the sub matrix.
*/
#define N 4
#define NRHS 1
#define LDA ((N-1) * (N-1))
#define LDB ((N-1) * (N-1))
#define DELTA_N (1.0 / (double) (N - 1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

//Coefficient Matrix
double *A;
//Diagonal Matrix used for constructing A
double *diag_block;
//RHS Vector that stores the Constants(in) and the solution(out) as well.
double *B;
//Storing output values
int info, *ipiv;

void allocate() {

    diag_block = (double *) malloc(sizeof(double) * (N-1) * (N-1));
    A = (double *) malloc(sizeof(double) * LDA * LDA);
    B = (double *) malloc(sizeof(double) * NRHS * LDB);
    ipiv = (int *) malloc(sizeof(int) * LDA);
}

void print_matrix(double *x, int n, int m) {

    int i, j;
    for (j=0; j<n; ++j) {
        for (i=0; i<m; ++i) {
             printf("%.5f ", x[IDX2C(i,j,m)]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int valid_index(int i, int j, int n, int m) {
    return IDX2C(i,j,n) < (n * m);
}

void create_coeff_matrix() {

    int i, j, k, l, idx;

/*
    For N = 4

    DIAG =
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


    // set the diagonal sub matrix 
    for (i=0; i<N-1; ++i) {

        diag_block[IDX2C(i,i,N-1)] = -4.0 / (double)(DELTA_N * DELTA_N);

        if (IDX2C(i,i+1,N-1) < (N-1) * (N-1))
            diag_block[IDX2C(i,i+1,N-1)] = 1.0 / (double)(DELTA_N * DELTA_N);

        if (IDX2C(i+1,i,N-1) < (N-1) * (N-1))
            diag_block[IDX2C(i+1,i,N-1)] = 1.0 / (double)(DELTA_N * DELTA_N);
    }

    for (j=0; j<LDA; ++j)
        for (i=0; i<LDA; ++i)
            A[IDX2C(i,j,LDA)] = 0.0;
    
    
    // set the main coeff matrix;
    for (i=0; i<LDA; ++i) {

        k = i % (N-1);
        
        A[IDX2C(i,i,LDA)] = diag_block[IDX2C(k,k,N-1)];

        //Fill Lower diagonal
        if (valid_index(i, i+1, LDA, LDA) && valid_index(k, k+1, N-1, N-1))
            A[IDX2C(i,i+1,LDA)] = diag_block[IDX2C(k,k+1,N-1)];
        
        //Fill Upper diagonal
        if (valid_index(i+1,i,LDA,LDA) && valid_index(k+1, k, N-1, N-1))
            A[IDX2C(i+1,i,LDA)] = diag_block[IDX2C(k+1, k, N-1)];

        //Fill the (N-1)th diagonal from center
        if (i+(N-1) < LDA && valid_index(i+(N-1),i,LDA, LDA))
            A[IDX2C(i+(N-1),i,LDA)] = 1.0 / (double)(DELTA_N * DELTA_N);
        
        //Fill the -(N-1)th diagonal from center
        if (i-(N-1) >= 0 && valid_index(i-(N-1),i,LDA, LDA))
            A[IDX2C(i-(N-1),i,LDA)] = 1.0 / (double)(DELTA_N * DELTA_N);
    }
}

void create_rhs() {

    double a, b;
    int i, j;
    /*
    The solution is a vector but below I am performing a 2D traversal because
    2 different values are required to calculate sin and cos
    */

    for (j=0; j<N-1; ++j) {
        for (i=0; i<N-1; ++i) {

            a = (double) (j + 1) * DELTA_N;
            b = (double) (i + 1) * DELTA_N;
            B[IDX2C(i,j,(N-1))] = sin(a) * cos(b);
        }
    }
}

void solve_linear_equations() {


    // info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, (N-1)*(N-1), (N-1)*(N-1), A, LDA, ipiv);
    // info = LAPACKE_dgetrs(LAPACK_COL_MAJOR,'N', (N-1)*(N-1), NRHS, A, LDA, ipiv, B, LDB);
    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, (N-1) * (N-1), NRHS, A, LDA, ipiv, B, LDB);
    printf("\nINFO = %d\n",info);
    if (info != 0)
        exit(1);
}

void deallocate() {

    free(diag_block);
    free(A);
    free(B);
    free(ipiv);
}

int main() {

    int i, j;
    struct timeval start, end;
    double time_taken;
    gettimeofday(&start, NULL);
    
    allocate();    
    create_coeff_matrix();
    create_rhs();
    solve_linear_equations();
	print_matrix(B, NRHS, LDB);
    deallocate();
    
    gettimeofday(&end, NULL);

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                              start.tv_usec)) * 1e-6;
    printf("\n\nTIME = %.3f sec\n", time_taken);
    return 0;
}


/*
MATLAB ANS:
  -0.029847
  -0.034869
  -0.023245
  -0.050164
  -0.057815
  -0.038467
  -0.048069
  -0.053762
  -0.035686
*/
