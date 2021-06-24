#include <cblas.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int i=0;
    struct timeval start, end;
    double time_taken;
    int N = 256;
    int th_model;

    if (argc > 1) {
        N = atoi(argv[1]);
        if (argc > 2 && strcmp(argv[2], "m") == 0) {
            printf("\nMulticore on\n\n");
            openblas_set_num_threads(omp_get_max_threads());
        }
    }
    
    double *A = (double *) malloc(sizeof(double) * N * N);
    double *B = (double *) malloc(sizeof(double) * N * N);
    double *C = (double *) malloc(sizeof(double) * N * N);
    
    for (i=0; i<N*N; ++i) {
        A[i] = 3.0;
        B[i] = 10.0;
    }

	printf("\n%d X %d MATRIX\n\n", N, N);
    th_model = openblas_get_parallel();
	
    switch(th_model) {
		case OPENBLAS_SEQUENTIAL:
			printf("OpenBLAS is compiled sequentially.\n");
			break;
		case OPENBLAS_THREAD:
			printf("OpenBLAS is compiled using the normal threading model\n");
			break;
		case OPENBLAS_OPENMP:
			printf("OpenBLAS is compiled using OpenMP\n");
			break;
	}

    gettimeofday(&start, NULL);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,N,N,N,1,A, N, B, N,0,C,N);
    gettimeofday(&end, NULL);

    for(i=0; i<9; i++)
        printf("%lf ", C[i]);
    

    time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - 
                              start.tv_usec)) * 1e-6;
    printf("\n\nTIME = %f\n", time_taken);


    free(A);
    free(B);
    free(C);

    return 0;
}
