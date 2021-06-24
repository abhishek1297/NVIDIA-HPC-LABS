/* 
 *     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <openacc.h>
#include <limits.h>
#include <cuda.h>

void acc_test1() {
    
    int n = (int) 10e6 * 5;      /* size of the vector */
    printf("\nSIZE = %d\n", n);
    float *a;  /* the vector */
    float *restrict r;  /* the results */
    float *e;  /* expected results */
    int i, nerrors;
    nerrors = 0;
    clock_t st, end;
    a = (float*)malloc(n*sizeof(float));
    r = (float*)malloc(n*sizeof(float));
    e = (float*)malloc(n*sizeof(float));
    //   printf("%d", __OPENACC);
    /* initialize */
    for( i = 0; i < n; ++i ) a[i] = (float)(i+1);

    {
    st = clock();
    #pragma acc parallel loop copyin(a[0:n]), create(r[0:n]), copyout(r[0:n])
    for(i=0; i<n; ++i)
        r[i] = a[i]*2.0f;
    end =  clock();
    }
    
    printf("\nGPU = %.8f\n", ((double)end - st)/CLOCKS_PER_SEC);
    st = clock();
    /* compute on the host to compare */
    for(i=0; i<n; ++i)
        e[i] = a[i]*2.0f;
    end = clock();
    printf("\nCPU = %.8f\n", ((double)end - st)/CLOCKS_PER_SEC);
    /* check the results */
    for( i = 0; i < n; ++i ) {
        if ( r[i] != e[i] ) {
           nerrors++;
        }
    }


    printf( "%d iterations completed\n", n );
    if ( nerrors != 0 ) {
        printf( "Test FAILED\n");
    } else {
        printf( "Test PASSED\n");
    }

    free(a); free(r); free(e);
}

int main(int argc, char* argv[]) {

    acc_device_t dev = acc_get_device_type();
    int num = acc_get_num_devices(dev);
    printf("Device %d\n", num);
    acc_test1();

    return 0;
}
