#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>
#define TOTAL_SIZE 1024
//#define TOTAL_SIZE (1024*1024*1024)
#define block_dim 1024
#define chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double *h_c, *h_a, *h_b;

double **d_c, **d_a, **d_b;

cudaStream_t *streams;

cudaEvent_t start, finish;


void allocate(int devices, int multi_gpu) {

    int i = 0, parts, rem;

    //h_c = (double *) malloc(sizeof(double) * TOTAL_SIZE);
    //h_a = (double *) malloc(sizeof(double) * TOTAL_SIZE);
    //h_b = (double *) malloc(sizeof(double) * TOTAL_SIZE);

    d_c = (double **) malloc(sizeof(double *) * devices);
    d_a = (double **) malloc(sizeof(double *) * devices);
    d_b = (double **) malloc(sizeof(double *) * devices);

    cudaMallocHost((void **) &h_c, sizeof(double) * TOTAL_SIZE);
    cudaMallocHost((void **) &h_a, sizeof(double) * TOTAL_SIZE);
    cudaMallocHost((void **) &h_b, sizeof(double) * TOTAL_SIZE);
    
    //cudaMallocHost((void **)d_c, sizeof(double *) * devices);
    //cudaMallocHost((void **)d_a, sizeof(double *) * devices);
    //cudaMallocHost((void **)d_b, sizeof(double *) * devices);

    streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * devices);
    
    for (i=0; i<devices; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    parts = TOTAL_SIZE / devices;
    rem = TOTAL_SIZE % devices;
    i = 0;

    if (multi_gpu) {
        for (i=0; i<devices-1; ++i) {
    
            cudaSetDevice(i);
            printf("\nS%d", streams[i]);
            chk(cudaMallocAsync((void **) &d_c[i], sizeof(double) * parts, streams[i]));
            chk(cudaMallocAsync((void **) &d_a[i], sizeof(double) * parts, streams[i]));
            chk(cudaMallocAsync((void **) &d_b[i], sizeof(double) * parts, streams[i]));
        }
    }

    cudaSetDevice(i);
    chk(cudaMallocAsync((void **) &d_c[i], sizeof(double) * (parts + rem), streams[i]));
    chk(cudaMallocAsync((void **) &d_a[i], sizeof(double) * (parts + rem), streams[i]));
    chk(cudaMallocAsync((void **) &d_b[i], sizeof(double) * (parts + rem), streams[i]));

    cudaEventCreate(&start);
    cudaEventCreate(&finish);            
}

extern "C" __global__ void vec_add(double *c, double *a, double *b, int PART_SIZE) {
    
    int t = threadIdx.x + blockIdx.x * blockDim.x;

    if (t < TOTAL_SIZE && t < PART_SIZE) {
    
        c[t] = a[t] + b[t];

        if (t % 100)
            printf("\n%f", c[t]);
    }

}

void kernels_launch(int devices, int multi_gpu) {

    int parts = TOTAL_SIZE / devices;
    int rem = TOTAL_SIZE % devices;
    int i = 0;

    if (multi_gpu) {
        for (i=0; i<devices-1; ++i) {
    
            cudaSetDevice(i);
            vec_add<<<parts/block_dim + 1, block_dim, 0, streams[i]>>>(d_c[i], d_a[i], d_b[i], parts);    
        }
    }

    cudaSetDevice(i);
    vec_add<<<(parts + rem)/block_dim + 1, block_dim, 0, streams[i]>>>(d_c[i], d_a[i], d_b[i], parts + rem);    

}


void data_transferHtoD(int devices, int multi_gpu) {

    
    int parts = TOTAL_SIZE / devices;
    int rem = TOTAL_SIZE % devices;
    int i = 0;

    if (multi_gpu) {
        for (i=0; i<devices-1; ++i) {
        
            cudaSetDevice(i);
            printf("\nS%d", streams[i]);
            chk(cudaMemcpyAsync(d_a[i], h_a + (parts * i), sizeof(double) * parts, cudaMemcpyHostToDevice, streams[i]));
            chk(cudaMemcpyAsync(d_b[i], h_b + (parts * i), sizeof(double) * parts, cudaMemcpyHostToDevice, streams[i]));
        }
    }

    cudaSetDevice(i);
    chk(cudaMemcpyAsync(d_a[i], h_a + (parts * i), sizeof(double) * (parts + rem), cudaMemcpyHostToDevice, streams[i]));
    chk(cudaMemcpyAsync(d_b[i], h_b + (parts * i), sizeof(double) * (parts + rem), cudaMemcpyHostToDevice, streams[i]));

}


void data_transferDtoH(int devices, int multi_gpu) {

    int parts = TOTAL_SIZE / devices;
    int rem = TOTAL_SIZE % devices;
    int i = 0;

    if (multi_gpu) {
        //Data trnsfer back
        for (i=0; i<devices-1; ++i) {
        
            cudaSetDevice(i);
            chk(cudaMemcpyAsync(h_c + (parts * i), d_c[i], sizeof(double) * parts, cudaMemcpyDeviceToHost, streams[i]));
        
        }
        cudaSetDevice(i);
        chk(cudaMemcpyAsync(h_c + (parts * i), d_c[i], sizeof(double) * (parts + rem), cudaMemcpyDeviceToHost, streams[i]));
    }
}

void deallocate(int devices) {

    for (int i=0; i<devices; ++i) {

        cudaSetDevice(i);
        cudaFreeAsync(d_c[i], streams[i]);
        cudaFreeAsync(d_a[i], streams[i]);
        cudaFreeAsync(d_b[i], streams[i]);
    }

    for (int i=0; i<devices; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    free(d_c);
    free(d_a);
    free(d_b);

    //free(h_a);
    //free(h_b);
    //free(h_c);

    //cudaFreeHost(d_c);
    //cudaFreeHost(d_a);
    //cudaFreeHost(d_b);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    cudaEventDestroy(start);
    cudaEventDestroy(finish); 

}

void verify() {

    double diff_sq = 0.0;
    double sum_sq  = 0.0;

    for (int i=0; i<TOTAL_SIZE; ++i) {

        sum_sq += h_c[i] * h_c[i];
        diff_sq += (h_c[i] - (h_a[i] + h_b[i])) * (h_c[i] - (h_a[i] + h_b[i]));
    }

    printf("\n%f\t%f\n", h_c[0], h_c[5]);
    printf("\n\nError Rate: %e\n", diff_sq / sum_sq);
}


int main(int argc, char **argv) {
 
    int i, parts, rem, devices = 1;
    float exec_time;
    int multi_gpu = 0;
    if (argc > 1 && strcmp(argv[1], "-m") == 0) {
       multi_gpu = 1; 
    }

    chk(cudaGetDeviceCount(&devices));
    printf("\nNum devices available = %d\n", devices);

    if (devices == 0) {
        printf("\nError: No devices found\n");
        exit(1);
    }
    
    if (devices ==1)
        multi_gpu = 0;

    allocate(devices, multi_gpu); 


    //Initialize data
    for (i=0; i<TOTAL_SIZE; ++i) {
        h_a[i] = i + 1;
        h_b[i] = i + 2;
    }


    data_transferHtoD(devices, multi_gpu);
    
    cudaEventRecord(start);
    kernels_launch(devices, multi_gpu);
    cudaEventRecord(finish);
    
    for (i=0; i<devices; ++i)
        cudaStreamWaitEvent(streams[i], finish);


    data_transferDtoH(devices, multi_gpu);

    for (i=0; i<devices; ++i)
        cudaStreamSynchronize(streams[i]);

    if (TOTAL_SIZE <= 2048) { 
        verify();
    }

    cudaEventElapsedTime(&exec_time, start, finish);
    
    printf("MultiGPU Time = %f", exec_time / 1000);
    deallocate(devices);
    printf("\nFinished.\n");
    return 0;
}
