#include "fileReader.h"
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    
    double execTime;
    int bytesPerLine = 33;
    struct timeval start, stop;
    int N = 4096;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    gettimeofday(&start, NULL);
    createRandomFileInputs(N);
    gettimeofday(&stop, NULL);
    execTime += (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
    printf("\nTotal Time: \t%.10f s\n", execTime);
    printf("\nFile Size: %f GB\n", (double)(bytesPerLine * N)/1e9);
    return 0;
}
