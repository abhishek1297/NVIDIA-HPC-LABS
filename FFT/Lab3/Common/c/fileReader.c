#include "fileReader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void generateFile(const char *fname) {

    FILE *fp = fopen(fname, "w");
    int lb = 500, ub = 50000;
    int i;
    if (fp == NULL) {
        printf("\nError: Unable to create the file.\n");
        exit(1);
    }
    for (i=0; i<2000; ++i) {
        fprintf(fp, "%d\n", rand() % (ub - lb + 1) + lb);
    }
    fclose(fp);
}

void loadFrequencies(const char *fname, int *freqArray, int n) {
    
    FILE *fp = fopen(fname, "r");
    char *line;
    size_t len;
    int i;
    if (fp == NULL) {
        printf("\nError: Unable to open the file.\n");
        exit(1);
    }
    for (i=0; i<n; ++i) {
        fscanf(fp, "%d", &freqArray[i]);
    }
    fclose(fp);
}
