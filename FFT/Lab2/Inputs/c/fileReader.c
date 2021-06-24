#include "fileReader.h"
#include <stdio.h>
#include <stdlib.h>

double randfrom(double min, double max)  {

    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void createRandomFileInputs(int n) {

    FILE *fp = fopen("complexInputs.txt", "w");
    double real;
    int i;
    for (i=0; i<n; ++i) {
        
        real = randfrom(1e-10, 0.99);
        fprintf(fp, "%.30f\n", real);
    }
    fclose(fp);

}

void readInputs(char *fname, double *inputs, int n) {

    FILE *fp = fopen(fname, "r");
    char *line;
    size_t len;
    double real;
    int i;

    for (i=0; i<n; ++i) {

        fscanf(fp, "%lf", &real);
        inputs[i] = real;
    }

    fclose(fp);

}
