#include "fileReader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define n 50
int main () {
    int i, freqArray[n];
    generateFile("frequencies.txt");

    loadFrequencies("frequencies.txt", freqArray, n);

    for (i=0; i<n; ++i) {
        printf("%d\n", freqArray[i]);
    }
    return 0;
}