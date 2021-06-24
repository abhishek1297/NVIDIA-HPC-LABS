C_CC := gcc
COMM_PATH := ../../Inputs/c
FLAGS := -lfftw3 -lfftw3_omp -lfftw3_threads -fopenmp -lm
OBJ := ${COMM_PATH}/fileReader.o
FFTPATH := -L/cm/extra/apps/FFTW/3.3.5/GCC-4.9.3_MVAPICH2-2.2-GDR/lib/
all: clean
	${C_CC}  ${OBJ} -o fftw.out fftw.c ${FFTPATH} ${FLAGS}
	./fftw.out ${ARGS}
clean:
	rm -rf *.o *.out
