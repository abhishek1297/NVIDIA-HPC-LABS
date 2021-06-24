FTN_CC := gfortran
COMM_PATH := ../../Inputs/fortran
FTN_PATH := -I/usr/include -I${COMM_PATH}
FFTW_PATH := -I/cm/extra/apps/FFTW/3.3.5/GCC-4.9.3_MVAPICH2-2.2-GDR/include 
FFT_PATH := -L/cm/extra/apps/FFTW/3.3.5/GCC-4.9.3_MVAPICH2-2.2-GDR/lib/
FLAGS := -lfftw3 -lfftw3_omp -lfftw3_threads -fopenmp -lm
OBJ := fileReader.o

all: clean ${OBJ}
	${FTN_CC} ${FTN_PATH} ${FFT_PATH} ${FFTW_PATH} ${OBJ} -o fftw.out fftw.f90 ${FLAGS}
	./fftw.out ${ARGS}

${OBJ}:
	gfortran -c ${COMM_PATH}/fileReader.f90

clean:
	rm -rf *.o *.out *.mod
