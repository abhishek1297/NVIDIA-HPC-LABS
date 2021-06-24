FTN_CC := pgfortran
FTN_FLAGS := -Mcudalib=cufft -Mcuda
FTN_PATH := -I/usr/include -I../../Inputs/fortran
OBJ := ../../Inputs/fortran/fileReader.o

all: clean
	${FTN_CC} ${FTN_PATH} ${OBJ} -o cufft.out cufftCUDA.f90 ${FTN_FLAGS}
	./cufft.out
clean:
	rm -rf *.out *.o *.mod