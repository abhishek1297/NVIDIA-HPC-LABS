FTN_CC := pgfortran
FTN_PATH := -I/usr/include -I../../Inputs/fortran
FLAGS := -fast -ta=tesla -Minfo=accel -Mcudalib=cufft
OBJ := ../../Inputs/fortran/fileReader.o

all: clean
	${FTN_CC} ${FTN_PATH} ${OBJ} -o accfft.out cufftACC.f90 ${FLAGS}
	./accfft.out
clean:
	rm -rf *.out *.o *.mod