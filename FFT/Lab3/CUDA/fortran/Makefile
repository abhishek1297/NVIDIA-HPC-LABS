FTN_CC := pgfortran
COMM_PATH := ../../Common/fortran
FTN_FLAGS := -Mcudalib=cufft -Mcuda -I${COMM_PATH}
OBJ := ${COMM_PATH}/fileReader.o ${COMM_PATH}/cuhelper.o


all: clean
	${FTN_CC} -o fftcuda.out cufftCUDA.cuf ${FTN_FLAGS} ${OBJ}
	./fftcuda.out ${ARGS}
	rm -rf *.o *.out *.mod
clean:
	rm -rf *.o *.out *.mod