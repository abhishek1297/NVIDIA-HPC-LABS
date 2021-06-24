FTN_CC := pgfortran
COMM_PATH := ../../Common/fortran
FLAGS := -Mcudalib=cufft -lm -ta=tesla -Minfo=accel -I${COMM_PATH} -w
OBJ := ${COMM_PATH}/fileReader.o ${COMM_PATH}/acchelper.o

all:
	${FTN_CC}  -o fftacc.out cufftACC.f90 ${FLAGS} ${OBJ}
	./fftacc.out ${ARGS}
	rm -rf *.out *.o *.mod
clean:
	rm -rf *.out *.o *.mod