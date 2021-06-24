C_CC := pgcc
COMM_PATH := ../../../Common/c
FLAGS := -Mcudalib=cufft -lm -ta=tesla -Minfo=accel -I${COMM_PATH} -w
OBJ := ${COMM_PATH}/fileReader.o ${COMM_PATH}/acchelper.o

all:
	${C_CC} -o fftacc.out cufftACC.c ${FLAGS} ${OBJ}
	./fftacc.out ${ARGS}
	rm -rf *.out *.o
clean:
	rm -rf *.out *.o