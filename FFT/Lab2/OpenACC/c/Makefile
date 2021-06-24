C_CC := pgcc
FLAGS := -fast -ta=tesla -Minfo=accel -Mcudalib=cufft
OBJ := ../../Inputs/c/fileReader.o

all: clean
	${C_CC} ${OBJ} -o accfft.out cufftACC.c ${FLAGS}
	./accfft.out
clean:
	rm -rf *.out *.o