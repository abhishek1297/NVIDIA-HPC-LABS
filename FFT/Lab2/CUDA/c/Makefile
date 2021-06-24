C_CC := nvcc
C_FLAGS := -lcuda -lcufft
OBJ := ../../Inputs/c/fileReader.o
CUPATH := -L/cm/extra/apps/NVIDIA-HPC-SDK/Linux_x86_64/20.9/math_libs/11.0/targets/x86_64-linux/lib

all: clean
	${C_CC} ${CUPATH} ${OBJ} -o cufft.out cufftCUDA.c ${C_FLAGS}
	./cufft.out
clean:
	rm -rf *.out *.o
