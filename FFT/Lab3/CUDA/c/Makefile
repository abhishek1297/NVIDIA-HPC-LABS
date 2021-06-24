C_CC := nvcc
C_FLAGS :=  -lcuda -lcufft -lm
OBJ := ../../Common/c/fileReader.o ../../Common/c/cuhelper.o
CUPATH := -L/cm/extra/apps/NVIDIA-HPC-SDK/Linux_x86_64/20.9/math_libs/11.0/targets/x86_64-linux/lib

all: 
	${C_CC} ${CUPATH} -o fftcuda.out cufftCUDA.cu ${C_FLAGS} ${OBJ}
	./fftcuda.out ${ARGS}
	rm -f *.out *.o
clean:
	rm -f *.out *.o
