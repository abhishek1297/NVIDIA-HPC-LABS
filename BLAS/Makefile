NVCC = nvcc
GCC = gcc

#HEADERS =   -I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/targets/x86_64-linux/include \
			-I/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/math_libs/include
#LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/math_libs/lib64 \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/21.2/cuda/11.2/targets/x86_64-linux/lib
OBJS = -llapacke -lcusolver -lcudart -lm


all:
	${GCC} -g -o poi_cpu.out poisson_CPU.c -llapacke -lm

sample:
	${NVCC} -o linear.out ${HEADERS} ${LIBS} linear.c ${OBJS}

clean:
	rm -rf *.out