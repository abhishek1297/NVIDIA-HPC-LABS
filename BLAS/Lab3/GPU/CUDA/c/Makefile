CC := nvcc
OBJS := -lm -lcusolver -lcudart
FLAGS := -w

all:
	${CC} ${FLAGS} -o poicuda.out poisson_CUDA.cu ${OBJS}
clean:
	rm -rf *.out
