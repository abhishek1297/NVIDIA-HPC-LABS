C_CC := nvcc
C_FLAGS := -lcuda

all: clean
	${C_CC} -o cumat.out matCUDA.cu ${C_FLAGS}
	./cumat.out ${ARGS}
clean:
	rm -rf *.o *.out