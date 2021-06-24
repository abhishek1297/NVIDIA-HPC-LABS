CC=pgc++
CPUFLAGS=-acc=multicore -Minfo=accel
GPUFLAGS=-ta:tesla:cc60 -Minfo=accel

jacobi_gpu:
	${CC} ${GPUFLAGS} -o acc_jacobi.out jacobi.cpp
jacobi_cpu:
	${CC} ${CPUFLAGS} -o acc_jacobi.out jacobi.cpp
cpu:
	${CC} ${CPUFLAGS} -o acc_cpu.out acc_test.c
gpu:
	${CC} ${GPUFLAGS} -o acc_gpu.out acc_test.c
clean:
	rm -f *.out
