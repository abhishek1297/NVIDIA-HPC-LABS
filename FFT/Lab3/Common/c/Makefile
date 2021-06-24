all:
	gcc -c fileReader.c
	nvcc -c cuhelper.c -lcufft
	pgcc -c acchelper.c -Mcudalib=cufft -w
clean:
	rm -rf *.out *.o