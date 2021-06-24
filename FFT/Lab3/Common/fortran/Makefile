all:
	pgfortran -c fileReader.f90
	pgfortran -c cuhelper.f90 -Mcuda
	pgfortran -c acchelper.f90
clean:
	rm -rf *.out *.o *.mod