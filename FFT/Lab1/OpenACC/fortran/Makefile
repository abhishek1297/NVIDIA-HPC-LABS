FTN_CC := pgfortran
FLAGS := -fast -ta=tesla -Minfo="accel"
all: clean
	${FTN_CC} -o accmat.out matACC.f90 ${FLAGS}
	./accmat.out ${ARGS}
clean:
	rm -f *.out *.o *.mod