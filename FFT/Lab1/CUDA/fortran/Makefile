FTN_CC := pgfortran
FTN_FLAGS := -Mcuda

all: clean
	${FTN_CC}  -o cumat.out matCUDA.cuf ${FTN_FLAGS}
	./cumat.out ${ARGS}
clean:
	rm -rf *.out *.o *.mod
