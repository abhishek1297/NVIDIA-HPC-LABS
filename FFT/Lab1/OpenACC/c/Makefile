C_CC := pgcc
FLAGS := -fast -ta=tesla -Minfo="accel"

all:clean
	${C_CC} -o accmat.out matACC.c ${FLAGS}
	./accmat.out ${ARGS}
clean:
	rm -rf *.o *.out