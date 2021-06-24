CC := pgcc
OBJS := -lm -Mcudalib=cusolver -Mcudalib=cusparse
FLAGS := -w

all:
	${CC} ${FLAGS} -o cuhost.out cusolverSp_eg.c ${OBJS}
clean:
	rm -rf *.out
