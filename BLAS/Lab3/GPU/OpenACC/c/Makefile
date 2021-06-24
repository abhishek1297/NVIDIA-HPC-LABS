CC := pgcc
OBJS := -lm -Mcudalib=cusolver -Mcudalib=cusparse
FLAGS := -ta=tesla -Minfo=accel -w

all:
	${CC} ${FLAGS} -o poiacc.out poisson_ACC.c ${OBJS}
clean:
	rm -rf *.out
