wfdb_libs = `wfdb-config --libs`
wfdb_cflags = `wfdb-config --cflags`
gcc_flags = -std=c11 -O3 -fPIC
o_flags = -std=c11 -O3
gsl_libs = `gsl-config --libs`
gsl_cflags = `gsl-config --cflags`

all: build


build: calc_rsa calc_cvc calc_dfa

calc_rsa: calc_rsa.c
	gcc $(o_flags) -c -fPIC calc_rsa.c
	gcc -shared calc_rsa.o $(o_flags) -o calc_rsa.so

calc_cvc: calc_cvc.c
	gcc $(o_flags) -c -fPIC calc_cvc.c
	gcc -shared calc_cvc.o $(o_flags) -o calc_cvc.so

calc_dfa: calc_dfa.c
	gcc $(o_flags) $(gsl_cflags) $(gsl_libs) -c -fPIC calc_dfa.c
	gcc -shared calc_dfa.o $(o_flags) $(gsl_cflags) $(gsl_libs) -o calc_dfa.so

clean:
	rm ./*.o
	rm ./*.so
