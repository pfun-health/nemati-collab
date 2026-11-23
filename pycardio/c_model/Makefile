gsl_libs = `gsl-config --libs`
gsl_cflags = `gsl-config --cflags`

all: build

build: resp

resp: resp_odes.c
	gcc $(gsl_cflags) $(gsl_libs) -O3 -c -fPIC resp_odes.c 
	gcc -shared resp_odes.o $(gsl_cflags) $(gsl_libs) -o resp_odes.so

clean:
	rm ./*.o && rm ./*.so
