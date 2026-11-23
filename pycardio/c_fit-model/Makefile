gsl_libs = `gsl-config --libs`
gsl_cflags = `gsl-config --cflags`

python_libs = `python-config --ldflags`
python_cflags = `python-config --cflags`

program_name := cmaes

c_files := $(wildcard *.c)
obj_files := ${c_files:.c=.o}
shared_obj := ${program_name}.so

all: build run

build: cmaes

cmaes: $(c_files)
	gcc $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) -O3 -c -fPIC $(c_files)
	gcc -shared $(obj_files) $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) -o $(shared_obj)

run: $(c_files)
	gcc -D DEBUG $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) $(c_files) -o $(program_name)
	./$(program_name)