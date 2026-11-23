gsl_libs = `gsl-config --libs`
gsl_cflags = `gsl-config --cflags`

python_libs = `python-config --ldflags`
python_cflags = `python-config --cflags`

cflags = -lm

program_name := utils

c_files := $(wildcard *.c)
obj_files := ${c_files:.c=.o}
shared_obj := ${program_name}.so

all: debug

build: utils

debug: $(c_files)
	gcc -D DEBUG $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) -O3 -c -fPIC $(c_files)
	gcc -shared $(obj_files) $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) $(cflags) -o $(shared_obj)

$(program_name): $(c_files)
	gcc $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) -O3 -c -fPIC $(c_files)
	gcc -shared $(obj_files) $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) $(cflags) -o $(shared_obj)

run: $(c_files)
	gcc -D DEBUG $(gsl_cflags) $(python_cflags) $(gsl_libs) $(python_libs) $(c_files) $(cflags) -o $(program_name)
	./$(program_name)
