GSL_CFLAGS = $(shell gsl-config --cflags)
GSL_LDFLAGS = $(shell gsl-config --libs)
CFLAGS_FAST = -fopenmp -std=c++1z -O3 -mtune=native -march=native -mfpmath=both  -Wall  -Wextra 

all : GeoDynamics.bin 

GeoDynamics.bin: GeoDynamics.cc
	$(CXX) -o $@ $< $(CFLAGS_FAST) $(LDFLAGS) $(GSL_CFLAGS) $(GSL_LDFLAGS) -lz
