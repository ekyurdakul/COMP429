all: Serial SerialSparse Parallel ParallelSparse

%.o: %.cu
	nvcc -c -o $@ $^ -std=c++11 -Xcompiler -fopenmp -O3

%.o: %.cpp
	g++ -c -o $@ $^ -std=c++11 -fopenmp -Wall -O3

Serial: Serial.o
	g++ -o $@ $^ -std=c++11 -fopenmp -Wall -O3

SerialSparse: SerialSparse.o
	g++ -o $@ $^ -std=c++11 -fopenmp -Wall -O3

Parallel: Parallel.o
	nvcc -o $@ $^ -std=c++11 -Xcompiler -fopenmp -O3

ParallelSparse: ParallelSparse.o
	nvcc -o $@ $^ -std=c++11 -Xcompiler -fopenmp -O3

clean:
	-rm *.o Serial SerialSparse Parallel ParallelSparse