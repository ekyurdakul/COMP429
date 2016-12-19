all: Serial SerialSparse Parallel ParallelSparse

run: runSerial runSerialSparse runParallel runParallelSparse

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

runSerial:
	-./Serial -file ../data/4x4
	-./Serial -file ../data/8x8
	-./Serial -file ../data/16x16
	-./Serial -file ../data/32x32
	-./Serial -file ../data/64x64
	-./Serial -file ../data/128x128
	-./Serial -file ../data/256x256
	-./Serial -file ../data/512x512
	-./Serial -file ../data/1024x1024
	-./Serial -file ../data/2048x2048
	-./Serial -file ../data/4096x4096
	-./Serial -file ../data/8192x8192

runParallel:
	-./Parallel -file ../data/4x4 -threads 4
	-./Parallel -file ../data/8x8 -threads 4
	-./Parallel -file ../data/16x16 -threads 4
	-./Parallel -file ../data/32x32 -threads 4
	-./Parallel -file ../data/64x64 -threads 4
	-./Parallel -file ../data/128x128 -threads 4
	-./Parallel -file ../data/256x256 -threads 4
	-./Parallel -file ../data/512x512 -threads 4
	-./Parallel -file ../data/1024x1024 -threads 4
	-./Parallel -file ../data/2048x2048 -threads 4
	-./Parallel -file ../data/4096x4096 -threads 4
	-./Parallel -file ../data/8192x8192 -threads 4

runSerialSparse:
	-./SerialSparse -file ../data/4x4 -n 4
	-./SerialSparse -file ../data/8x8 -n 8
	-./SerialSparse -file ../data/16x16 -n 16
	-./SerialSparse -file ../data/32x32 -n 32
	-./SerialSparse -file ../data/64x64 -n 64
	-./SerialSparse -file ../data/128x128 -n 128
	-./SerialSparse -file ../data/256x256 -n 256
	-./SerialSparse -file ../data/512x512 -n 512
	-./SerialSparse -file ../data/1024x1024 -n 1024
	-./SerialSparse -file ../data/2048x2048 -n 2048
	-./SerialSparse -file ../data/4096x4096 -n 4096
	-./SerialSparse -file ../data/8192x8192 -n 8192

runParallelSparse:
	-./ParallelSparse -file ../data/4x4 -n 4 -threads 4
	-./ParallelSparse -file ../data/8x8 -n 8 -threads 4
	-./ParallelSparse -file ../data/16x16 -n 16 -threads 4
	-./ParallelSparse -file ../data/32x32 -n 32 -threads 4
	-./ParallelSparse -file ../data/64x64 -n 64 -threads 4
	-./ParallelSparse -file ../data/128x128 -n 128 -threads 4
	-./ParallelSparse -file ../data/256x256 -n 256 -threads 4
	-./ParallelSparse -file ../data/512x512 -n 512 -threads 4
	-./ParallelSparse -file ../data/1024x1024 -n 1024 -threads 4
	-./ParallelSparse -file ../data/2048x2048 -n 2048 -threads 4
	-./ParallelSparse -file ../data/4096x4096 -n 4096 -threads 4
	-./ParallelSparse -file ../data/8192x8192 -n 8192 -threads 4