all: GenerateDense GenerateSparse dense sparse

%.o: %.cpp
	g++ -c -o $@ $^ -std=c++11 -Wall -O3

GenerateDense : GenerateDense.o
	g++ -o $@ $^ -std=c++11 -Wall -O3

GenerateSparse : GenerateSparse.o
	g++ -o $@ $^ -std=c++11 -Wall -O3

dense:
	-./GenerateDense 4x4.txt 4
	-./GenerateDense 8x8.txt 8
	-./GenerateDense 16x16.txt 16
	-./GenerateDense 32x32.txt 32
	-./GenerateDense 64x64.txt 64
	-./GenerateDense 128x128.txt 128
	-./GenerateDense 256x256.txt 256
	-./GenerateDense 512x512.txt 512
	-./GenerateDense 1024x1024.txt 1024
	-./GenerateDense 2048x2048.txt 2048
	-./GenerateDense 4096x4096.txt 4096
	-./GenerateDense 8192x8192.txt 8192

sparse:
	-./GenerateSparse 4x4 4 0.2
	-./GenerateSparse 8x8 8 0.2
	-./GenerateSparse 16x16 16 0.1
	-./GenerateSparse 32x32 32 0.1
	-./GenerateSparse 64x64 64 0.1
	-./GenerateSparse 128x128 128 0.1
	-./GenerateSparse 256x256 256 0.1
	-./GenerateSparse 512x512 512 0.1
	-./GenerateSparse 1024x1024 1024 0.1
	-./GenerateSparse 2048x2048 2048 0.1
	-./GenerateSparse 4096x4096 4096 0.1
	-./GenerateSparse 8192x8192 8192 0.1

clean:
	-rm *.o GenerateDense GenerateSparse