#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <string>
using namespace std;
using namespace chrono;

int numberOfEntries(string filename)
{
	ifstream file(filename);
	if (file.is_open())
	{
		string line;
		int i = 0;
		while (getline(file, line))
			i++;
		file.close();
		return i;
	}
	else
	{
		cout << "Couldn't find file " << filename << " !" << endl;
		return -1;
	}
}
void readFileToArray(string filename, double* arr, int N)
{
	ifstream file(filename);
	if (file.is_open())
	{
		string line;
		int i = 0;
		while (getline(file, line) && i<N)
		{
			arr[i] = atof(line.c_str());
			i++;
		}
		file.close();
	}
	else
	{
		cout << "Couldn't find file " << filename << " !" << endl;
	}
}

__global__ void calculateRanks(double* r_next, double* P, double* r, double* c, int N)
{
	int	i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < N)
	{
		r_next[i] = c[i];
		for (int j = 0; j < N; j++)
			r_next[i] += P[i*N + j] * r[j];
	}
}
__global__ void updateValues(double* r, double* r_next, int N)
{
	int	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<N)
		r[i] = r_next[i];
}



int main(int argc, char *argv[])
{
	//Default settings
	int N = -1;
	int N2 = N*N;
	double a = 0.2;
	int maxIter = 50;
	double epsilon = pow(10, -5);
	int OpenMPThreads = 1;
	string filename = "";

	if (argc < 3)
	{
		cout << "Need at least -file argument !" << endl;
		return -1;
	}

	//Override with command line arguments
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-file") == 0)
		{
			filename = argv[i + 1];
			i++;
		}
		else if (strcmp(argv[i], "-iter") == 0)
		{
			maxIter = atoi(argv[i + 1]);
			i++;
		}
		else if (strcmp(argv[i], "-epsilon") == 0)
		{
			epsilon = pow(10, atoi(argv[i + 1]));
			i++;
		}
		else if (strcmp(argv[i], "-threads") == 0)
		{
			OpenMPThreads = atoi(argv[i + 1]);
			i++;
		}
	}

	omp_set_num_threads(OpenMPThreads);

	if (filename == "")
	{
		cout << "Need a link matrix file !" << endl;
		return -1;
	}

	N2 = numberOfEntries(filename);

	if (N2 == -1)
	{
		cout << "Error reading a file !" << endl;
		return -1;
	}

	N = sqrt(N2);

	//Show arguments
	cout << N << "x" << N << " matrix " << endl;
	cout << "Maximum iteration count = " << maxIter << endl;
	cout << "Epsilon = " << epsilon << endl;
	cout << "OpenMP threads = " << OpenMPThreads << endl;

	//Memory allocations
	double* c = new double[N];
	double* r = new double[N];
	double* r_next = new double[N];
	double* P = new double[N2];

	//Init arrays
	#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		c[i] = 1.0 - a;
		r[i] = 1.0 / N;
		r_next[i] = 0.0;
	}
	//Override link matrix with file
	readFileToArray(filename, P, N2);

	//Damping factor
	#pragma omp parallel for
	for (int i = 0; i < N2; i++)
		P[i] *= a;

	//GPU memory allocations
	double* c_GPU, *r_GPU, *r_next_GPU, *P_GPU;
	cudaMalloc((void**)&c_GPU, sizeof(double)*N);
	cudaMalloc((void**)&r_GPU, sizeof(double)*N);
	cudaMalloc((void**)&r_next_GPU, sizeof(double)*N);
	cudaMalloc((void**)&P_GPU, sizeof(double)*N2);
	//Copy to GPU
	cudaMemcpy(c_GPU, c, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(r_GPU, r, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(r_next_GPU, r_next, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(P_GPU, P, sizeof(double)*N2, cudaMemcpyHostToDevice);


	//Start calculating
	int n = 0;
	auto startTime = high_resolution_clock::now();
	while (true)
	{
		//Calculate r_next
		int threads = 1024;
		int blocks = N / 1024;
		if (blocks == 0)
			blocks = 1;
		calculateRanks << <blocks, threads >> > (r_next_GPU, P_GPU, r_GPU, c_GPU, N);
		cudaThreadSynchronize();
		n++;

		//Calculate the stopping condition
		cudaMemcpy(r, r_GPU, sizeof(double)*N, cudaMemcpyDeviceToHost);
		cudaMemcpy(r_next, r_next_GPU, sizeof(double)*N, cudaMemcpyDeviceToHost);
		double z = 0.0;
		#pragma omp parallel for reduction(+:z)
		for (int i = 0; i < N; i++)
			z += abs(r_next[i] - r[i]);

		//Switch pointers
		updateValues << <N, 1 >> > (r_GPU, r_next_GPU, N);
		cudaThreadSynchronize();

		//Check if should stop
		if (z <= epsilon || maxIter == n)
			break;
	}

	//Print stats
	auto endTime = high_resolution_clock::now();
	auto duration = endTime - startTime;
	cudaMemcpy(r, r_GPU, sizeof(double)*N, cudaMemcpyDeviceToHost);
	cout << "Calculation done in " << (duration_cast<milliseconds>(duration).count()) << " ms!" << endl;
	cout << "Number of iterations calculated: " << n << endl;

	ofstream out(filename + "_Parallel_" + to_string(OpenMPThreads) + " _PageRanks.txt");
	if (out.is_open())
	{
		for (int i = 0; i < N; i++)
			out << r[i] << endl;
		out.close();
	}
	else
		cout << "Couldn't write result to file !" << endl;


	//Free host memory
	delete c;
	delete r;
	delete r_next;
	delete P;
	//Free device memory
	cudaFree(c_GPU);
	cudaFree(r_GPU);
	cudaFree(r_next_GPU);
	cudaFree(P_GPU);

	return 0;
}