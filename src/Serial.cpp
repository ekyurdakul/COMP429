#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <string.h>
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

int main(int argc, char *argv[])
{
	//Default settings
	int N = -1;
	int N2 = N*N;
	double a = 0.2;
	int maxIter = 50;
	double epsilon = pow(10, -5);
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
	}

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

	//Memory allocations
	double* c = new double[N];
	double* r = new double[N];
	double* r_next = new double[N];
	double* P = new double[N2];

	//Init arrays
	for (int i = 0; i < N; i++)
	{
		c[i] = 1.0 - a;
		r[i] = 1.0 / N;
		r_next[i] = 0.0;
	}
	//Override link matrix with file
	readFileToArray(filename, P, N2);

	//Damping factor
	for (int i = 0; i < N2; i++)
		P[i] *= a;

	//Start calculating
	int n = 0;
	auto startTime = high_resolution_clock::now();
	while (true)
	{
		//Calculate r_next
		for (int i = 0; i < N; i++)
		{
			r_next[i] = c[i];
			for (int j = 0; j < N; j++)
				r_next[i] += P[i*N + j] * r[j];
		}
		n++;

		//Calculate the stopping condition
		double z = 0.0;
		for (int i = 0; i < N; i++)
			z += abs(r_next[i] - r[i]);

		//Switch pointers
		for (int i = 0; i < N; i++)
			r[i] = r_next[i];

		//Check if should stop
		if (z <= epsilon || maxIter == n)
			break;
	}

	//Print stats
	auto endTime = high_resolution_clock::now();
	auto duration = endTime - startTime;
	cout << "Calculation done in " << (duration_cast<milliseconds>(duration).count()) << " ms!" << endl;
	cout << "Number of iterations calculated: " << n << endl;

	ofstream out(filename + "_Serial_PageRanks.txt");
	if (out.is_open())
	{
		for (int i = 0; i < N; i++)
			out << r[i] << endl;
		out.close();
	}
	else
		cout << "Couldn't write result to file !" << endl;

	//Free memory
	delete c;
	delete r;
	delete r_next;
	delete P;

	return 0;
}