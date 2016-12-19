#include <random>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Enter output filename and matrix size !" << endl;
        return -1;
    }

    string filename = argv[1];
    int N = atoi(argv[2]);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0,1);

    ofstream outputfile(filename);
    if (outputfile.is_open())
    {
        for (unsigned int i = 0; i < N*N; i++)
            outputfile << dist(gen) << endl;
        outputfile.close();
    }
    else
        cout << "Error writing to output file !" << endl;

    return 0;
}