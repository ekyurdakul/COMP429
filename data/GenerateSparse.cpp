#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Enter output filename, matrix size and sparsity level !" << endl;
        return -1;
    }

    string filename = argv[1];
    unsigned int N = atoi(argv[2]);
    float sparsity = atof(argv[3]);

    unsigned int actualN = floor(N*N*sparsity);

    //P file
    ofstream pfile(filename + "_P.txt");
    if (pfile.is_open())
    {
        for (unsigned int i = 0; i < actualN; i++)
            pfile << 1 << endl;
        pfile.close();
    }
    else
        cout << "Error writing to output file !" << endl;

    random_device rd;
    mt19937 gen(rd());
    //Rows,cols files
    ofstream rows(filename + "_N1.txt");
    ofstream cols(filename + "_N2.txt");
    if (rows.is_open() && cols.is_open())
    {
        vector<unsigned int> indices = {};
        for (unsigned int i = 0; i < N; i++)
            indices.push_back(i);

        vector<unsigned int> vrows = {1};
        while (vrows[vrows.size()-1] != actualN + 1)
        {
            vrows.clear();
            vrows.push_back(1);
            for (unsigned int i = 0; i < N; i++)
            {
                if (vrows[vrows.size()-1] == actualN)
                    break;
                uniform_int_distribution<int> dist(0,N);
                unsigned int newel = vrows[vrows.size()-1] + dist(gen);
                if (newel > actualN)
                    vrows.push_back(actualN+1);
                else
                    vrows.push_back(newel);
            }
        }

        for (auto& i : vrows)
            rows << i << endl;

        for (unsigned int i = 0; i < vrows.size()-1; i++)
        {
            unsigned int tempN = vrows[i+1] - vrows[i];

            shuffle(indices.begin(),indices.end(),gen);
            
            vector<unsigned int> tempIndices = {};
            for (unsigned int i = 0; i < tempN; i++)
                tempIndices.push_back(indices[i]);

            sort(tempIndices.begin(),tempIndices.end());

            for (auto& i : tempIndices)
                cols << i << endl;
        }

        rows.close();
        cols.close();
    }
    else
        cout << "Error writing to output file !" << endl;

    return 0;
}