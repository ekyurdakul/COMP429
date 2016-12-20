COMP 429 - Parallel Programming Term Project
============================================

Implementation of Google's PageRank algorithm with CUDA and OpenMP in C++.

Index
-----
- `Versions`_
- `Build`_
- `Run`_
- `Arguments`_
- `References`_

Versions
--------
#) Serial
    Operates on dense matrices without any parallelization
#) Parallel
    Operates on dense matrices with CUDA and OpenMP parallelization
#) Serial Sparse
    Operates on sparse matrices without any parallelization
#) Parallel Sparse
    Operates on sparse matrices with CUDA and OpenMP parallelization

Build
-----

Compile the code and generate data by executing in terminal:

::

    cd data/
    make
    cd ../src/
    make

Run
---

Run all test cases in the makefile by executing in terminal:

::

    cd src/
    make run


Arguments
---------

Description of the program arguments in the makefile.

Mandatory Arguments:

- file
    link matrix file name
- n (sparse versions only)
    n by n square matrix size
- threads (parallel versions only)
    number of OpenMP threads

Optional Arguments:

- iter
    number of maximum iterations
- epsilon
    termination criteria 10^epsilon

References
----------
#) \S. Brin, L. Page. The Anatomy of a Large-Scale Hypertextual Web Search Engine. http://infolab.stanford.edu/~backrub/google.html
#) \I. Rogers. The Google Pagerank Algorithm and How It Works. http://www.cs.princeton.edu/~chazelle/courses/BIB/pagerank.htm
#) \K. Bryan, T. Leise. The $25,000,000,000 Eigenvector The Linear Algebra Behind Google. http://www.rose-hulman.edu/~bryan/googleFinalVersionFixed.pdf
#) http://www.cmpe.boun.edu.tr/~ozturan/etm555/google.pdf