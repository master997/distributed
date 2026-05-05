#include <vector>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <iomanip>

#include "MatrixOperations.h"
#include "FileWrite.h"

void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    int dim = srcMatrix->size();

    std::vector<std::vector<double>> op1Matrix(dim);
    std::vector<std::vector<double>> op2Matrix(dim);
    std::vector<std::vector<double>> op3Matrix(dim);

    int cpuCount = std::thread::hardware_concurrency();

    for (int i = 0; i < dim; i++)
    {
        op1Matrix[i].resize(dim);
        op2Matrix[i].resize(dim);
        op3Matrix[i].resize(dim);
    }

    operation1(srcMatrix, &op1Matrix);
    operation2(&op1Matrix, &op2Matrix);
    operation3(&op2Matrix, &op3Matrix);

    for (int i = 0; i < dim; i++)
    {
        dstMatrix->at(i).resize(dim);
        for (int j = 0; j < dim; j++)
        {
            dstMatrix->at(i).at(j) = op3Matrix.at(i).at(j);
        }
    }

    // The shipped template wrote five matrices to disk on every call.
    // That meant the timed loop in main.cpp was measuring file I/O instead
    // of our algorithm, and the spec explicitly tells us to "remove any
    // debug printing". They're behind VERIFY now: when we want to inspect
    // the intermediate matrices we re-build with -DVERIFY, otherwise they
    // never run during the assessed timed loop.
#ifdef VERIFY
    fileWrite("srcMatrix.txt", srcMatrix);
    fileWrite("op1Matrix.txt", &op1Matrix);
    fileWrite("op2Matrix.txt", &op2Matrix);
    fileWrite("op3Matrix.txt", &op3Matrix);
    fileWrite("dstMatrix.txt", dstMatrix);
#endif
}

// OPERATION 1 - Matrix transposition.
// Walks every cell of the source and writes it to the swapped position in the
// destination, so dst[i][j] becomes src[j][i] (mirror across the diagonal).
// We do this sequentially first so we can prove correctness against the
// shipped example matrices before adding any threading on top.
void operation1(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // Using [] instead of .at() because .at() does a bounds check on every
    // access. We control the loop bounds here, so the check is wasted work.
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            (*dstMatrix)[i][j] = (*srcMatrix)[j][i];
}

// OPERATION 2 - Zone sum (3x3 stencil).
// Each destination cell is the sum of the corresponding source cell and
// every neighbour that exists. So a corner cell sums 4 values, an edge
// cell sums 6, and an interior cell sums 9. We just bounds-check the
// neighbour coordinates and skip whatever falls outside the matrix.
// Sequential here on purpose — we'll add threads in a later change once
// we know the math is right.
void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int di = -1; di <= 1; ++di) {
                int ni = i + di;
                if (ni < 0 || ni >= N) continue;
                for (int dj = -1; dj <= 1; ++dj) {
                    int nj = j + dj;
                    if (nj < 0 || nj >= N) continue;
                    sum += (*srcMatrix)[ni][nj];
                }
            }
            (*dstMatrix)[i][j] = sum;
        }
    }
}

// OPERATION 3 - Matrix multiplication, dst = src x src.
// Loop order is (i, k, j), NOT the textbook (i, j, k).
// Why this order: with row-major storage, the inner j-loop walks one row of
// dst and one row of src in step. Both reads and writes are sequential, so
// each cache line is reused for many iterations before being evicted.
// The standard (i, j, k) order makes the inner k-loop walk a column of src
// (k stride of N doubles), which trashes the cache. Same maths, same result,
// but on N=1000 we typically see a 3-5x improvement just from the reorder.
// Still sequential here — threads added in a later change.
void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // Matmul accumulates into dst, so it has to start at zero.
    for (int i = 0; i < N; ++i)
        std::fill((*dstMatrix)[i].begin(), (*dstMatrix)[i].end(), 0.0);

    for (int i = 0; i < N; ++i) {
        auto& row_i_dst = (*dstMatrix)[i];
        for (int k = 0; k < N; ++k) {
            // Pulling a_ik out of the inner loop so the compiler can keep it
            // in a register instead of re-reading it on every j step.
            const double a_ik = (*srcMatrix)[i][k];
            const auto& row_k = (*srcMatrix)[k];
            for (int j = 0; j < N; ++j)
                row_i_dst[j] += a_ik * row_k[j];
        }
    }
}
