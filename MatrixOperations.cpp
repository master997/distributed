#include <vector>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <iomanip>
#include <algorithm>   // std::fill, std::copy

#include "MatrixOperations.h"
#include "FileWrite.h"

void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int dim = (int)srcMatrix->size();

    // Static intermediate matrices: allocated once, reused on every call.
    // main.cpp calls this function 10 times in the timing loop. The shipped
    // version built and destroyed three N x N double matrices per call,
    // which at N=1000 is around 24 MB of allocation per call - 240 MB
    // across the 10 timed runs - just to throw it away. By making them
    // static we pay the cost once for the whole program.
    static std::vector<std::vector<double>> op1Matrix, op2Matrix, op3Matrix;
    if ((int)op1Matrix.size() != dim) {
        op1Matrix.assign(dim, std::vector<double>(dim));
        op2Matrix.assign(dim, std::vector<double>(dim));
        op3Matrix.assign(dim, std::vector<double>(dim));
    }

    operation1(srcMatrix,  &op1Matrix);
    operation2(&op1Matrix, &op2Matrix);
    operation3(&op2Matrix, &op3Matrix);

    // Copy the final result into the destination buffer that main.cpp owns.
    // Using std::copy on contiguous double rows lets the compiler emit a
    // single memcpy under the hood instead of a per-element loop.
    for (int i = 0; i < dim; ++i) {
        if ((int)(*dstMatrix)[i].size() != dim) (*dstMatrix)[i].resize(dim);
        std::copy(op3Matrix[i].begin(), op3Matrix[i].end(), (*dstMatrix)[i].begin());
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

// OPERATION 3 - Matrix multiplication, dst = src x src, parallelised by row-strips.
// Each worker thread owns a contiguous block of dst rows. The threads only
// ever WRITE to their own rows, so two threads can never touch the same
// memory location - that means we don't need any mutex. The reads from src
// are read-only and shared, which is also safe (DC-3 lecture).
// Loop order inside each worker stays (i, k, j) for the same cache reason
// as in the sequential version: the inner j-loop streams one row of dst and
// one row of src sequentially.
// Pattern lifted from DC-2 Tutorial Example 12a: build a vector of threads
// in a for loop, then join them in a second for loop.
void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // Matmul accumulates into dst, so the destination must start at zero.
    for (int i = 0; i < N; ++i)
        std::fill((*dstMatrix)[i].begin(), (*dstMatrix)[i].end(), 0.0);

    // Use as many worker threads as the hardware reports cores for. On an
    // M-series Mac that's typically 8-10 (performance + efficiency cores).
    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());

    // Each worker handles rows [row_start, row_end). Row-strip rather than
    // round-robin because contiguous rows share L2 cache lines.
    auto worker = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            auto& row_i_dst = (*dstMatrix)[i];
            for (int k = 0; k < N; ++k) {
                const double a_ik = (*srcMatrix)[i][k];
                const auto& row_k = (*srcMatrix)[k];
                for (int j = 0; j < N; ++j)
                    row_i_dst[j] += a_ik * row_k[j];
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(T);
    const int chunk = (N + (int)T - 1) / (int)T;
    for (unsigned int t = 0; t < T; ++t) {
        int s = (int)t * chunk;
        int e = std::min(N, s + chunk);
        if (s < e) threads.emplace_back(worker, s, e);
    }
    for (auto& th : threads) th.join();
}
