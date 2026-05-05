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

// OPERATION 1 - Matrix transposition, parallelised by row-strips.
// Each worker thread fills a contiguous block of dst rows. Transpose only
// reads from src and only writes to dst, and each thread writes to a
// disjoint set of rows, so there's no race - no mutex needed (the same
// safety reasoning as in operation3, from DC-3 lecture).
// Same row-strip pattern as DC-2 Tutorial Example 12a.
void operation1(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();
    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());

    auto worker = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i)
            for (int j = 0; j < N; ++j)
                // Using [] instead of .at() because .at() does a bounds
                // check we don't need - the loop guarantees we're in range.
                (*dstMatrix)[i][j] = (*srcMatrix)[j][i];
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

// OPERATION 2 - Zone sum (3x3 stencil), parallelised by row-strips.
// Each cell of dst is the sum of the matching src cell and every neighbour
// that exists - so corners sum 4 values, edges sum 6, interior cells sum 9.
// Threads each own a contiguous block of dst rows and only read from src,
// so once again no mutex is needed (DC-3 lecture's safety reasoning).
// Same row-strip pattern as operations 1 and 3; the unit of work is one row.
void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();
    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());

    auto worker = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                // Walk the 3x3 window centred on (i, j). The two if-checks
                // skip neighbours that fall outside the matrix - that's
                // what makes corners sum 4 and edges sum 6 for free, no
                // separate corner/edge code paths needed.
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
