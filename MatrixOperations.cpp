#include <vector>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <future>
#include <memory>

#include "MatrixOperations.h"
#include "FileWrite.h"
#include "ThreadPool.h"

// Single thread pool shared by all three operations.
// We create it once and keep it alive for the whole program so we dont
// pay the cost of spawning threads on every single function call.
// Without this, each operation would spawn and destroy 8 threads per call,
// and main.cpp calls this 10 times - thats 240 thread spawns for nothing.
// Putting it in a static local means it gets created on the first call
// and stays alive until the program ends.
static ThreadPool& globalPool()
{
    static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    return pool;
}

void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int dim = (int)srcMatrix->size();

    // We need three intermediate matrices to pass results between the operations.
    // Making them static means they only get allocated once - on the first call.
    // The original code created and destroyed these on every call which at N=1000
    // is about 24MB of allocations per call, 240MB total across the 10 runs.
    // Not worth it when we can just reuse them.
    static std::vector<std::vector<double>> op1Matrix, op2Matrix, op3Matrix;
    if ((int)op1Matrix.size() != dim) {
        op1Matrix.assign(dim, std::vector<double>(dim));
        op2Matrix.assign(dim, std::vector<double>(dim));
        op3Matrix.assign(dim, std::vector<double>(dim));
    }

    // Time each operation seperatley so we can see where the time is actually
    // going. We accumulate over all 10 runs and print the average at the end.
    // The chrono calls themselves are basically free so they dont affect the result.
    static double op1_total_ms = 0.0, op2_total_ms = 0.0, op3_total_ms = 0.0;
    static int    call_count   = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    operation1(srcMatrix,  &op1Matrix);
    auto t1 = std::chrono::high_resolution_clock::now();
    operation2(&op1Matrix, &op2Matrix);
    auto t2 = std::chrono::high_resolution_clock::now();
    operation3(&op2Matrix, &op3Matrix);
    auto t3 = std::chrono::high_resolution_clock::now();

    op1_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    op2_total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    op3_total_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
    ++call_count;

    // Print the per-op breakdown after the last iteration so we can see
    // which operation is the bottleneck. Goes to stderr so it doesnt
    // interfere with the Average: line that main.cpp prints.
    if (call_count == 10) {
        std::cerr << "[profile] op1 transpose avg: " << (op1_total_ms / 10.0) << " ms"
                  << " | op2 zone_sum avg: " << (op2_total_ms / 10.0) << " ms"
                  << " | op3 matmul avg: "   << (op3_total_ms / 10.0) << " ms"
                  << "  (total: " << ((op1_total_ms + op2_total_ms + op3_total_ms) / 10.0) << " ms)"
                  << std::endl;
    }

    // Copy the final result into the output buffer that main.cpp gave us.
    // std::copy on a row of doubles is effectively a memcpy under the hood.
    for (int i = 0; i < dim; ++i) {
        if ((int)(*dstMatrix)[i].size() != dim) (*dstMatrix)[i].resize(dim);
        std::copy(op3Matrix[i].begin(), op3Matrix[i].end(), (*dstMatrix)[i].begin());
    }

    // The original template wrote 5 files to disk inside this function every
    // single call. That means the timer in main.cpp was mostly measuring file I/O
    // instead of the actual algorithm - on N=1000 that was way off.
    // Moved them behind VERIFY so they only run when we need to check correctness.
#ifdef VERIFY
    fileWrite("srcMatrix.txt", srcMatrix);
    fileWrite("op1Matrix.txt", &op1Matrix);
    fileWrite("op2Matrix.txt", &op2Matrix);
    fileWrite("op3Matrix.txt", &op3Matrix);
    fileWrite("dstMatrix.txt", dstMatrix);
#endif
}

// Below this size threads arent worth it - the overhead of enqueuing tasks
// is bigger than the work itself. 64 is a safe cutoff based on measured timings.
static constexpr int kParallelThreshold = 64;

// OPERATION 1 - Transpose
// Flips the matrix across its diagonal so dst[i][j] = src[j][i].
// Split into row strips and run each strip on a thread from the pool.
// No mutex needed because each thread writes to its own rows only - two
// threads will never touch the same cell in dst.
void operation1(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // For small matrices just do it inline - not worth the thread dispatch overhead.
    if (N < kParallelThreshold) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                (*dstMatrix)[i][j] = (*srcMatrix)[j][i];
        return;
    }

    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());

    // Each thread handles a block of rows. Writing [i][j] instead of .at(i).at(j)
    // avoids the bounds check on every element - we know the indices are valid.
    auto worker = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i)
            for (int j = 0; j < N; ++j)
                (*dstMatrix)[i][j] = (*srcMatrix)[j][i];
    };

    std::vector<std::future<void>> futures;
    futures.reserve(T);
    const int chunk = (N + (int)T - 1) / (int)T;
    for (unsigned int t = 0; t < T; ++t) {
        int s = (int)t * chunk;
        int e = std::min(N, s + chunk);
        if (s < e) futures.emplace_back(globalPool().enqueue(worker, s, e));
    }
    // Wait for all threads to finish before we move on to operation2.
    for (auto& f : futures) f.get();
}

// OPERATION 2 - Zone sum
// Each cell in the destination gets the sum of itself and all its neighbours
// in the source matrix. Corner cells have 3 neighbours (4 total), edge cells
// have 5 neighbours (6 total), and everything else has 8 neighbours (9 total).
// The bounds checks inside the loop handle all three cases automaticly.
void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // Lambda so we can reuse the same logic for both the single-thread
    // fast path and the multi-thread path below.
    auto compute_rows = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                // Loop over the 3x3 window around (i, j). If a neighbour
                // is outside the matrix we just skip it - thats what gives
                // corner cells 4 values and edge cells 6 without any special casing.
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

    if (N < kParallelThreshold) { compute_rows(0, N); return; }

    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    futures.reserve(T);
    const int chunk = (N + (int)T - 1) / (int)T;
    for (unsigned int t = 0; t < T; ++t) {
        int s = (int)t * chunk;
        int e = std::min(N, s + chunk);
        if (s < e) futures.emplace_back(globalPool().enqueue(compute_rows, s, e));
    }
    for (auto& f : futures) f.get();
}

// OPERATION 3 - Matrix multiplication (src x src)
// This is the expensive one - at N=1000 it accounts for about 96% of the
// total runtime so this is where the parallel speedup really matters.
//
// Loop order is (i, k, j) not the usual (i, j, k). The reason is cache:
// with row-major storage the inner j loop walks one full row of dst and one
// full row of src sequentially. That keeps the cache hot. The standard
// (i, j, k) order steps through a column of src on every inner iteration
// which has a stride of N doubles - terrible for the cache at large N.
void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    const int N = (int)srcMatrix->size();

    // dst accumulates products so it needs to start at zero.
    for (int i = 0; i < N; ++i)
        std::fill((*dstMatrix)[i].begin(), (*dstMatrix)[i].end(), 0.0);

    auto compute_rows = [srcMatrix, dstMatrix, N](int row_start, int row_end) {
        for (int i = row_start; i < row_end; ++i) {
            auto& row_i_dst = (*dstMatrix)[i];
            for (int k = 0; k < N; ++k) {
                // Pull a_ik out of the j loop so it only gets loaded once
                // per k iteration instead of N times. The compiler can keep
                // it in a register for the entire inner loop.
                const double a_ik = (*srcMatrix)[i][k];
                const auto& row_k = (*srcMatrix)[k];
                for (int j = 0; j < N; ++j)
                    row_i_dst[j] += a_ik * row_k[j];
            }
        }
    };

    if (N < kParallelThreshold) { compute_rows(0, N); return; }

    // Split the rows evenly across however many cores we have.
    // Each thread writes to its own rows so theres no data race and no
    // need for any locking.
    const unsigned int T = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    futures.reserve(T);
    const int chunk = (N + (int)T - 1) / (int)T;
    for (unsigned int t = 0; t < T; ++t) {
        int s = (int)t * chunk;
        int e = std::min(N, s + chunk);
        if (s < e) futures.emplace_back(globalPool().enqueue(compute_rows, s, e));
    }
    for (auto& f : futures) f.get();
}
