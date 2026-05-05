#pragma once

#include <vector>
#include "Barrier.h"

// Public entry point - prototype must stay exactly as provided
void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix);

// Internal operations work on flat 1D arrays (row-major, index = i*N + j).
// Flat arrays are used internally because vector<vector<double>> stores each
// row as a separate heap allocation, meaning every row access follows a pointer
// to a different memory location. At N=1000 with 1 billion inner loop iterations
// in matmul, that pointer chasing destroys cache performance.
// Flat arrays put all N*N doubles in one contiguous block so the CPU prefetcher
// can do its job properly.
void operation1(const double* src, double* dst, int N);  // transpose
void operation2(const double* src, double* dst, int N);  // zone sum
void operation3(const double* src, double* dst, int N);  // matrix multiply
