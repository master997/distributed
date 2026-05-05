// One-time correctness harness.
// Reads srcMatrix.txt from a directory full of "ExampleMatricesN" files,
// runs each of operation1 / operation2 / operation3 in turn, and diffs the
// result against the expected opNMatrix.txt.
// Build (separately from the main app):
//   clang++ -std=c++17 -O2 verify.cpp MatrixOperations.cpp FileRead.cpp -o verify
// Run:
//   ./verify /tmp/example1
// NOT part of the submission — kept around just so we can re-check correctness
// after every change without having to read the matrices by eye.

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

#include "FileRead.h"
#include "MatrixOperations.h"

static bool diff(const std::vector<std::vector<double>>& a,
                 const std::vector<std::vector<double>>& b,
                 const std::string& label)
{
    if (a.size() != b.size()) {
        std::cerr << label << ": size mismatch " << a.size() << " vs " << b.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            std::cerr << label << ": row " << i << " size mismatch\n";
            return false;
        }
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (std::fabs(a[i][j] - b[i][j]) > 1e-6) {
                std::cerr << label << ": cell ["<<i<<"]["<<j<<"] = " << a[i][j]
                          << " expected " << b[i][j] << "\n";
                return false;
            }
        }
    }
    std::cout << label << ": OK (" << a.size() << "x" << a.size() << ")\n";
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: verify <example_dir>\n";
        return 1;
    }
    std::string dir = argv[1];

    // Read the source matrix and its expected outputs from disk. The first
    // line of every file is the matrix dimension, followed by the values.
    auto src    = fileRead(dir + "/srcMatrix.txt");
    auto expOp1 = fileRead(dir + "/op1Matrix.txt");
    auto expOp2 = fileRead(dir + "/op2Matrix.txt");
    auto expOp3 = fileRead(dir + "/op3Matrix.txt");
    int N = (int)src.size();

    // Allocate matching destination buffers for each operation.
    std::vector<std::vector<double>> op1(N, std::vector<double>(N));
    std::vector<std::vector<double>> op2(N, std::vector<double>(N));
    std::vector<std::vector<double>> op3(N, std::vector<double>(N));

    operation1(&src, &op1);
    bool ok1 = diff(op1, expOp1, "operation1 (transpose)");

    operation2(&op1, &op2);
    bool ok2 = diff(op2, expOp2, "operation2 (zone_sum)");

    operation3(&op2, &op3);
    bool ok3 = diff(op3, expOp3, "operation3 (matmul)");

    return (ok1 && ok2 && ok3) ? 0 : 1;
}
