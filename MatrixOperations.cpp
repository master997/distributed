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

    fileWrite("srcMatrix.txt", srcMatrix);
    fileWrite("op1Matrix.txt", &op1Matrix);
    fileWrite("op2Matrix.txt", &op2Matrix);
    fileWrite("op3Matrix.txt", &op3Matrix);
    fileWrite("dstMatrix.txt", dstMatrix);
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

void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    for (int i = 0; i < srcMatrix->size(); i++)
        for (int j = 0; j < srcMatrix->at(i).size(); j++)
            dstMatrix->at(i).at(j) = srcMatrix->at(i).at(j);
}

void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    for (int i = 0; i < srcMatrix->size(); i++)
        for (int j = 0; j < srcMatrix->at(i).size(); j++)
            dstMatrix->at(i).at(j) = srcMatrix->at(i).at(j);
}
