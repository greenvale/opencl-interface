#include <iostream>

#include "OpenCLInterface.hpp"
#include <vector>
#include <string>

int main() 
{
    // initialise values
    float* A = new float[10];
    float* B = new float[10];
    float* C = new float[10];

    for (int i = 0; i < 10; ++i)
    {
        A[i] = i * 1.0;
        B[i] = i * 2.0;
    }

    // kernel path and name
    std::string kernelPath = "myKernel.cl";
    std::string kernelName = "myKernel";

    // Create opencl app object
    OpenCLInterface myInterface(0, 0);

    myApp.addKernel(
        kernelPath,
        kernelName,
        {10, 10},
        {10}
    );

    myApp.runKernel(
        10,
        5,
        {std::make_tuple(A, 10), std::make_tuple(B, 10)},
        {std::make_tuple(C, 10)}
    );

    for (int i = 0; i < 10; ++i)
    {
        std::cout << C[i] << std::endl;
    }

}