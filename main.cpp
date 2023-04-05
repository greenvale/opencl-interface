#include <iostream>

#include "OpenCL_Interface.hpp"

int main() 
{
    std::cout << "Added to test Git pushing from Visual Studio" << std::endl;

    // initialise values
    float* A = new float[10];
    float* B = new float[10];
    float* C = new float[10];

    for (int i = 0; i < 10; ++i)
    {
        A[i] = i * 1.0;
        B[i] = i * 5.0;
    }

    // kernel path and name
    std::string kernelPath = "myKernel.cl";
    std::string kernelName = "myKernel";

    // Create opencl app object
    OpenCL_Interface myInterface(0, 0);

    myInterface.addKernel(
        kernelPath,
        kernelName,
        {10, 10},
        {10}
    );

    myInterface.runKernel(
        0,
        10,
        5,
        {A, B},
        {C}
    );

    for (int i = 0; i < 10; ++i)
    {
        std::cout << C[i] << std::endl;
    }

}