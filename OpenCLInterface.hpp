#pragma once

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <cassert>

struct OpenCLKernel
{
    int numInputArrays;
    int numOutputArrays;
    std::vector<int> inputArraySizes;
    std::vector<int> outputArraySizes;
    std::vector<cl::Buffer> inputBuffers;
    std::vector<cl::Buffer> outputBuffers;
};

class OpenCLInterface
{
private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform platform;
    cl::Device device;

    cl::Context context;
    cl::CommandQueue queue;

    cl::Kernel kernel;

    std::vector<cl::Buffer> inputBuffers;
    std::vector<cl::Buffer> outputBuffers;

public:
    OpenCLInterface();

    OpenCLInterface(
        const int& targetPlatformIndex, 
        const int& targetDeviceIndex
    );

    void addKernel(
        const std::string& kernelPath, 
        const std::string& kernelName,
        const std::vector<int>& inputArraySizes,
        const std::vector<int>& outputArraySizes
    );

    void runKernel(
        const int& numElements,
        const int& workgroupSize,
        const std::vector<std::tuple<float*, int>>& inputArrays,
        const std::vector<std::tuple<float*, int>>& outputArrays
    );

    cl::Kernel getKernel(
        const std::string& kernelSource, 
        const std::string& kernelName
    );

    static std::string getKernelSource(
        const std::string& kernelPath
    );

};
