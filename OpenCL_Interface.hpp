#pragma once

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <cassert>

class OpenCL_KernelInterface
{
public:
    std::string name;
    int numInputArrays;
    int numOutputArrays;
    std::vector<int> inputArraySizes;
    std::vector<int> outputArraySizes;
    std::vector<cl::Buffer> inputBuffers;
    std::vector<cl::Buffer> outputBuffers;
    cl::Kernel kernel;

public:
    OpenCL_KernelInterface();
    OpenCL_KernelInterface(
        cl::Device* devicePtr,
        cl::Context* contextPtr,
        const std::string& kernelPath,
        const std::string& kernelName,
        const std::vector<int>& newInputArraySizes, 
        const std::vector<int>& newOutputArraySizes
    );

    int getNumInputArrays();
    int getNumOutputArrays();
    int getInputArraySize(const int& index);
    int getOutputArraySize(const int& index);

    cl::Buffer* getInputBufferPtr(const int& index);
    cl::Buffer* getOutputBufferPtr(const int& index);
    cl::Kernel* getKernelPtr();

    static cl::Kernel getKernel(
        cl::Device* devicePtr,
        cl::Context* contextPtr,
        const std::string& kernelSource, 
        const std::string& kernelName
    );

    static std::string getKernelSource(
        const std::string& kernelPath
    );

};

class OpenCL_Interface
{
private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform platform;
    cl::Device device;

    cl::Context context;
    cl::CommandQueue queue;

    std::vector<OpenCL_KernelInterface> kernelInterfaces;

public:
    OpenCL_Interface();

    OpenCL_Interface(
        const int& platformIndex, 
        const int& deviceIndex
    );

    void addKernel(
        const std::string& kernelPath, 
        const std::string& kernelName,
        const std::vector<int>& inputArraySizes,
        const std::vector<int>& outputArraySizes
    );

    void runKernel(
        const int& index,
        const int& numElements,
        const int& workgroupSize,
        const std::vector<float*>& inputArrays,
        const std::vector<float*>& outputArrays
    );

};
