/* 
    ==============================================================================================
     OpenCL_Interface and OpenCL_KernelInterface class header file (William Denny, 12th Aug 2022)
    ==============================================================================================
*/

#pragma once

#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <string>
#include <cassert>

// ========================================================
// OPENCL_KERNELINTERFACE 

class OpenCL_KernelInterface
{
private:
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
        const std::vector<int>& kernelInputArraySizes, 
        const std::vector<int>& kernelOutputArraySizes
    );

    void setInputBuffers(
        cl::CommandQueue* queuePtr,
        const std::vector<float*>& inputArrayPtrs
    );

    void setKernelArgs();

    void runKernel(
        cl::CommandQueue* queuePtr,
        const int& numWorkitems,
        const int& workgroupSize
    );

    void getOutputBuffers(
        cl::CommandQueue* queuePtr,
        const std::vector<float*>& outputArrayPtrs
    );

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

// ========================================================
// OPENCL_INTERFACE 

class OpenCL_Interface
{
private:
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
        const int& numWorkitems,
        const int& workgroupSize,
        const std::vector<float*>& inputArrayPtrs,
        const std::vector<float*>& outputArrayPtrs
    );

};
