#include "OpenCL_Interface.hpp"

OpenCL_KernelInterface::OpenCL_KernelInterface()
{

}

OpenCL_KernelInterface::OpenCL_KernelInterface(
    cl::Device* devicePtr,
    cl::Context* contextPtr,
    const std::string& kernelPath,
    const std::string& kernelName,
    const std::vector<int>& newInputArraySizes, 
    const std::vector<int>& newOutputArraySizes
)
{
    // get kernel source
    std::string kernelSource = getKernelSource(kernelPath);

    // get kernel
    kernel = getKernel(devicePtr, contextPtr, kernelSource, kernelName);

    // get number of input and output arrays required
    numInputArrays = newInputArraySizes.size();
    numOutputArrays = newOutputArraySizes.size();

    // get vector input and output array sizes
    inputArraySizes = newInputArraySizes;
    outputArraySizes = newOutputArraySizes;
}

int OpenCL_KernelInterface::getNumInputArrays()
{
    return numInputArrays;
}

int OpenCL_KernelInterface::getNumOutputArrays()
{
    return numOutputArrays;
}

int OpenCL_KernelInterface::getInputArraySize(const int& index)
{
    return inputArraySizes[index];
}

int OpenCL_KernelInterface::getOutputArraySize(const int& index)
{
    return outputArraySizes[index];
}

cl::Buffer* OpenCL_KernelInterface::getInputBufferPtr(const int& index)
{
    return &(inputBuffers[index]);
}

cl::Buffer* OpenCL_KernelInterface::getOutputBufferPtr(const int& index)
{
    return &(outputBuffers[index]);
}

cl::Kernel* OpenCL_KernelInterface::getKernelPtr()
{
    return &kernel;
}

// ====================================================

cl::Kernel OpenCL_KernelInterface::getKernel(
    cl::Device* devicePtr,
    cl::Context* contextPtr,
    const std::string& kernelSource, 
    const std::string& kernelName
)
{
    // initialise program sources, takes array with const char* c string and length
    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.length() });

    // build program
    cl::Program program(*contextPtr, sources);

    assert(("Kernel did not build", program.build({ *devicePtr }) == CL_SUCCESS));

    // create kernel object
    cl::Kernel kernel = cl::Kernel(program, kernelName.c_str());
    
    return kernel;
}

// create kernel source given path
std::string OpenCL_KernelInterface::getKernelSource(
    const std::string& kernelPath
)
{
    // create kernel source string
    std::ifstream file(kernelPath);
    std::string input;
    std::string kernelSource;

    while (file >> input)
    {
        input.append(" ");
        kernelSource.append(input);
    }

    return kernelSource;
}
