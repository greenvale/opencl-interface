/* 
    ==============================================================================================
     OpenCL_KernelInterface class source file (William Denny, 12th Aug 2022)
    ==============================================================================================
*/

#include "OpenCL_Interface.hpp"

// empty constructor
OpenCL_KernelInterface::OpenCL_KernelInterface()
{

}

// main constructor
OpenCL_KernelInterface::OpenCL_KernelInterface(
    cl::Device* devicePtr,
    cl::Context* contextPtr,
    const std::string& kernelPath,
    const std::string& kernelName,
    const std::vector<int>& kernelInputArraySizes, 
    const std::vector<int>& kernelOutputArraySizes
)
{
    // get kernel source
    std::string kernelSource = getKernelSource(kernelPath);

    // get kernel
    kernel = getKernel(devicePtr, contextPtr, kernelSource, kernelName);

    // get number of input and output arrays required
    numInputArrays = kernelInputArraySizes.size();
    numOutputArrays = kernelOutputArraySizes.size();

    // get vector input and output array sizes
    inputArraySizes = kernelInputArraySizes;
    outputArraySizes = kernelOutputArraySizes;

    // create input buffers
    inputBuffers = {};
    for (int i = 0; i < numInputArrays; ++i)
    {
        cl::Buffer buffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(float) * inputArraySizes[i]);

        inputBuffers.push_back(buffer);
    }

    // create output buffers
    outputBuffers = {};
    for (int i = 0; i < numOutputArrays; ++i)
    {
        cl::Buffer buffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(float) * outputArraySizes[i]);

        outputBuffers.push_back(buffer);
    }
}

// transfer input arrays into input buffers
void OpenCL_KernelInterface::setInputBuffers(
    cl::CommandQueue* queuePtr,
    const std::vector<float*>& inputArrayPtrs
)
{
    for (int i = 0; i < numInputArrays; ++i)
    {
        (*queuePtr).enqueueWriteBuffer(inputBuffers[i], CL_TRUE, 0, sizeof(float) * inputArraySizes[i], inputArrayPtrs[i]);
    }
}

// set kernel arguments
void OpenCL_KernelInterface::setKernelArgs()
{
    for (int i = 0; i < numInputArrays; ++i)
    {
        kernel.setArg(i, inputBuffers[i]);
    }
    for (int i = 0; i < numOutputArrays; ++i)
    {
        kernel.setArg(i + numInputArrays, outputBuffers[i]);
    }
}

// run kernel
void OpenCL_KernelInterface::runKernel(
    cl::CommandQueue* queuePtr,
    const int& numWorkitems,
    const int& workgroupSize
)
{
    (*queuePtr).enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numWorkitems), cl::NDRange(workgroupSize));
}

// get output buffers
void OpenCL_KernelInterface::getOutputBuffers(
    cl::CommandQueue* queuePtr,
    const std::vector<float*>& outputArrayPtrs
)
{
    for (int i = 0; i < numOutputArrays; ++i)
    {
        (*queuePtr).enqueueReadBuffer(outputBuffers[i], CL_TRUE, 0, sizeof(float) * outputArraySizes[i], outputArrayPtrs[i]);
    }
} 

// (static function) get kernel
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

// (static function) create kernel source given path
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
