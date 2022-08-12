#include <CL/opencl.hpp>

#include "OpenCL_Interface.hpp"

OpenCL_Interface::OpenCL_Interface() 
{
    std::cout << "Empty constructor called" << std::endl;
}

// ====================================================

OpenCL_Interface::OpenCL_Interface(
    const int& platformIndex, 
    const int& deviceIndex
)
{
    // get vector of platforms
    cl::Platform::get(&platforms);

    // abort if no platforms found
    assert(("No platforms found", platforms.size() > 0));

    // get target platform
    platform = platforms[platformIndex];

    // get vector of devices
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // abort if no devices found
    assert(("No devices found", devices.size() > 0));

    // get target device
    device = devices[deviceIndex];

    // Output success to user
    std::cout << "Connected to accelerator device: ";
    std::cout << device.getInfo<CL_DEVICE_VENDOR>() << ", " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

    // create context
    context = cl::Context({ device });

    // create queue
    queue = cl::CommandQueue(context, device);
}

// ====================================================

void OpenCL_Interface::addKernel(
    const std::string& kernelPath, 
    const std::string& kernelName,
    const std::vector<int>& inputArraySizes,
    const std::vector<int>& outputArraySizes
)
{
    // create kernel interface object with input and output dimensions
    OpenCL_KernelInterface kernelInterface = OpenCL_KernelInterface(&device, &context, kernelPath, kernelName, inputArraySizes, outputArraySizes);

    // get kernel source
    //std::string kernelSource = getKernelSource(kernelPath);

    // get kernel
    //kernelInterface.kernel = getKernel(kernelSource, kernelName);

    // create read buffer vector
    kernelInterface.inputBuffers = {}; // empty vector

    for (int i = 0; i < kernelInterface.numInputArrays; ++i)
    {
        int numArrayElements = kernelInterface.inputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_READ_ONLY, sizeof(float) * numArrayElements);

        kernelInterface.inputBuffers.push_back(buffer);
    }

    // create write buffer vector
    kernelInterface.outputBuffers = {}; // empty vector

    for (int i = 0; i < kernelInterface.numOutputArrays; ++i)
    {
        int numArrayElements = kernelInterface.outputArraySizes[i];

        cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * numArrayElements);

        kernelInterface.outputBuffers.push_back(buffer);
    }

    kernelInterfaces.push_back(kernelInterface);
}

// ====================================================

void OpenCL_Interface::runKernel(
    const int& index,
    const int& numElements,
    const int& workgroupSize,
    const std::vector<float*>& inputArrays, 
    const std::vector<float*>& outputArrays
)
{
    // add input to input buffer
    for (int i = 0; i < kernelInterfaces[index].getNumInputArrays(); ++i)
    {
        float* array = inputArrays[i];
        cl::Buffer* inputBufferPtr = kernelInterfaces[index].getInputBufferPtr(i);

        queue.enqueueWriteBuffer(*inputBufferPtr, CL_TRUE, 0, sizeof(float) * kernelInterfaces[index].getInputArraySize(i), array);
    }

    // set kernel arguments
    for (int i = 0; i < kernelInterfaces[index].numInputArrays; ++i)
    {
        kernelInterfaces[index].kernel.setArg(i, kernelInterfaces[index].inputBuffers[i]);
    }
    for (int i = 0; i < kernelInterfaces[index].numOutputArrays; ++i)
    {
        kernelInterfaces[index].kernel.setArg(i + kernelInterfaces[index].numInputArrays, kernelInterfaces[index].outputBuffers[i]);
    }

    // get pointer for kernel
    cl::Kernel* kernelPtr = kernelInterfaces[index].getKernelPtr();

    // run kernel
    queue.enqueueNDRangeKernel(*kernelPtr, cl::NullRange, cl::NDRange(numElements), cl::NDRange(workgroupSize));

    // wait for kernel to finish
    queue.finish();

    // transfer output buffer to output
    for (int i = 0; i < kernelInterfaces[index].numOutputArrays; ++i)
    {
        float* array = outputArrays[i];

        queue.enqueueReadBuffer(kernelInterfaces[index].outputBuffers[i], CL_TRUE, 0, sizeof(float) * kernelInterfaces[index].outputArraySizes[i], array);
    }
}